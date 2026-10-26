import os
import json
import time
import datetime
import threading
import litellm
import logging
from functions.results import shorten
from functions.read_text import read_text
from config import CHECKPOINT_AT, EMERGENCY_AT

litellm.drop_params = True     # a model that can't do something we ask for (e.g. thinking): leave it out, don't crash
logging.getLogger("LiteLLM").setLevel(logging.ERROR)   # litellm is chatty: only show real errors, no warnings
litellm.suppress_debug_info = True                     # no "Provider List: ..." lines from litellm's bookkeeping

RETRIES = 3                    # tries after the first one, only for errors that may go away by themselves
RETRY_ON = (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
            litellm.InternalServerError, litellm.ServiceUnavailableError)

# what the program says to the model about checkpoints (see ask, and tools/basic/checkpoint.py).
# "[from the program, not the user]": so the model doesn't think you wrote it
CHECKPOINT_REQUEST = """[from the program, not the user] CHECKPOINT. Your context is getting full. Write your checkpoint
now and give it to the checkpoint tool. It replaces your whole conversation: afterwards you only have your system
prompt with this checkpoint in it, so put in everything you need to go on. It also replaces your previous checkpoint
(in your system prompt, if there is one): carry over everything from it that is still true. Call only the checkpoint
tool in this reply. Short bullets, facts not stories, exactly these sections:
## Summary
3-15 sentences: what this whole conversation has been about so far, including what earlier checkpoints covered,
and where things stand now. Written so someone with no context understands it
## Goal
what the user wants, and what "done" means
## Decisions
what was decided, and why
## Facts
what was found out, each with where it comes from
## Done
what is finished, so it is not done twice
## Failed
what was tried and did not work, and why, so it is not tried again
## Open
what is left, and the very next step
## Files
the saved results that matter (results-1, ...: read_result shows them again), and any other files
## Last exchange
the user's last message and your last answer, in short"""

EMERGENCY_NOTE = """
You are in the MIDDLE of a task: say exactly where you are in it, and what you were about to do next."""

GO_ON = """[from the program, not the user] Your conversation was emptied in the middle of the task, after your
checkpoint. Your checkpoint (in your system prompt) says where you were: go on with the task from there. Long results
you saw before are still saved: read_result and search_result show them."""

class Agent:
    def __init__(self, model, api_key, instructions=None, stream=False, effort=None, print_thinking=False,
                 tools=None, skills=None, conversation=None, log_file=None, results_dir=None,
                 checkpoint_file=None, context_limit=None):
        self.model = model
        self.api_key = api_key
        self.instructions = instructions       # who the agent is (the soul): the first part of its system prompt
        self.tools = tools or {}               # the tools it may use: name -> {"schema", "run"} (functions/load_tools.py)
        self.skills = skills or {}             # the skills it may load: name -> {"description", "path"}
                                               # (functions/load_skills.py). the menu goes into the system prompt
        self.stream = stream                   # True = the answer is printed piece by piece while it arrives
        self.effort = effort                   # how hard the model thinks: "low", "medium", "high". None = its default
        self.print_thinking = print_thinking   # True = the thinking is printed too. False = only the answer
        self.printing = None                   # what kind of text was printed last: "THINKING", "ANSWER" or None
        self.tokens = 0                        # how big the conversation is, in tokens, as of the last answer.
                                               # 0 = no answer yet. the next call sends this + the new message
        self.log_file = log_file               # where every message is written, one line each (log.jsonl). None = no log
        self.logged_system = None              # the system prompt as it was last written into the log
        self.results_dir = results_dir         # where tool results too long for the context are saved, in parts
                                               # (functions/results.py). None = never saved, always given whole
        self.checkpoint_file = checkpoint_file # checkpoint.md: the model's notes on where it stands, in the system
                                               # prompt. written by the checkpoint tool. None = no checkpoints
        self.context_limit = context_limit     # tokens the conversation may have. None = no checkpoints are asked for
        self.checkpoint_asked = None           # None, or why a checkpoint was asked for: "end" (a task ended above
                                               # CHECKPOINT_AT) or "emergency" (above EMERGENCY_AT in a task)
        self.cleared = False                   # True right after the checkpoint tool emptied the conversation
        self.conversation = conversation or []
        self.refresh_system()  # always. no instructions = no system message (see refresh_system)
        self.counted = len(self.conversation)  # how many messages self.tokens covers. the ones after it (tool results,
                                               # your next message) are not counted yet: size() adds them

    # --- the log: every message, one json line each, only ever added to. only for you to read ---
    def log(self, message):
        """write one message into the log, at the end of the file. called right after a message arrived (your
        question, the model's answer, a tool's result). before it, the system prompt, but only if it changed since
        it was last logged: so the log shows which system prompt the messages after it were sent with.
        the log never looks at the conversation list again: whatever happens to it later (a message taken back,
        state.json deleted), what was said stays in the log"""
        if not self.log_file:                  # no log for this agent
            return
        now = datetime.datetime.now().isoformat(timespec="seconds")   # e.g. 2026-10-23T20:31:05
        lines = []
        system = self.conversation[0]["content"] if self.conversation and self.conversation[0]["role"] == "system" else ""
        if system != self.logged_system:       # 1. the system prompt: new, or changed since it was last logged
            lines.append({"time": now, "kind": "system", "content": system})
            self.logged_system = system
        lines.append({"time": now, "kind": "message", "message": message})   # 2. the message itself
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)   # the folder appears with the first line
        with open(self.log_file, "a", encoding="utf-8") as f:        # "a" = add at the end, never replace.
            for line in lines:                                       # opened and closed every time, so the file is
                f.write(json.dumps(line, ensure_ascii=False) + "\n") # never held open (a /rename works)

    def build_system(self):
        """put the system prompt together from its parts, always in the same order, and return it as text.
        the same parts always give exactly the same text, so the provider's cache stays valid"""
        system = self.instructions or ""  # 1. who the agent is. none given: start with an empty text
        if self.skills:                   # 2. the skills menu: only name + one line each. the instructions
                                          #    themselves are loaded on demand, with the load_skill tool
            system += ("\n\nYou have skills: step-by-step instructions for certain kinds of tasks. When a task "
                       "matches a skill, first call load_skill with its name, then follow what it returns.\nSkills:")
            for name, skill in self.skills.items():
                system += "\n- " + name + ": " + skill["description"]
        checkpoint = read_text(self.checkpoint_file) if self.checkpoint_file else ""   # "" if there is none yet
        if checkpoint:                    # 3. its checkpoint, once it wrote one: where it stands
            system += ("\n\nYour checkpoint: your own notes from earlier in this conversation, written before the "
                       "conversation was emptied to make room. It is where you stand.\n<checkpoint>\n"
                       + checkpoint + "\n</checkpoint>")
        return system.strip()             # strip: no skills and no soul -> "" -> no system message at all

    def refresh_system(self):
        """build the system prompt, and put it in as the first message. an empty prompt means no system
        message at all: an old one is removed"""
        text = self.build_system()
        has_one = bool(self.conversation) and self.conversation[0]["role"] == "system"
        if not text:  # nothing to say: no system message at all
            if has_one:
                self.conversation.pop(0)  # an old one is removed, so it can't stay outdated
            return
        system = {"role": "system", "content": text}
        if has_one:
            self.conversation[0] = system
        else:
            self.conversation.insert(0, system)

    def ask(self, text):
        """send a message, and loop until the model answers without asking for a tool:
        model -> it asks for tools -> run them -> their results go back to the model -> model -> ...
        the size is checked before every model call, and after the final answer. too full: the model is asked for its
        checkpoint (see checkpoint). prints everything while it happens, and returns the final answer as text"""
        question = {"role": "user", "content": text}
        self.conversation.append(question)
        self.log(question)
        final = ""                                 # the task's final answer, once there is one
        tries = 0                                  # how often a checkpoint was asked for and NOT written yet: after
                                                   # 2 ignored requests, the model isn't asked again in this task
        try:
            while True:                            # one round = one model call (+ the tools it asked for)
                if not self.checkpoint_asked and self.too_full(EMERGENCY_AT) and tries < 2:   # right before every
                    tries += 1                     # call: too full to go on with the task? then the next message is
                    self.ask_for_checkpoint("emergency")   # the request, not the next step. (no new request while
                                                           # one is still open: the model is answering that one)
                response = self.call_model()
                response = self.show(response)     # inside the try: when streaming, the connection can also fail
                                                   # here, in the middle of the answer. afterwards: the finished response
                message = response.choices[0].message   # the answer: text, thinking, tool calls
                answer = message.model_dump(exclude_none=True)   # the WHOLE message as a dict
                self.conversation.append(answer)
                self.log(answer)
                self.tokens = self.count_tokens(response)   # the conversation grew: how big is it now?
                self.counted = len(self.conversation)       # that count covers every message up to here

                if message.tool_calls:             # it wants tools: run them. their results must come right after
                    self.run_tools(message.tool_calls)   # its request, so this happens first, whatever comes next
                    if self.cleared:               # it called checkpoint: the conversation is empty now
                        self.cleared = False
                        self.counted = len(self.conversation)   # the checkpoint tool counted the new, short one
                        tries = 0                  # it did write one: a long task may need another one later
                        why = self.checkpoint_asked
                        self.checkpoint_asked = None
                        if why == "end":           # the task was finished before the checkpoint: done
                            return final
                        self.tell_model(GO_ON)     # in the middle of a task: go on from the checkpoint
                        continue
                    if not any(call.function.name == "checkpoint" for call in message.tool_calls):
                        self.checkpoint_asked = None   # it went on with other tools: a request it had was ignored.
                                                       # (it called checkpoint but it failed, e.g. empty: the request
                                                       # stays, so it may call it again)
                    continue                       # ask the model again, with the tool results. (too full now? the
                                                   # check at the top of the loop asks for the checkpoint first)

                if not self.checkpoint_asked:      # no tools: a text answer. if it isn't a reply to a checkpoint
                    final = message.content or ""  # request, it is the task's final answer
                if self.too_full(CHECKPOINT_AT) and tries < 2:   # the task is done: checkpoint now, if full enough
                    tries += 1
                    self.ask_for_checkpoint("end")
                    continue
                self.checkpoint_asked = None       # (asked twice and it didn't: given up for this task)
                return final
        except Exception:
            if self.conversation[-1] is question:  # it failed before anything happened: take the question back,
                self.conversation.pop()            # so the list is as before. (if tools already ran, their work stays)
            raise                                  # and let the caller know. main.py prints it

    # --- checkpoints: when the conversation gets too full, the model writes down where it stands (the tool does it) ---
    def size(self):
        """how big the conversation is right now, in tokens. self.tokens is exact (the provider counted it), but only
        up to the last answer. what came after it (tool results, your message, the program's) is estimated and added"""
        since = self.conversation[self.counted:]   # the messages self.tokens doesn't cover yet
        if not since:
            return self.tokens
        return self.tokens + litellm.token_counter(model=self.model, messages=since)

    def too_full(self, share):
        """is the conversation above this share of the context limit? (no limit: never)"""
        return bool(self.context_limit) and self.size() >= share * self.context_limit

    def ask_for_checkpoint(self, why):
        """the next message to the model: write your checkpoint. why: "end" (the task is done) or "emergency" (in the
        middle of a task: then it also has to say where it is in the task)"""
        self.checkpoint_asked = why
        print("\n[" + format(self.size(), ",") + " of " + format(self.context_limit, ",") + " tokens used: asking for "
              + ("an emergency " if why == "emergency" else "a ") + "checkpoint]")
        self.tell_model(CHECKPOINT_REQUEST + (EMERGENCY_NOTE if why == "emergency" else ""))

    def tell_model(self, text):
        """a message from the program to the model (not from you): into the conversation and the log"""
        message = {"role": "user", "content": text}
        self.conversation.append(message)
        self.log(message)

    # --- tools: every tool call of one answer runs in its own thread, all at the same time ---
    def run_tools(self, tool_calls):
        """run all tool calls of one answer at the same time, wait until every one is done, then put the results
        into the conversation, in the order the model asked for them. each result carries the id of its call:
        that id is the only way the model knows which result belongs to which call"""
        results = {}                               # call id -> its result. every thread writes into this one dict
        threads = []
        for call in tool_calls:                    # 1. start one thread per call. they all run at once
            print("\nTOOL CALL: " + call.function.name + "(" + (call.function.arguments or "") + ")")
            thread = threading.Thread(target=self.run_tool, args=(call, results), daemon=True)
            thread.start()                         #    daemon: a running tool can't keep the program from ending
            threads.append(thread)
        for thread in threads:                     # 2. wait until EVERY one is done. they finish in any order,
            thread.join()                          #    that's why they write into a dict by id, not into a list
        for call in tool_calls:                    # 3. in the order the model ASKED, not the order they finished
            result = results.get(call.id, "error: the tool gave no result")   # never missing: see run_tool
            result = shorten(self.results_dir, result)   # too long? saved into results/, the model gets a preview
            print("TOOL RESULT: " + call.function.name + " -> " + result)
            tool_message = {"role": "tool",
                            "tool_call_id": call.id,   # which call this answers: must match exactly
                            "name": call.function.name,
                            "content": result}
            self.log(tool_message)
            if not self.cleared:                   # the checkpoint tool emptied the conversation: the request this
                self.conversation.append(tool_message)   # answers is gone with it, so the result can't go in
                                                         # (a result without its request is rejected). it is logged

    def run_tool(self, call, results):
        """run ONE tool call (in its own thread) and put its result into results[call.id]. it never fails:
        every problem becomes an error text for the model, so every call always gets an answer. (a call without
        an answer breaks the whole conversation: the provider rejects every request after it)"""
        name = call.function.name
        try:
            if name not in self.tools:             # the model asked for a tool it doesn't have
                raise Exception("there is no tool called '" + str(name) + "'. yours: " + (", ".join(self.tools) or "none"))
            args = json.loads(call.function.arguments or "{}")   # the arguments arrive as json TEXT: make a dict
            result = str(self.tools[name]["run"](self, **args))  # run(agent, expression=...). always text
        except Exception as e:                     # wrong arguments, broken json, the tool crashed: tell the model,
            result = "error: " + type(e).__name__ + ": " + str(e)   # so it can try again
        results[call.id] = result

    def count_tokens(self, response):
        """how big the conversation is now, in tokens: everything that was sent + what the model wrote back,
        as the provider counted it (response.usage). exact, and free: it comes with every answer.
        no count from the provider (some local models): litellm estimates it from the conversation instead"""
        usage = getattr(response, "usage", None)            # None if the response has no usage at all
        if usage and usage.total_tokens:                     # total = prompt_tokens (sent) + completion_tokens (written)
            return usage.total_tokens
        return litellm.token_counter(model=self.model, messages=self.conversation)   # an estimate, good enough

    def call_model(self):
        """one model call. errors that may go away by themselves (busy, network) are tried again, waiting
        longer each time. errors that won't (wrong key, bad request) are raised right away.
        streaming: this returns as soon as the answer STARTS to arrive. a failure after that is not retried
        here (half of the answer is printed already): it shows up in show(), and ask() handles it"""
        wait = 1                               # seconds before the first retry: 1, 2, 4
        for attempt in range(RETRIES + 1):
            try:
                return litellm.completion(model=self.model,
                                          api_key=self.api_key,
                                          messages=self.conversation,
                                          tools=[t["schema"] for t in self.tools.values()] or None,
                                                   # what the model may ask for. no tools: None
                                          stream=self.stream,
                                          stream_options={"include_usage": True} if self.stream else None,
                                                   # streaming: also send the token count, in one last chunk.
                                                   # (without streaming it always comes, and this must be None)
                                          reasoning_effort=self.effort,
                                          cache_control_injection_points=[{"location": "message", "index": -1}],
                                          timeout=120)
            except RETRY_ON as e:
                if attempt == RETRIES:         # the last try failed too: give up
                    raise
                print("[Error calling the model - " + type(e).__name__ + ", trying again in " + str(wait) + " s]")
                time.sleep(wait)
                wait *= 2

    def show(self, response):
        """print the model's thinking and answer, and return the finished response (the answer is in
        .choices[0].message, the token count in .usage).
        not streaming: the whole answer arrives at once, in one response, and is printed at once.
        streaming: the response is a stream of small pieces (chunks). each is printed the moment it arrives,
        then all of them are glued back together into one finished response, the same as without streaming"""
        self.printing = None                               # a new answer: nothing printed for it yet

        if not self.stream:
            message = response.choices[0].message          # message = the whole answer
            self.print_part("THINKING", getattr(message, "reasoning_content", None))   # None if it didn't think
            self.print_part("ANSWER", message.content)
            print()                                        # end the last line
            return response

        chunks = []                                        # every chunk, to glue the full message together at the end
        for chunk in response:                             # waits for each next chunk while the model writes
            chunks.append(chunk)
            if not chunk.choices:                          # some chunks are only bookkeeping, e.g. the last one
                continue                                   # with the token count. kept in chunks, nothing to print
            delta = chunk.choices[0].delta                 # delta = only what is new in this chunk
            self.print_part("THINKING", getattr(delta, "reasoning_content", None))   # a piece of thinking, or None
            self.print_part("ANSWER", delta.content)       # a piece of the answer, or None
        print()                                            # the answer is complete: end its line
        return litellm.stream_chunk_builder(chunks, messages=self.conversation)   # litellm glues the chunks into
                                                           # one response. messages: if no chunk had a token count,
                                                           # litellm counts the conversation itself

    def print_part(self, kind, text):
        """print one piece of THINKING or ANSWER text. a title appears whenever the kind changes"""
        if not text:                                       # no text of this kind in this piece: nothing to do
            return
        if kind == "THINKING" and not self.print_thinking: # thinking is hidden: skip it (it is still stored)
            return
        if kind != self.printing:                          # a different kind than the last piece: title first
            if self.printing:                              # something was printed before: an empty line between
                print("\n")
            print(kind + ":")
            self.printing = kind                           # remember: now we are in this kind
        print(text, end="", flush=True)                    # end="": no new line after each piece.
                                                           # flush: show it right now, not when python feels like it
