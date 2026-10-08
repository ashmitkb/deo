"""
Wraps chat + tool-calling for Deo, with Groq (cloud) as the primary LLM and
Ollama (local) as an automatic fallback.

DESIGN: self.history is kept in ONE canonical internal shape, independent
of either provider's exact wire format:
    {"role": "user"/"assistant"/"system", "content": str_or_None}
    {"role": "assistant", "content": str_or_None, "tool_calls": [
        {"id": str, "name": str, "arguments": dict}, ...
    ]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": json_str}

Each provider's exact message format (Groq/OpenAI-style vs Ollama's looser
style) is generated FRESH from this canonical history on every single call,
via _to_openai_messages()/_to_ollama_messages(). This is what makes true
mid-conversation fallback possible: if Groq fails on turn 5 after handling
turns 1-4 fine, Ollama can still pick up that exact conversation correctly,
because neither provider's response ever gets stored in its own raw shape.

Grounding rule (unchanged): tool functions return {"success": bool,
"message": str}. That full dict (as JSON) is what gets fed back to the LLM
as the tool result -- not a vague summary -- so the model cannot "round up"
a failure into a success.

STREAMING: ask_streaming() yields the reply a sentence at a time as the model
writes it, so tts.py can start speaking before generation has finished --
that overlap is the bulk of the per-turn latency. ask() is still here and
still returns the whole string; it just drains the stream. Streaming makes one
thing genuinely harder: a spoken sentence cannot be unspoken, so text has to be
proven not-reasoning BEFORE it is emitted rather than stripped afterwards.
_SpeechGuard is what enforces that.

Both providers are stateless, so the entire history is re-sent on every call.
_trim_history() therefore caps it -- otherwise each turn is slower and more
expensive than the last, until it eventually overflows the context window.
"""

import json
import os
import re
import time
import uuid
import ollama
from . import config
from . import Memory
from . import providers
from .tools import TOOL_SCHEMAS, TOOL_REGISTRY
from . import permissions, gamemode

# Well-formed <think>...</think> blocks.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Defensive fallback: an opening <think> tag with NO matching close (e.g. the
# model's reasoning got cut off mid-thought). Without this, everything from
# that point to the end of the message -- raw chain-of-thought -- slips
# through to TTS and the GUI transcript untouched.
_THINK_UNCLOSED = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
# Defensive fallback #2, confirmed happening in practice with qwen3 on the
# tool-call follow-up turn specifically: the opening <think> goes missing
# but a stray closing </think> and the raw reasoning text stay in `content`.
# Content never legitimately contains a literal "</think>" any other way,
# so if one shows up with no matching open tag, everything up to and
# including it is leaked reasoning. Kept as a safety net for both providers
# -- GPT-OSS models on Groq have their own "reasoning" behavior and there's
# no guarantee it can't leak the same way.
_THINK_CLOSE_ONLY = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _strip_thinking(text: str) -> str:
    """Removes <think>...</think> reasoning blocks, however malformed.
    Deo should never speak or display its internal reasoning, from either
    provider."""
    if not text:
        return text

    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    if "<think>" not in cleaned.lower() and "</think>" in cleaned.lower():
        cleaned = _THINK_CLOSE_ONLY.sub("", cleaned)
    cleaned = cleaned.strip()

    if cleaned != text.strip():
        print("[brain] stripped leaked <think> content from model output")

    return cleaned if cleaned else text  # don't return empty if stripping ate everything


# ---------------------------------------------------------------------------
# Canonical history <-> provider wire format converters
# ---------------------------------------------------------------------------

def _to_openai_messages(history: list) -> list:
    """Groq's API is OpenAI-compatible: tool_calls need an explicit id/type
    wrapper and arguments as a JSON string, and tool results need a
    tool_call_id to pair back up."""
    out = []
    for m in history:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                    }
                    for tc in m["tool_calls"]
                ],
            })
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def _to_ollama_messages(history: list) -> list:
    """Ollama's tool-calling is looser: no id/type wrapper needed, arguments
    stay as a native dict, and tool results don't need a tool_call_id."""
    out = []
    for m in history:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": [
                    {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in m["tool_calls"]
                ],
            })
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def _normalize_groq_response(message) -> dict:
    """message is the Groq SDK's response object (attribute access, not a
    dict) -- normalize it into the canonical shape."""
    tool_calls = []
    for call in (message.tool_calls or []):
        tool_calls.append({
            "id": call.id,
            "name": call.function.name,
            "arguments": json.loads(call.function.arguments),
        })
    return {"content": message.content, "tool_calls": tool_calls}


def _normalize_ollama_response(message: dict) -> dict:
    tool_calls = []
    for call in (message.get("tool_calls") or []):
        tool_calls.append({
            # Ollama doesn't always include a call id -- synthesize one so
            # the canonical history always has something to key off of.
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],  # already a dict from Ollama
        })
    return {"content": message.get("content"), "tool_calls": tool_calls}


def _parse_arguments(raw):
    """Streamed tool-call arguments arrive as JSON text in fragments. If the
    reassembled string doesn't parse, hand the tool an empty dict -- it will
    report a real failure, which is far better than this raising mid-turn."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        print(f"[brain] could not parse streamed tool arguments: {raw!r}")
        return {}


# ---------------------------------------------------------------------------
# Streaming safety
# ---------------------------------------------------------------------------

_ACTION_CLAIM = re.compile(
    r"\b(i'?m|i am|i'?ve|i have|i'?ll|i will|let me)\s+(now\s+|just\s+|go ahead and\s+)?"
    r"(open|launch|start|clos|set|creat|run|delet|mov|search|look|check|play|turn)\w*"
    r"|\b(has|have) been\s+(opened|closed|set|launched|created|deleted|moved|started)\b"
    r"|\b(is|are)\s+((now|already|still)\s+)?(open|opening|closed|closing|running|starting|launching|set|playing)\b"
    r"|\b(timer|reminder|alarm) (is |has been )?set\b"
    r"|\b\w+'s ((now|already|still) )?(open|running|closed)\b",
    re.IGNORECASE)


def claims_action(text: str) -> bool:
    """Does a reply say Deo did (or is doing) something? Used to catch the
    model narrating an action it never actually took."""
    return bool(text) and bool(_ACTION_CLAIM.search(text))


class _SpeechGuard:
    """Turns a stream of raw model deltas into sentences that are safe to speak.

    Stripping reasoning after the fact (see _strip_thinking) is enough when the
    whole message is in hand. Streaming isn't so forgiving -- a sentence handed
    to TTS is already being spoken -- so anything that might turn out to be a
    <think> block is held back here until it resolves one way or the other.
    Nothing is emitted mid-tag either: a buffer ending in "<thi" waits.
    """

    def __init__(self):
        self._buf = ""       # not yet emitted
        self._raw = ""       # everything ever fed, for the flush-time fallback
        self._out = []       # every sentence emitted, in order

    def feed(self, delta: str) -> list:
        self._raw += delta
        self._buf += delta
        return self._drain(final=False)

    def flush(self) -> list:
        out = self._drain(final=True)
        if not self._out and not out:
            # Everything looked like reasoning (e.g. an unclosed <think>).
            # Fall back to the after-the-fact stripper so the turn isn't silent.
            salvaged = _strip_thinking(self._raw).strip()
            if salvaged and salvaged != self._raw.strip():
                self._out.append(salvaged)
                return [salvaged]
        return out

    @property
    def text(self) -> str:
        return " ".join(self._out).strip()

    def _drain(self, final: bool) -> list:
        """Split the buffer into (speakable now, must keep holding), pull whole
        sentences out of the first part, and keep the rest for next time."""
        # Resolve any complete block first: "a <think>x</think> b" -> "a  b".
        while True:
            low = self._buf.lower()
            open_at = low.find(_OPEN_TAG)
            close_at = low.find(_CLOSE_TAG)

            if open_at != -1 and close_at > open_at:
                self._buf = self._buf[:open_at] + self._buf[close_at + len(_CLOSE_TAG):]
                continue

            if close_at != -1 and open_at == -1:
                # A close with no open: everything before it was reasoning that
                # lost its opening tag. Whatever already went out is gone, but
                # this stops the remainder.
                print("[brain] dropped leaked reasoning ahead of a stray </think>")
                self._buf = self._buf[close_at + len(_CLOSE_TAG):]
                continue

            break

        if open_at != -1:
            # An opening tag with no close yet: everything from it onward is
            # reasoning-until-proven-otherwise, so hold it.
            speakable, held = self._buf[:open_at], self._buf[open_at:]
        else:
            # Never emit a partial "<thi" as if it were words.
            speakable, held = self._split_partial_tag(self._buf)

        # Imported here to keep the import dependency one-way at module load.
        from .tts import _extract_sentences
        sentences, leftover = _extract_sentences(speakable, final=final and not held)
        self._buf = leftover + held

        out = [s for s in (x.strip() for x in sentences) if s]
        self._out.extend(out)
        return out

    @staticmethod
    def _split_partial_tag(buf: str):
        """If buf ends with a prefix of '<think>' / '</think>', hold that back."""
        for tag in (_CLOSE_TAG, _OPEN_TAG):
            for n in range(len(tag) - 1, 0, -1):
                if buf.lower().endswith(tag[:n]):
                    return buf[:-n], buf[-n:]
        return buf, ""


class Brain:
    def __init__(self, mcp_bridge=None):
        self.ollama_client = ollama.Client(host=config.OLLAMA_HOST)
        system_content = config.SYSTEM_PROMPT
        memory_block = Memory.memory_as_prompt_block()
        if memory_block:
            system_content = system_content + "\n\n" + memory_block
        self.history = [{"role": "system", "content": system_content}]
        self.mcp_bridge = mcp_bridge  # optional MCPToolBridge, see mcp_client.py
        self._reasoning_format_ok = True  # cleared if a model rejects the parameter
        self._restore_conversation()

    # ---- conversation that survives restarts ----
    # Every restart used to wipe the conversation: after one, "run the code"
    # got "which code?" because Deo had forgotten writing it a minute earlier.
    # Only the plain back-and-forth is kept -- tool calls and results are
    # bulky and are only meaningful inside the turn that made them.

    def _restore_conversation(self):
        path = getattr(config, "CONVERSATION_FILE", "")
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
        except Exception as e:
            print(f"[brain] couldn't read saved conversation: {e}")
            return
        max_age = getattr(config, "CONVERSATION_RESUME_HOURS", 6) * 3600
        if time.time() - saved.get("saved_at", 0) > max_age:
            return
        messages = [m for m in saved.get("messages", [])
                    if m.get("role") in ("user", "assistant") and m.get("content")]
        if messages:
            self.history += [{"role": m["role"], "content": m["content"]} for m in messages]
            print(f"[brain] picked up the conversation where it left off ({len(messages)} messages)")

    def record_interrupted(self, spoken: str):
        """Called when you cut Deo off mid-reply. The reply is normally only
        stored once it's finished, so an interrupted one vanished: "okay,
        you can continue" got "continue what?". Store what was actually said,
        marked as cut off, so the model can pick up from that point."""
        spoken = (spoken or "").strip()
        if not spoken:
            return
        self.history.append({"role": "assistant",
                             "content": spoken + " [I was interrupted here by the user.]"})
        self._trim_history()
        self._save_conversation()

    def _save_conversation(self):
        path = getattr(config, "CONVERSATION_FILE", "")
        if not path:
            return
        keep = getattr(config, "CONVERSATION_KEEP_MESSAGES", 12)
        messages = [{"role": m["role"], "content": m["content"]} for m in self.history[1:]
                    if m.get("role") in ("user", "assistant") and m.get("content")
                    and not m.get("tool_calls")][-keep:]
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"saved_at": time.time(), "messages": messages}, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[brain] couldn't save conversation: {e}")

    def _tool_schemas(self) -> list:
        schemas = list(TOOL_SCHEMAS)
        if self.mcp_bridge:
            schemas += self.mcp_bridge.schemas()
        return schemas

    def _run_tool(self, fn_name: str, fn_args: dict) -> dict:
        """Run a tool and cap what it hands back. One `list_directory` of
        Downloads (346 files) blew the model's context in a single call and
        killed the turn, so long results are trimmed with a note telling the
        model to narrow the request instead."""
        result = self._run_tool_uncapped(fn_name, fn_args)
        cap = getattr(config, "MAX_TOOL_RESULT_CHARS", 6000)
        message = result.get("message")
        if isinstance(message, str) and len(message) > cap:
            result = dict(result)
            result["message"] = (
                message[:cap]
                + f"\n...[cut: {len(message) - cap} more characters not shown. If what you need "
                  f"isn't above, use a narrower tool call -- e.g. search_files for a name "
                  f"instead of listing a whole folder.]"
            )
        return result

    def _run_tool_uncapped(self, fn_name: str, fn_args: dict) -> dict:
        # Every tool -- built-in and MCP alike -- passes the permission gate
        # first: destructive actions need a click in a Windows dialog, and a
        # few things are never done at all. See deo/permissions.py.
        refusal = permissions.check(fn_name, fn_args or {})
        if refusal is not None:
            return refusal

        fn = TOOL_REGISTRY.get(fn_name)
        if fn:
            try:
                result = fn(**fn_args)
                # enforce the contract -- if a tool forgets to return the right shape,
                # don't let that silently become an assumed success downstream
                if not isinstance(result, dict) or "success" not in result:
                    return {
                        "success": False,
                        "message": f"Tool '{fn_name}' returned a malformed result: {result!r}",
                    }
                return result
            except Exception as e:
                return {"success": False, "message": f"Tool '{fn_name}' raised an exception: {e}"}

        if self.mcp_bridge and self.mcp_bridge.has_tool(fn_name):
            return self.mcp_bridge.call_tool(fn_name, fn_args)

        return {"success": False, "message": f"Unknown tool: {fn_name}"}

    def _trim_history(self):
        """Cap the conversation at config.MAX_HISTORY_MESSAGES behind the system
        prompt. The system prompt (which carries long-term memory) always
        survives; the oldest exchanges are dropped first."""
        limit = getattr(config, "MAX_HISTORY_MESSAGES", 0)
        if limit <= 0 or len(self.history) <= limit + 1:
            return

        system, rest = self.history[0], self.history[1:]
        tail = rest[-limit:]
        # A tool result whose originating assistant message got trimmed away is
        # an error on the Groq/OpenAI side ("tool_call_id not found"), so drop
        # any orphans left at the front of the window.
        while tail and tail[0]["role"] == "tool":
            tail.pop(0)
        self.history = [system] + tail

    # ---- provider calls (non-streaming) ----

    def _groq_kwargs(self, tools, stream: bool) -> dict:
        kwargs = dict(
            model=config.GROQ_LLM_MODEL,
            messages=_to_openai_messages(self.history),
        )
        if stream:
            kwargs["stream"] = True
        if tools:
            kwargs["tools"] = tools
        fmt = getattr(config, "GROQ_REASONING_FORMAT", "")
        if fmt and self._reasoning_format_ok:
            kwargs["reasoning_format"] = fmt
        return kwargs

    def _call_groq(self, tools) -> dict:
        client = providers.get_groq_client()
        if not client:
            raise RuntimeError("Groq client not configured (missing key, bad install, or disabled)")
        response = client.chat.completions.create(**self._groq_kwargs(tools, stream=False))
        return _normalize_groq_response(response.choices[0].message)

    def is_addressed_to_me(self, heard: str, last_reply: str) -> bool:
        """Was a follow-up actually said TO Deo? After a reply the mic stays
        open for a few seconds, and it was answering half of a phone call:
        "How was it? Nice." / "I'm getting ready". One tiny local-model call
        (~0.3s, a single word out) tells conversation-with-Deo from
        conversation-with-someone-else. Errs towards yes if it can't decide."""
        prompt = (
            "A voice assistant called Deo just replied to its user. Then its microphone "
            "picked up more speech. Decide whether that speech is said TO Deo (a request, "
            "question, or reply meant for the assistant) or to someone else -- a phone "
            "call, a person in the room, talking to oneself.\n\n"
            f"Deo's last reply: \"{last_reply[:300]}\"\n"
            f"Speech picked up: \"{heard}\"\n\n"
            "Answer with exactly one word: DEO or OTHER."
        )
        try:
            response = self.ollama_client.chat(
                model=gamemode.current_model(),
                messages=[{"role": "user", "content": prompt}],
                think=False,
                keep_alive=config.OLLAMA_KEEP_ALIVE,
                options={"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": 4, "temperature": 0},
            )
            answer = (response["message"]["content"] or "").strip().upper()
            return not answer.startswith("OTHER")
        except Exception as e:
            print(f"[brain] addressee check failed ({e}), assuming it was for Deo")
            return True

    def _call_ollama(self, tools) -> dict:
        kwargs = dict(
            model=gamemode.current_model(),
            messages=_to_ollama_messages(self.history),
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            options={"num_ctx": config.OLLAMA_NUM_CTX},
            think=False,
        )
        if tools:
            kwargs["tools"] = tools
        response = self.ollama_client.chat(**kwargs)
        return _normalize_ollama_response(response["message"])

    def _chat(self, tools) -> dict:
        """Tries Groq first (if enabled), falls back to local Ollama on ANY
        failure -- rate limit, network error, bad/stale model name, whatever.
        This is the one place the fallback chain actually happens."""
        if config.USE_CLOUD_LLM:
            try:
                return self._call_groq(tools)
            except Exception as e:
                print(f"[brain] Groq call failed ({e}), falling back to local Ollama")
        return self._call_ollama(tools)

    # ---- provider calls (streaming) ----
    # Each of these yields raw text deltas and *returns* the canonical
    # {"content", "tool_calls"} dict, so callers use `yield from`.

    def _stream_groq(self, tools, progress):
        client = providers.get_groq_client()
        if not client:
            raise RuntimeError("Groq client not configured (missing key, bad install, or disabled)")

        content = []
        pending = {}  # tool-call index -> partial {id, name, arguments}

        for chunk in client.chat.completions.create(**self._groq_kwargs(tools, stream=True)):
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                content.append(text)
                progress["emitted"] = True
                yield text
            for call in (getattr(delta, "tool_calls", None) or []):
                slot = pending.setdefault(
                    getattr(call, "index", 0), {"id": None, "name": "", "arguments": ""}
                )
                if getattr(call, "id", None):
                    slot["id"] = call.id
                fn = getattr(call, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        return {
            "content": "".join(content),
            "tool_calls": [
                {
                    "id": slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                    "name": slot["name"],
                    "arguments": _parse_arguments(slot["arguments"]),
                }
                for _, slot in sorted(pending.items())
                if slot["name"]
            ],
        }

    def _stream_ollama(self, tools, progress):
        kwargs = dict(
            model=gamemode.current_model(),
            messages=_to_ollama_messages(self.history),
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            options={"num_ctx": config.OLLAMA_NUM_CTX},
            think=False,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        content = []
        tool_calls = []
        for chunk in self.ollama_client.chat(**kwargs):
            message = chunk["message"]
            text = message.get("content")
            if text:
                content.append(text)
                progress["emitted"] = True
                yield text
            # Ollama sends each tool call whole rather than in fragments.
            for call in (message.get("tool_calls") or []):
                tool_calls.append({
                    "id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": call["function"]["name"],
                    "arguments": _parse_arguments(call["function"]["arguments"]),
                })

        return {"content": "".join(content), "tool_calls": tool_calls}

    def _stream_chat(self, tools):
        """Streaming twin of _chat(). Same Groq-then-Ollama fallback, with one
        extra rule: once a delta has been handed out it may already be audible,
        so a failure after that point cannot be retried on the other provider
        without Deo repeating itself. In that case the partial reply stands."""
        progress = {"emitted": False}

        if config.USE_CLOUD_LLM:
            try:
                return (yield from self._stream_groq(tools, progress))
            except Exception as e:
                if progress["emitted"]:
                    print(f"[brain] Groq stream broke mid-reply ({e}) -- keeping what was said")
                    return {"content": "", "tool_calls": []}

                # A model that doesn't accept reasoning_format rejects it as an
                # ordinary 400, and the message wording is not something to rely
                # on. So don't pattern-match it: retry once without the
                # parameter, and only keep it switched off if that retry is what
                # fixed things. Otherwise the parameter was innocent and a
                # single rate limit would have silently cost Deo its protection
                # against spoken reasoning for the rest of the session.
                if self._reasoning_format_ok and getattr(config, "GROQ_REASONING_FORMAT", ""):
                    self._reasoning_format_ok = False
                    try:
                        result = yield from self._stream_groq(tools, progress)
                        print("[brain] this model rejects reasoning_format -- dropped for this session")
                        return result
                    except Exception as retry_error:
                        self._reasoning_format_ok = True  # not the parameter's fault
                        e = retry_error
                        if progress["emitted"]:
                            return {"content": "", "tool_calls": []}

                print(f"[brain] Groq call failed ({e}), falling back to local Ollama")

        return (yield from self._stream_ollama(tools, progress))

    # ---- public API ----

    def ask_streaming(self, user_text: str):
        """Generator yielding the reply one speakable sentence at a time, in
        order. Callers MUST drain it -- history is only updated at the end.
        ask() is the drain-it-all-for-me wrapper."""
        permissions.note_request(user_text)   # shown in any permission dialog this turn
        if not config.LLM_STREAMING:
            reply = self.ask(user_text)
            if reply:
                yield reply
            return

        self.history.append({"role": "user", "content": user_text})

        final_text = ""
        tools_called = False
        corrected = False
        # finally, not except: if the stream dies or the caller stops consuming
        # part-way, the turn still has to leave history in a coherent state --
        # a user message with no assistant reply behind it would corrupt every
        # subsequent call.
        try:
            max_rounds = max(1, getattr(config, "MAX_TOOL_ITERATIONS", 1))

            # Loop rather than "one round of tools, then phrase the answer". A
            # single round can't chain: the model has to see what open_app or
            # get_latest_youtube_video actually returned before it can decide
            # what to do next. Looping is what lets a request like "find the
            # latest video from X and put it on my other screen" work without a
            # bespoke combined tool for that exact phrasing.
            for round_number in range(1, max_rounds + 1):
                last_round = round_number == max_rounds
                # On the final permitted round, withhold the tools so the model
                # has no option but to answer with what it already knows.
                tools = None if last_round else self._tool_schemas()

                guard = _SpeechGuard()
                result = yield from self._guarded_stream(tools, guard)

                if not result["tool_calls"]:
                    # Claimed an action but never called a tool -- the 9B model
                    # said "I'm opening amazon.in in Brave" and opened nothing.
                    # Once, hand it back: do it for real, or admit it can't.
                    if (not tools_called and not corrected and not last_round
                            and len(guard.text) < 200 and claims_action(guard.text)):
                        # (short replies only: confirmations are short; a long
                        # answer mentioning "the shop is open" is just a fact)
                        corrected = True
                        print(f"[brain] claimed an action without doing it, retrying: {guard.text[:80]!r}")
                        self.history.append({"role": "assistant", "content": guard.text})
                        self.history.append({"role": "user", "content": (
                            "[Automatic check -- the user can't see this, so never mention it: your "
                            "last reply claimed an action but no tool was called, so nothing happened. "
                            "Call the right tool now. Afterwards reply with at most five words, like "
                            "'Done.' -- no apology, no explanation. If it can't be done, say so in one "
                            "short sentence.]")})
                        continue
                    # The model stopped reaching for tools, so this is the answer.
                    final_text = guard.text
                    break
                tools_called = True

                self.history.append({
                    "role": "assistant",
                    "content": result["content"],
                    "tool_calls": result["tool_calls"],
                })

                for call in result["tool_calls"]:
                    print(f"[tool call] ({round_number}) {call['name']}({call['arguments']})")
                    tool_result = self._run_tool(call["name"], call["arguments"])
                    print(f"[tool result] success={tool_result.get('success')} -- {tool_result.get('message')}")

                    # feed the FULL structured result back, not a paraphrase
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": json.dumps(tool_result),
                    })

                if last_round:
                    # Out of rounds with the model still asking for tools. Keep
                    # whatever it said alongside the call ("let me check that")
                    # rather than returning nothing at all.
                    print(f"[brain] hit the {max_rounds}-round tool limit")
                    final_text = guard.text
        finally:
            self.history.append({"role": "assistant", "content": final_text})
            self._save_conversation()
            self._trim_history()

    def _guarded_stream(self, tools, guard):
        """Runs one provider call, yielding only sentences the guard has
        cleared. Returns that call's canonical result."""
        result = yield from _pump(self._stream_chat(tools), guard)
        for sentence in guard.flush():
            yield sentence
        return result

    def ask(self, user_text: str) -> str:
        """Non-streaming turn: identical behaviour to before, for callers that
        just want the finished string (screen QA, the typed GUI path)."""
        self.history.append({"role": "user", "content": user_text})

        final_text = ""
        max_rounds = max(1, getattr(config, "MAX_TOOL_ITERATIONS", 1))

        for round_number in range(1, max_rounds + 1):
            last_round = round_number == max_rounds
            result = self._chat(None if last_round else self._tool_schemas())

            if not result["tool_calls"]:
                final_text = _strip_thinking(result["content"] or "")
                break

            self.history.append({
                "role": "assistant",
                "content": result["content"],
                "tool_calls": result["tool_calls"],
            })

            for call in result["tool_calls"]:
                print(f"[tool call] ({round_number}) {call['name']}({call['arguments']})")
                tool_result = self._run_tool(call["name"], call["arguments"])
                print(f"[tool result] success={tool_result.get('success')} -- {tool_result.get('message')}")

                self.history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(tool_result),
                })

            if last_round:
                print(f"[brain] hit the {max_rounds}-round tool limit")
                final_text = _strip_thinking(result["content"] or "")

        self.history.append({"role": "assistant", "content": final_text})
        self._save_conversation()
        self._trim_history()
        return final_text


def _pump(stream, guard):
    """Drive a delta-stream generator, yielding guard-approved sentences and
    returning the stream's canonical result dict."""
    while True:
        try:
            delta = next(stream)
        except StopIteration as done:
            return done.value or {"content": "", "tool_calls": []}
        for sentence in guard.feed(delta):
            yield sentence
