"""
Detecting Deo hearing itself.

When a reply finishes and the microphone reopens for a follow-up, the last of
Deo's own speech can still be coming out of the speakers -- so it gets recorded,
transcribed, and answered as if the user had said it. Observed in practice:

    Deo: "...What would you like to do next?"      ->  You: next
    Deo: "...just let me know."                    ->  You: Let me know.
    Deo: "...anything else I can do."              ->  You: I can do it.

The primary defence is config.FOLLOWUP_GUARD_SECONDS, which stops the mic
opening into the tail in the first place. This module is the backstop for what
still gets through on a loud speaker or in a reverberant room: compare the
transcript against what Deo just said, and drop it if it is really just an echo.

Kept deliberately conservative. Discarding something the user actually said is a
worse failure than answering an echo, so this only rejects SHORT transcripts
(echo is a sentence tail) that closely match the end of the previous reply.
"""

import difflib
import re

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- so "next" matches
    "next?" and "Let me know." matches "let me know"."""
    if not text:
        return ""
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", text.lower())).strip()


def looks_like_echo(heard: str, spoken: str,
                    similarity: float = 0.75, max_words: int = 8) -> bool:
    """True if `heard` is probably `spoken` coming back through the microphone.

    heard:  what the transcriber produced for the follow-up
    spoken: what Deo said immediately before opening the mic
    """
    heard_norm = _normalize(heard)
    spoken_norm = _normalize(spoken)
    if not heard_norm or not spoken_norm:
        return False

    heard_words = heard_norm.split()

    # Echo is the END of what was just played, so every test below anchors to the
    # end of the reply. Matching anywhere in the reply instead was too loose: it
    # discarded "open it" spoken right after "Sure, I can open it for you", which
    # is a perfectly good command.
    if spoken_norm.endswith(heard_norm):
        return True

    if len(heard_words) > max_words:
        # At this length it is the user talking, and the verbatim check above is
        # the only thing that would have justified dropping it.
        return False

    # Otherwise compare against the TAIL of the reply, which is the part still
    # audible when the mic opens. A slightly longer window than the transcript
    # absorbs transcription slop at the boundary.
    spoken_words = spoken_norm.split()
    window = spoken_words[-(len(heard_words) + 3):]
    if not window:
        return False

    ratio = difflib.SequenceMatcher(None, heard_words, window).ratio()
    if ratio >= similarity:
        return True

    # Echo that the transcriber partially garbled: most of the transcript's words
    # appear in the tail, in order, but it invented or dropped one ("...anything
    # else I can do" coming back as "I can do it").
    #
    # The decisive extra condition is that the match must REACH THE REPLY'S FINAL
    # WORD, because echo is by definition the end of what was just played. That
    # is what keeps a legitimate command which happens to reuse Deo's wording --
    # "open it", right after "I can open it for you" -- from being thrown away.
    if 2 <= len(heard_words) <= 5:
        matcher = difflib.SequenceMatcher(None, window, heard_words)
        blocks = matcher.get_matching_blocks()
        covered = sum(block.size for block in blocks)
        matched_spoken = {block.a + k for block in blocks for k in range(block.size)}
        reaches_end = (len(window) - 1) in matched_spoken
        if reaches_end and covered / len(heard_words) >= 0.65:
            return True

    return False
