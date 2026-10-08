"""
Shared cloud-provider client helpers for Deo's hybrid local/cloud setup.

Every getter here is lazy (constructs on first use, not at import time) and
cached, and every getter returns None instead of raising if the provider
isn't configured or the SDK isn't installed. Callers are expected to treat
None as "fall back to local" -- this file is the one place that knows how
to construct each client, so brain.py, stt.py, and tts.py don't each
reinvent that check.
"""

from . import config

_groq_client = None
_groq_unavailable = False  # sticky -- once construction fails, stop retrying every call


def get_groq_client():
    """Returns a shared Groq client, or None if GROQ_API_KEY isn't set, the
    `groq` package isn't installed, or client construction failed for any
    other reason. Never raises."""
    global _groq_client, _groq_unavailable

    if _groq_client is not None:
        return _groq_client
    if _groq_unavailable or not config.GROQ_API_KEY:
        return None

    try:
        from groq import Groq
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
        return _groq_client
    except Exception as e:
        print(f"[providers] Groq client unavailable, staying local-only: {e}")
        _groq_unavailable = True
        return None
