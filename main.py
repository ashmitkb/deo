import time
_t0 = time.monotonic()

import os
import sys
import threading
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from deo import config

# Only pay the CUDA DLL-loading cost when Whisper is actually configured to
# use it -- this was running unconditionally before, adding import overhead
# on every single startup even while WHISPER_DEVICE="cpu".
if sys.platform == "win32" and config.WHISPER_DEVICE == "cuda":
    import nvidia.cublas
    import nvidia.cudnn
    import nvidia.cuda_runtime
    import nvidia.nvjitlink
    for mod in (nvidia.cublas, nvidia.cudnn, nvidia.cuda_runtime, nvidia.nvjitlink):
        bin_dir = os.path.join(list(mod.__path__)[0], "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            # add_dll_directory alone isn't enough: CTranslate2 loads cuBLAS
            # with a plain LoadLibrary that only searches PATH, and failed
            # with "cublas64_12.dll is not found" until this was added.
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

import keyboard

from deo.echo import looks_like_echo
from deo.stt import record_until_silence, record_while_held, transcribe, wait_for_push_to_talk
from deo import stt, tts
from deo.brain import Brain
from deo.state import speech_lock, mic_lock, brain_lock
from deo import screen_qa
from deo.tools import move_active_window
from deo.mcp_client import MCPToolBridge
from deo.wakeword import WakeWordListener
from deo.gui import DeoGUI

if config.DEBUG_STARTUP_TIMING:
    print(f"[timing] imports done: {time.monotonic() - _t0:.2f}s")


# What Deo is doing right now, for barge-in: a wake heard while "thinking" or
# "speaking" interrupts; one heard while "listening" is ignored (Deo's already
# listening to you). Kept in step with the GUI status by _status().
_phase = "idle"
_conversation_active = threading.Event()
# Paused from the window or the tray: wake word, gestures and push-to-talk are
# all ignored until resumed. Typed messages still work.
_paused = threading.Event()
_barge_in = threading.Event()
_current_reply = []


def _status(gui: DeoGUI, status: str):
    global _phase
    _phase = status
    gui.push_status(status)


def _ask_and_speak(brain: Brain, gui: DeoGUI, user_text: str):
    """Shared turn logic for every input path (push-to-talk, wake-word,
    typed GUI input, ...). Wrapped in brain_lock so two turns can never
    interleave against the same Brain.history.

    The whole turn is a pipeline rather than three waits in a row: brain.ask_
    streaming() hands over one sentence at a time as the model writes it, and
    tts.speak_stream() starts synthesising and speaking that sentence while the
    rest is still being generated. The transcript bubble lands the moment the
    text is complete, which is well before the speech has finished."""
    with brain_lock:
        _status(gui, "thinking")
        spoken = []

        global _current_reply
        _current_reply = spoken   # read by the barge-in self-echo guard

        def sentence_stream():
            for sentence in brain.ask_streaming(user_text):
                if not spoken:
                    _status(gui, "speaking")  # first sentence => audio starts now
                spoken.append(sentence)
                gui.push_reply_chunk(" ".join(spoken))
                yield sentence
            # Generation finished; audio is very likely still playing.
            reply = " ".join(spoken).strip()
            if reply:
                print(f"{config.ASSISTANT_NAME}: {reply}")
                gui.push_assistant_message(reply)
            else:
                # Rare, but silence with no explanation looks like a hang. It
                # happens when the model produced nothing but reasoning, which
                # the guard in brain.py correctly refuses to speak.
                print(f"[main] {config.ASSISTANT_NAME} produced no speakable reply")
                gui.push_system_message("(no reply to speak -- try asking again)")

        try:
            with speech_lock:
                said = tts.speak_stream(sentence_stream())
            if _barge_in.is_set():
                # Cut off mid-reply: keep what was actually said, so "carry
                # on" / "continue" can resume from there.
                brain.record_interrupted(said)
        except Exception as e:
            # Before this, an error mid-turn (the model rejecting an oversized
            # request, say) escaped to the wake-word thread: nothing was
            # spoken and the window sat on "thinking" forever.
            print(f"[main] turn failed: {e}")
            gui.push_system_message(f"Error: {e}")
            with speech_lock:
                tts.speak("Sorry, something went wrong with that one. Try asking again.")
            spoken.clear()
        finally:
            _status(gui, "idle")
        # Returned so the next turn can tell Deo's own voice from the user's.
        return " ".join(spoken).strip()


def _listen_and_answer(brain: Brain, gui: DeoGUI, start_timeout: float = None,
                       after_speaking: str = "", followup: bool = False):
    """One open-mic turn. Returns what Deo replied, or None if nothing usable was
    said -- which is how the follow-up loop below knows the conversation is over.
    Assumes mic_lock is already held by the caller.

    after_speaking: what Deo said immediately before this, used to recognise its
    own voice coming back through the microphone.
    """
    _status(gui, "listening")
    audio = record_until_silence(start_timeout=start_timeout)
    user_text = transcribe(audio)
    if not user_text:
        return None

    if after_speaking and config.ECHO_REJECTION and looks_like_echo(
        user_text, after_speaking,
        similarity=config.ECHO_SIMILARITY, max_words=config.ECHO_MAX_WORDS,
    ):
        # Deo heard itself. Answering this would have it talking to itself in a
        # loop, which is exactly what happened before FOLLOWUP_GUARD_SECONDS and
        # this check existed.
        print(f"(ignored own echo: {user_text!r})")
        return None

    from deo import speaker
    mine, score = speaker.is_owner(audio)
    if not mine:
        # Someone else (TV, a video, a person nearby). Stay quiet.
        print(f"(not your voice ({score:.2f}), ignoring: {user_text!r})")
        return None

    if followup and config.FOLLOWUP_ADDRESSEE_CHECK and not brain.is_addressed_to_me(
            user_text, after_speaking):
        # Said to someone else (a phone call, a person in the room). Stay quiet
        # and end the follow-up window -- say "hey Deo" to come back.
        print(f"(not for Deo, staying quiet: {user_text!r})")
        return None

    print(f"You: {user_text}")
    gui.push_user_message(user_text)
    return _ask_and_speak(brain, gui, user_text)


def _deo_just_said_its_name() -> bool:
    """True if Deo's own last couple of sentences contained "Deo". Then a wake
    heard during playback is most likely Deo hearing itself through the
    speakers, not you -- so it mustn't cut its own sentence off."""
    from deo.stt import mentions_wake_word
    return any(mentions_wake_word(s) for s in _current_reply[-2:])


def toggle_pause() -> bool:
    """Flip paused/listening; returns the new paused state."""
    if _paused.is_set():
        _paused.clear()
    else:
        _paused.set()
        tts.stop()
    print(f"[main] {'paused -- not listening' if _paused.is_set() else 'listening again'}")
    return _paused.is_set()


def handle_wake(brain: Brain, gui: DeoGUI):
    """Called from the wake-word listener's thread on every verified wake.

    Returns immediately. The conversation runs on its own thread so the
    listener keeps listening the whole time -- that's what makes barge-in
    possible: say "hey Deo" while Deo is talking (or thinking) and it stops
    mid-sentence, says "Yeah?", and listens to you. Before, the conversation
    ran ON the listener's thread, so nothing heard the wake word until Deo had
    completely finished."""
    if _paused.is_set():
        return
    if _phase in ("thinking", "speaking"):
        if _deo_just_said_its_name():
            print("[barge-in] ignored: Deo just said its own name")
            return
        print("[barge-in] interrupted -- listening")
        _barge_in.set()
        tts.stop()
        if _conversation_active.is_set():
            return            # the running conversation picks it up
    elif _conversation_active.is_set():
        return                # already listening to you
    _conversation_active.set()
    threading.Thread(target=_conversation, args=(brain, gui), daemon=True).start()


def _speak_ack(gui: DeoGUI):
    _status(gui, "speaking")
    with speech_lock:
        # Pre-rendered at startup, so this is instant instead of a process
        # spawn (or a network round trip) before the mic even opens.
        tts.speak(config.WAKE_ACK_PHRASE)


def _conversation(brain: Brain, gui: DeoGUI):
    """Wake -> "Yeah?" -> listen -> answer, then keep listening for
    follow-ups (no wake word needed) until you stop talking. A barge-in
    restarts the exchange with a fresh "Yeah?" and a fresh follow-up budget."""
    if not mic_lock.acquire(timeout=5):
        print("[wakeword] microphone busy (push-to-talk?), ignoring this wake")
        _conversation_active.clear()
        return
    try:
        _speak_ack(gui)
        last_said, timeout, turns = config.WAKE_ACK_PHRASE, None, 0
        while True:
            _barge_in.clear()
            reply = _listen_and_answer(brain, gui, start_timeout=timeout,
                                       after_speaking=last_said, followup=turns > 0)
            if _barge_in.is_set():
                # You cut Deo off. Acknowledge, then listen properly -- no
                # timeout, since you've clearly got something to say.
                _speak_ack(gui)
                last_said, timeout, turns = config.WAKE_ACK_PHRASE, None, 0
                continue
            if reply is None:
                if turns == 0:
                    print("(wake: didn't catch anything)")
                break
            turns += 1
            if turns > config.FOLLOWUP_TURNS:
                break
            last_said, timeout = reply, config.FOLLOWUP_LISTEN_SECONDS
            # Let the speakers actually finish before listening again. Without
            # this pause the mic opened into the tail of Deo's own reply and
            # transcribed it as the user.
            time.sleep(config.FOLLOWUP_GUARD_SECONDS)
    except Exception as e:
        print(f"[main] conversation error: {e}")
    finally:
        _status(gui, "idle")
        mic_lock.release()
        _conversation_active.clear()


def handle_gesture(direction: str, gui: DeoGUI):
    """Called from the gesture listener's own background thread whenever a
    swipe is detected. Goes straight to the window-move tool -- no brain.ask()
    round-trip -- so a gesture feels instant rather than waiting on Ollama."""
    if _paused.is_set():
        return
    gui.push_status("thinking")
    result = move_active_window(direction)
    print(f"[gesture] {result['message']}")
    gui.push_system_message(f"Gesture: {direction} -- {result['message']}", island=True)

    if result["success"]:
        with speech_lock:
            tts.speak(f"Moved to the {direction} screen.")
    gui.push_status("idle")


def voice_loop(brain: Brain, gui: DeoGUI, stop_event: threading.Event):
    """Push-to-talk loop. Runs on a background thread now -- the GUI's
    Tk mainloop owns the main thread instead."""
    while not stop_event.is_set():
        try:
            gui.push_status("idle")

            # Wait for the key OUTSIDE the mic lock. Holding it across the wait
            # meant push-to-talk owned the microphone permanently while idle,
            # and the wake word was rejected every single time it fired.
            wait_for_push_to_talk()
            if stop_event.is_set():
                break
            if _paused.is_set():
                continue

            with mic_lock:
                audio = record_while_held(on_start=lambda: gui.push_status("listening"))
            # Transcription doesn't need the microphone -- release it first so a
            # wake trigger isn't blocked behind a slow transcription.
            user_text = transcribe(audio)

            if not user_text:
                print("(didn't catch anything, try again)")
                continue

            print(f"You: {user_text}")
            gui.push_user_message(user_text)
            _ask_and_speak(brain, gui, user_text)

        except Exception as e:
            print(f"[main] error in voice loop: {e}")
            gui.push_system_message(f"Error: {e}")
            gui.push_status("idle")


def _provider_status_line() -> str:
    """Human-readable summary of which providers are actually active, since
    the whole point of the fallback chain is that Deo can silently run on
    a different stack turn to turn -- this makes that visible at a glance."""
    llm = f"Groq ({config.GROQ_LLM_MODEL})" if (config.USE_CLOUD_LLM and config.GROQ_API_KEY) else f"local Ollama -- {config.OLLAMA_MODEL}, {config.OLLAMA_MODEL_GAMING} while gaming"
    llm_fb = f", Ollama ({config.OLLAMA_MODEL} / {config.OLLAMA_MODEL_GAMING} while gaming) fallback ready" if (config.USE_CLOUD_LLM and config.GROQ_API_KEY) else ""
    stt = f"Groq ({config.GROQ_STT_MODEL})" if (config.USE_CLOUD_STT and config.GROQ_API_KEY) else "local Whisper only"
    stt_fb = f", faster-whisper ({config.WHISPER_MODEL_SIZE}) fallback ready" if (config.USE_CLOUD_STT and config.GROQ_API_KEY) else ""
    local_voice = (f"Kokoro ({config.KOKORO_VOICE})"
                   if config.USE_KOKORO_TTS and os.path.exists(config.KOKORO_MODEL_PATH) else "Piper")
    tts = f"Orpheus ({config.GROQ_TTS_VOICE})" if (config.USE_CLOUD_TTS and config.GROQ_API_KEY) else local_voice
    tts_fb = f", {local_voice} fallback ready" if (config.USE_CLOUD_TTS and config.GROQ_API_KEY) else (
        ", Piper fallback ready" if local_voice != "Piper" else "")
    return f"LLM: {llm}{llm_fb}\nSTT: {stt}{stt_fb}\nTTS: {tts}{tts_fb}"


def _already_running() -> bool:
    """One Deo at a time. With Deo starting at login, running it again by hand
    would put two copies on the same mic, camera and wake word -- double
    replies and a fight over the webcam. A named Windows mutex is released
    automatically when the process exits, even if it crashes."""
    import ctypes
    global _instance_mutex
    _instance_mutex = ctypes.windll.kernel32.CreateMutexW(None, False, r"Local\DeoVoiceAssistant")
    return ctypes.windll.kernel32.GetLastError() == 183   # ERROR_ALREADY_EXISTS


_instance_mutex = None


def main(start_hidden: bool = False):
    """start_hidden: open straight to the tray with no window -- used at login,
    and so restarting Deo never pops a window over a fullscreen game."""
    if _already_running():
        print(f"{config.ASSISTANT_NAME} is already running -- not starting a second copy.")
        return
    print(f"=== {config.ASSISTANT_NAME} is online ===")
    print(_provider_status_line())
    if not config.GROQ_API_KEY:
        print("[startup] No GROQ_API_KEY found in .env -- running fully local. "
              "Add one to console.groq.com's free tier to enable the cloud path.\n")
    else:
        print()

    mcp_bridge = MCPToolBridge()
    mcp_bridge.start()  # connects in the background, does not block startup

    # Synthesise the wake acknowledgement now, off the main thread, so the very
    # first "Deo" doesn't pay for it. Everything else here carries on meanwhile.
    if config.PREWARM_TTS:
        threading.Thread(target=tts.prewarm, daemon=True).start()

    # Same idea for Whisper, which local STT and the wake verifier both use.
    # On the GPU the first decode costs ~20s of kernel compilation.
    if config.WHISPER_PREWARM and (not config.USE_CLOUD_STT or config.WAKE_VERIFY):
        threading.Thread(target=stt.prewarm, daemon=True).start()

    brain = Brain(mcp_bridge=mcp_bridge)

    # Load the voice check now rather than on the first thing you say.
    from deo import speaker
    threading.Thread(target=speaker.prewarm, daemon=True).start()

    # 9B brain normally, 4B while a game is running (frees the GPU for it).
    from deo import gamemode
    gamemode.start()

    # GUI must be built on the main thread (Tkinter requirement). Background
    # threads only ever reach it through the thread-safe push_* methods.
    # The web window (deo/webui, shown through Edge WebView2) by default; the
    # Tk window if that can't load for any reason.
    GuiClass = DeoGUI
    if getattr(config, "GUI_STYLE", "web") == "web":
        try:
            from deo.gui_web import DeoWebGUI
            GuiClass = DeoWebGUI
        except Exception as e:
            print(f"[gui] web window unavailable ({e}), using the classic window")
    gui = GuiClass(
        assistant_name=config.ASSISTANT_NAME,
        on_text_submit=lambda text: _ask_and_speak(brain, gui, text),
        on_stop=lambda: (tts.stop(), gui.push_status("idle")),
        on_pause_toggle=lambda: _toggle_pause_everywhere(),
        start_hidden=start_hidden,
    )

    # The island: Siri-style glass pill at the top of the screen while the
    # window is closed. Long replies hand off to the window (gui.py).
    if config.ISLAND_ENABLED:
        try:
            from deo.island import Island
            # Pulling the island down far enough opens the full Deo window.
            gui.island = Island(on_open=lambda: gui.request_show()).start()
            if config.EDGE_GLOW_ENABLED:
                from deo.edgeglow import EdgeGlow
                gui.edge_glow = EdgeGlow().start()
            from deo import gamemode as _gm
            gui.is_gaming = lambda: bool(_gm.running_game())
        except Exception as e:
            print(f"[island] unavailable: {e}")

    # Tray icon: closing the window hides it there instead of quitting.
    tray = None
    try:
        from deo.tray import DeoTray
        tray = DeoTray(gui, is_paused=_paused.is_set, toggle_pause=toggle_pause,
                       on_quit=gui.request_quit)
        tray.start()
    except Exception as e:
        print(f"[tray] unavailable ({e}) -- closing the window will quit Deo")

    def _toggle_pause_everywhere():
        paused = toggle_pause()
        if tray:
            tray.set_paused(paused)
        return paused

    # The small line under the status: which brain, paused or not, next timer.
    def _info_loop():
        while True:
            try:
                from deo import gamemode as gm, timers as tm
                model = gm.current_model()
                game = gm.running_game()
                parts = [f"Brain {model.split(':')[0]}{' (game mode)' if game else ''}",
                         "mic paused" if _paused.is_set() else "mic live"]
                nxt = tm.next_summary()
                if nxt:
                    parts.append(nxt)
                gui.push_info("  ·  ".join(parts))
            except Exception as e:
                print(f"[main] info line: {e}")
            time.sleep(5)
    threading.Thread(target=_info_loop, daemon=True).start()
    gui.push_system_message(
        f"{config.ASSISTANT_NAME} is online. "
        + ("Groq (cloud) primary, local fallback ready." if config.GROQ_API_KEY else "Running fully local -- no GROQ_API_KEY set.")
    )

    # Timers/reminders announce themselves -- even mid-conversation, they
    # just wait for Deo to finish its current sentence.
    from deo import timers

    def announce(text):
        print(f"[timer] {text}")
        gui.push_assistant_message(text)
        with speech_lock:
            tts.speak(text)
    timers.on_fire = announce
    timers.start()

    screen_qa.register(brain, gui)  # arms the hotkeys in the background

    # tts.stop() rather than sd.stop(): streamed speech plays through its own
    # output stream and a live synthesiser process, neither of which the
    # module-level sd.stop() knows anything about.
    keyboard.add_hotkey(config.INTERRUPT_KEY, tts.stop)
    print(f"[interrupt] press [{config.INTERRUPT_KEY.upper()}] to stop {config.ASSISTANT_NAME} mid-speech")

    wake_listener = None
    try:
        wake_listener = WakeWordListener(on_wake=lambda: handle_wake(brain, gui))
        wake_listener.playback_text = lambda: (
            " ".join(_current_reply) if _phase in ("speaking", "thinking") else "")
        wake_listener.start_in_background()
    except Exception as e:
        print(f"[wakeword] failed to start, falling back to push-to-talk only: {e}")
        gui.push_system_message("Wake-word unavailable -- push-to-talk only.")

    # Imported lazily -- mediapipe/opencv are heavier, more recently-added
    # dependencies, so a missing install here shouldn't take down voice/GUI.
    gesture_listener = None
    if config.GESTURE_ENABLED:
        try:
            from deo.gesture import GestureListener
            gesture_listener = GestureListener(on_swipe=lambda direction: handle_gesture(direction, gui))
            gesture_listener.start_in_background()
        except Exception as e:
            print(f"[gesture] failed to start: {e}")
            gui.push_system_message("Gesture control unavailable.")

    stop_event = threading.Event()
    threading.Thread(target=voice_loop, args=(brain, gui, stop_event), daemon=True).start()

    if config.DEBUG_STARTUP_TIMING:
        print(f"[timing] entering main loop: {time.monotonic() - _t0:.2f}s")

    if start_hidden:
        if tray:
            print("[main] started in the tray")
        else:
            gui.request_show()    # no tray to reopen it from -- show the window after all

    try:
        gui.run()  # blocks on the Tk mainloop until the window is closed
    finally:
        stop_event.set()
        if wake_listener:
            wake_listener.stop()
        if gesture_listener:
            gesture_listener.stop()
        mcp_bridge.stop()
        if gui.island is not None:
            gui.island.post("quit")
        if getattr(gui, "edge_glow", None) is not None:
            gui.edge_glow.post("quit")
        if tray:
            try:
                tray.icon.stop()
            except Exception:
                pass
        tts.shutdown()  # kills the pre-spawned idle Piper, if there is one
        print("\nShutting down. Bye!")


if __name__ == "__main__":
    main()
