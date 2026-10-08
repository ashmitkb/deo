import threading

# Held while anything is speaking, so the voice loop and screen-QA
# hotkey can't both call tts.speak() at the same time.
speech_lock = threading.Lock()
# Held while anything is actively listening/recording, so push-to-talk
# and the wake-word listener can't both try to record at the same time.
mic_lock = threading.Lock()
# Held around a full brain.ask() call. Brain.history is one shared list
# mutated non-atomically (append user msg, call Ollama, append reply) --
# with the GUI now able to trigger a brain.ask() from typed input at any
# time, alongside voice, wake-word, and screen-QA, this stops two turns
# from interleaving and corrupting the conversation history.
brain_lock = threading.Lock()
