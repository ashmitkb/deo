"""
Always-on hand-gesture window control for Deo.

Runs a background thread that continuously reads frames from a webcam,
feeds them through MediaPipe's HandLandmarker task, and watches one hand
landmark's horizontal position for a fast, sustained swipe left or right.
A detected swipe calls on_swipe(direction) -- main.py wires that straight
to tools.move_active_window(), bypassing the LLM entirely so a gesture
feels instant rather than waiting on an Ollama round-trip.

NOTE: this uses MediaPipe's current Tasks API (HandLandmarker), not the
older `mp.solutions.hands` API, which Google deprecated in 2023.

Setup:
    pip install mediapipe opencv-python

Model:
    First run downloads hand_landmarker.task (a few MB, one-time, needs
    internet for that download only) to config.GESTURE_MODEL_PATH.
    Detection itself then runs fully offline, same as everything else here.
"""

import os
import time
import threading
import urllib.request

import cv2
import mediapipe as mp

from . import config

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# Landmark 9 = middle-finger MCP joint, roughly the palm's geometric center.
# More stable than the wrist (0), which drifts a lot as the hand rotates.
_PALM_LANDMARK = 9


def trim_history(history, now: float, window_seconds: float) -> list:
    """Drop palm positions older than the swipe window."""
    cutoff = now - window_seconds
    return [(t, x) for t, x in history if t >= cutoff]


def detect_swipe(history, min_distance: float):
    """Decide whether a trajectory is a swipe. Returns (direction, dx) or None.

    Split out of the camera loop deliberately: this is the part with the actual
    logic in it, and keeping it a pure function means it can be tested against
    synthetic trajectories instead of requiring a webcam and a human hand.

    history: [(timestamp, normalized_palm_x), ...], oldest first.
    """
    if len(history) < 2:
        return None
    dx = history[-1][1] - history[0][1]
    if abs(dx) < min_distance:
        return None
    # x is already mirrored by the caller (cv2.flip), so +dx is the user's right.
    return ("right" if dx > 0 else "left"), dx


# (tip, middle knuckle) landmark ids for each finger, thumb excluded: people
# hold "1" and "2" both with the thumb tucked and with it out, so it's noise.
_FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
_WRIST = 0
# Tip/knuckle distance-from-wrist ratios. A straight finger's tip is well past
# its middle knuckle (~1.3-1.6); a curled one's is at or inside it (<1.0). The
# band between is deliberately "don't know", so a half-bent finger can't flip
# a 1 into a 2.
_EXTENDED_RATIO = 1.15
_FOLDED_RATIO = 1.05

# Which pose means which screen. 1 finger -> right, 2 fingers -> left.
POSES = {("index",): "right", ("index", "middle"): "left"}


def finger_states(points):
    """{finger: True (up) / False (down) / None (unclear)} from 21 (x, y)
    landmark points in pixels. Distance-from-wrist, so it works whether the
    hand faces the camera square-on or is tilted."""
    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    wrist = points[_WRIST]
    states = {}
    for name, (tip, knuckle) in _FINGERS.items():
        ratio = dist(points[tip], wrist) / max(dist(points[knuckle], wrist), 1e-6)
        states[name] = True if ratio > _EXTENDED_RATIO else False if ratio < _FOLDED_RATIO else None
    return states


# Max tilt of the hand from straight up, in degrees. A deliberate "1" or "2"
# is held with the fingers pointing at the ceiling; scratching your face points
# the finger sideways or back at you -- the false trigger this rule exists for.
_MAX_TILT_DEGREES = 35


def hand_is_upright(points) -> bool:
    """Wrist -> middle-finger knuckle points up (image y grows downward)."""
    import math
    wx, wy = points[_WRIST]
    mx, my = points[9]
    dx, dy = mx - wx, wy - my            # dy > 0 means the knuckle is above the wrist
    if dy <= 0:
        return False
    return math.degrees(math.atan2(abs(dx), dy)) <= _MAX_TILT_DEGREES


def classify_pose(points):
    """'right', 'left', or None for one hand's landmarks. None whenever any
    finger is ambiguous or the hand isn't held upright: a deliberate, held
    pose is the whole point."""
    if not hand_is_upright(points):
        return None
    states = finger_states(points)
    if any(v is None for v in states.values()):
        return None
    up = tuple(name for name in _FINGERS if states[name])
    return POSES.get(up)


class PoseHold:
    """Fires once when the same pose has been held for `hold_seconds`, then
    stays quiet until the pose changes or the hand goes away -- so keeping two
    fingers up moves the window once, not every half second."""

    def __init__(self, hold_seconds: float, max_drift: float = 0.06):
        self.hold_seconds = hold_seconds
        # How far (fraction of frame width) the hand may wander during the hold.
        # A held gesture stays put; a hand scratching your face doesn't.
        self.max_drift = max_drift
        self._pose = None
        self._since = 0.0
        self._fired = False
        self._anchor = None

    def feed(self, pose, now: float, position=None):
        moved = (position is not None and self._anchor is not None and
                 ((position[0] - self._anchor[0]) ** 2 + (position[1] - self._anchor[1]) ** 2) ** 0.5
                 > self.max_drift)
        if pose != self._pose or moved:
            self._pose, self._since, self._fired = pose, now, False
            self._anchor = position
            return None
        if pose is None or self._fired or now - self._since < self.hold_seconds:
            return None
        self._fired = True
        return pose


def _ensure_model(path: str):
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[gesture] downloading hand landmark model to {path}...")
    urllib.request.urlretrieve(_MODEL_URL, path)


class GestureListener:
    def __init__(self, on_swipe):
        """on_swipe: callable(direction: str), direction is 'left' or
        'right'. Invoked from this listener's own background thread."""
        self._on_swipe = on_swipe
        self._stop = threading.Event()
        self._thread = None

        _ensure_model(config.GESTURE_MODEL_PATH)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.GESTURE_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.GESTURE_MIN_CONFIDENCE,
            min_hand_presence_confidence=config.GESTURE_MIN_CONFIDENCE,
            min_tracking_confidence=config.GESTURE_MIN_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        self._mode = getattr(config, "GESTURE_MODE", "swipe")
        self._hold = PoseHold(getattr(config, "GESTURE_HOLD_SECONDS", 0.5))

    def _open_camera(self):
        """On Windows, OpenCV's default MSMF backend is slow to open a webcam and
        sometimes fails outright where DirectShow works, so try DSHOW first."""
        attempts = []
        if hasattr(cv2, "CAP_DSHOW"):
            attempts.append(("DSHOW", cv2.CAP_DSHOW))
        attempts.append(("default", None))

        for label, backend in attempts:
            cap = cv2.VideoCapture(config.GESTURE_CAMERA_INDEX) if backend is None \
                else cv2.VideoCapture(config.GESTURE_CAMERA_INDEX, backend)
            if cap.isOpened():
                ok, _ = cap.read()      # opened isn't the same as delivering frames
                if ok:
                    print(f"[gesture] camera {config.GESTURE_CAMERA_INDEX} open via {label}")
                    return cap
                print(f"[gesture] camera opened via {label} but returned no frames")
            cap.release()
        return None

    def _wait_for_camera(self):
        """Open the camera, retrying every few seconds instead of giving up --
        after the laptop sleeps (or another app had it) it comes back on its
        own. Returns None only when the listener is being stopped."""
        cap = self._open_camera()
        if cap is None:
            print(f"[gesture] camera {config.GESTURE_CAMERA_INDEX} unavailable "
                  f"(asleep, or another app has it) -- retrying every 5s")
        while cap is None and not self._stop.wait(5):
            cap = self._open_camera()
        return cap

    def _run(self):
        cap = self._wait_for_camera()
        if cap is None:
            return
        if self._mode == "fingers":
            print("[gesture] watching for fingers: 1 = right screen, 2 = left screen")
        else:
            print("[gesture] watching for swipe left/right...")

        history = []  # [(timestamp, normalized_palm_x), ...]
        cooldown_until = 0.0
        last_seen = 0.0         # when a hand was last detected
        last_timestamp_ms = -1  # detect_for_video requires strictly increasing stamps
        read_failures = 0

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    # Don't spin at full speed on a dead camera.
                    read_failures += 1
                    if read_failures == 1 or read_failures % 100 == 0:
                        print(f"[gesture] camera read failed ({read_failures}x)")
                    if read_failures > 300:
                        # Typically the laptop slept. Used to give up for good.
                        print("[gesture] camera stopped delivering frames -- reopening it")
                        cap.release()
                        cap = self._wait_for_camera()
                        if cap is None:
                            return
                        read_failures = 0
                        continue
                    time.sleep(0.05)
                    continue
                read_failures = 0

                # Mirror the frame so motion matches the user's own sense of
                # left/right, the way a mirror (or a selfie camera) would --
                # without this, a swipe to their right reads as "left".
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # Strictly increasing, or MediaPipe rejects the frame outright.
                timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
                now = time.monotonic()

                if now < cooldown_until:
                    history.clear()
                    continue

                if not result.hand_landmarks:
                    # THE FIX for "swipes never register": a swipe fast enough to
                    # cover GESTURE_SWIPE_MIN_DISTANCE motion-blurs, MediaPipe
                    # drops a frame or two mid-motion, and the old code cleared
                    # the whole trajectory on the very first missed frame. That
                    # left only swipes slow enough to stay tracked -- which are
                    # too slow to cover the required distance. So tolerate a
                    # short gap and keep the trajectory.
                    if now - last_seen > config.GESTURE_LOST_GRACE_SECONDS:
                        history.clear()
                        self._hold.feed(None, now)   # hand gone: re-arm the pose
                    continue

                last_seen = now

                if self._mode == "fingers":
                    h, w = frame.shape[:2]
                    points = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
                    palm = result.hand_landmarks[0][_PALM_LANDMARK]
                    direction = self._hold.feed(classify_pose(points), now, (palm.x, palm.y))
                    if direction:
                        print(f"[gesture] {'1 finger' if direction == 'right' else '2 fingers'}"
                              f" -> {direction} screen")
                        cooldown_until = now + config.GESTURE_COOLDOWN
                        try:
                            self._on_swipe(direction)
                        except Exception as e:
                            print(f"[gesture] on_swipe callback raised: {e}")
                    continue

                palm_x = result.hand_landmarks[0][_PALM_LANDMARK].x  # normalized 0..1
                history.append((now, palm_x))
                history = trim_history(history, now, config.GESTURE_SWIPE_WINDOW_SECONDS)

                swipe = detect_swipe(history, config.GESTURE_SWIPE_MIN_DISTANCE)
                if swipe:
                    direction, dx = swipe
                    print(f"[gesture] swipe {direction} detected (dx={dx:.2f})")
                    history = []
                    cooldown_until = now + config.GESTURE_COOLDOWN
                    try:
                        self._on_swipe(direction)
                    except Exception as e:
                        print(f"[gesture] on_swipe callback raised: {e}")
        finally:
            cap.release()
            self._landmarker.close()

    def start_in_background(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
