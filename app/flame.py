"""Small flames (a lighter, a match, a cigarette being lit) on a few chosen cameras.

The fire model looks at each camera every few seconds and needs a real fire. A lighter burns for a fraction of a
second and is a few pixels across, so the cameras the owner picks (up to 4) are watched differently:
every picture the camera sends, at full quality, through a quick check that needs no model:

  1. flame coloured: bright, and red / orange / yellow (at night, black-and-white cameras: very bright only);
  2. new: much brighter than that spot usually is (a lamp that is always on is part of the background);
  3. small: a light switched on changes a big part of the picture, a flame a small spot;
  4. alive: seen in at least 3 of the last 8 pictures at the same place, and flickering (its size
     changes from picture to picture, a phone screen or a reflection is steady).

When all four hold, the camera alerts with the whole picture marked and a close-up of the flame.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("cctv.flame")

WIDTH = 640                 # pictures are checked at this width
MAX_CAMERAS = 4
WINDOW, NEED = 8, 3         # seen in 3 of the last 8 pictures ...
NEAR = 24                   # ... within this many pixels of each other
FLICKER = 0.12              # its size must vary by this part between pictures
DELAY = 10                  # a spot's own last pictures do not count as "this place is always busy" ...
BUSY = 3.0                  # ... but flame-like in this many older pictures (fading over ~10 s) means it is:
                            # the camera's clock, whose white digits change every second, or a shimmering edge
CELL = 16                   # busy places are remembered in squares of this many pixels
WARMUP = 40                 # pictures to learn the busy places after the program starts, before the first alert
SETTLE_S = 1.0              # seconds to settle after the background was learned anew (lights, exposure)


class FlameWatch:
    """One per chosen camera: takes every picture the camera sends, checks it in its own thread."""

    def __init__(self, worker, cfg: dict):
        self.worker, self.cfg = worker, cfg
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._bg: np.ndarray | None = None
        self._recent: collections.deque = collections.deque(maxlen=WINDOW)
        self._delay: collections.deque = collections.deque(maxlen=DELAY)   # the last flame masks, not yet in _busy
        self._busy: np.ndarray | None = None                                # how often each place looked flame-like
        self._renewed_at = float("-inf")    # when the background was last learned anew
        self._alerted_at = float("-inf")
        self.checked = 0
        self.dropped = 0
        self.rate = 0.0
        self.last_spot = ""
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name=f"flame-{worker.id}").start()

    @property
    def h(self) -> dict:
        return self.cfg["hazard"]

    def stop(self) -> None:
        self._stop.set()

    def feed(self, frame: np.ndarray, ts: float) -> None:
        """From the camera's reading thread: never waits. If checking falls behind, the oldest picture goes."""
        try:
            self._q.put_nowait((frame, ts))
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((frame, ts))
            except queue.Full:
                pass

    def _run(self) -> None:
        t_prev = time.time()
        while not self._stop.is_set():
            try:
                frame, ts = self._q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self.check(frame, ts)
            except Exception:  # noqa: BLE001
                log.exception("small flame check failed on camera %s", self.worker.id)
            now = time.time()
            self.rate = 0.9 * self.rate + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now

    # ---------- the check ----------
    def check(self, frame: np.ndarray, ts: float) -> list[tuple]:
        """One picture. Returns the flame-like spots found in it (for tests); alerts when one is confirmed."""
        self.checked += 1
        fh, fw = frame.shape[:2]
        k = WIDTH / fw
        small = cv2.resize(frame, (WIDTH, max(1, int(fh * k))), interpolation=cv2.INTER_AREA) if fw > WIDTH else frame
        k = small.shape[1] / fw
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        v = val.astype(np.float32)
        if self._bg is None or self._bg.shape != v.shape:
            self._bg = v.copy()
            self._busy = np.zeros(((v.shape[0] + CELL - 1) // CELL, (v.shape[1] + CELL - 1) // CELL), np.float32)
            self._delay.clear()
            return []
        new = v - self._bg
        grey = float(sat.mean()) < 12                       # night: the camera sends black and white
        if grey:
            flame = (val >= 250) & (new >= 70)
        else:
            # the flame's edge: strongly coloured red / orange / yellow, bright, and new at that spot
            colour = ((hue <= 35) | (hue >= 170)) & (sat >= 90) & (val >= 170) & (new >= 30)
            # the middle of a flame is white-hot: take bright new spots that are coloured or touch that colour
            hot = (val >= 215) & (new >= 45)
            near_colour = cv2.dilate(colour.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            flame = hot & near_colour
        # places that keep looking flame-like (the clock's digits, a glinting edge) are not a flame
        if len(self._delay) == DELAY:
            self._busy *= 0.995
            self._busy += self._delay[0]
        bh, bw = self._busy.shape
        cells = np.zeros((bh * CELL, bw * CELL), np.uint8)
        cells[:flame.shape[0], :flame.shape[1]] = flame
        self._delay.append(cells.reshape(bh, CELL, bw, CELL).max(axis=(1, 3)).astype(np.float32))
        busy = cv2.dilate((self._busy >= BUSY).astype(np.uint8), np.ones((3, 3), np.uint8))
        busy = np.kron(busy, np.ones((CELL, CELL), np.uint8))[:flame.shape[0], :flame.shape[1]] > 0
        flame = flame & ~busy
        # the background slowly learns what is normal here; flame spots are kept out of it
        learn = np.where(flame, 0.0, 0.03 if self.checked > 30 else 0.2).astype(np.float32)
        self._bg += (v - self._bg) * learn
        changed = float((np.abs(new) > 40).mean())
        if changed > 0.05:
            # lights switched on or off, the camera changed to night mode: learn the new picture quickly
            self._bg = v.copy()
            self._recent.clear()
            self._renewed_at = ts
            return []
        n, _, stats, cents = cv2.connectedComponentsWithStats(flame.astype(np.uint8), connectivity=8)
        area_max = 0.004 * small.shape[0] * small.shape[1]
        spots = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 3 <= area <= area_max:
                x, y, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                         cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                if min(w, h) < 2:
                    continue                    # a one-pixel line: the edge of a letter or a wire, never a flame
                bright = float(val[y:y + h, x:x + w].mean())
                spots.append((float(cents[i][0]), float(cents[i][1]), area, bright, (x, y, w, h)))
        self._recent.append(spots)
        settling = ts - self._renewed_at <= SETTLE_S
        for s in spots:
            if self.checked <= WARMUP or (settling and self._seen_before(s)) or not self._alive(s):
                continue
            x, y, w, h = s[4]
            box = (x / k, y / k, (x + w) / k, (y + h) / k)
            self.last_spot = f"{int(box[0])},{int(box[1])} {s[2]} px"
            self._alert(ts, frame, box)
            break
        return spots

    def _seen_before(self, s) -> bool:
        """This place looked flame-like before (the clock's digits do, again and again; a new fire does not)."""
        cy, cx = int(s[1]) // CELL, int(s[0]) // CELL
        return float(self._busy[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].max()) >= 1.0

    def _alive(self, s) -> bool:
        """Seen at about the same place in enough recent pictures, and flickering."""
        same = [t for spots in self._recent for t in spots
                if abs(t[0] - s[0]) <= NEAR and abs(t[1] - s[1]) <= NEAR]
        frames = sum(any(abs(t[0] - s[0]) <= NEAR and abs(t[1] - s[1]) <= NEAR for t in spots) for spots in self._recent)
        if frames < NEED or len(self._recent) < NEED:
            return False
        areas = np.array([t[2] for t in same], np.float32)
        area_var = float(areas.std() / max(areas.mean(), 1.0))
        # a flame changes its size from picture to picture. Brightness alone is not enough: the camera's clock and
        # letters shimmer in brightness with the compression but keep their size
        return area_var >= FLICKER

    def _alert(self, ts: float, frame: np.ndarray, box) -> None:
        pause = float(self.h.get("flame_cooldown_minutes", 2)) * 60
        if ts - self._alerted_at < pause:
            return
        self._alerted_at = ts
        self._recent.clear()
        self.worker.flame_alert(ts, frame, box)
