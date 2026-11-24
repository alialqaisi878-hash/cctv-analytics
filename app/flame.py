"""Small flames (a lighter, a match, a cigarette being lit) on a few chosen cameras.

The fire model looks at each camera every few seconds and needs a real fire. A lighter burns for a fraction of a
second and is a few pixels across, so the cameras the owner picks (up to 4) are watched differently:
every picture the camera sends, at full quality, through a quick check that needs no model:

  1. flame coloured: bright, and red / orange / yellow (at night, black-and-white cameras: very bright only);
  2. new: much brighter than that spot usually is (a lamp that is always on is part of the background);
  3. small: a light switched on changes a big part of the picture, a flame a small spot;
  4. alive: seen in at least 3 of the last 8 pictures at the same place, and flickering (its size or brightness
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
FLICKER = 0.12              # size or brightness must vary by this part between pictures


class FlameWatch:
    """One per chosen camera: takes every picture the camera sends, checks it in its own thread."""

    def __init__(self, worker, cfg: dict):
        self.worker, self.cfg = worker, cfg
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._bg: np.ndarray | None = None
        self._recent: collections.deque = collections.deque(maxlen=WINDOW)
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
        # the background slowly learns what is normal here; flame spots are kept out of it
        learn = np.where(flame, 0.0, 0.03 if self.checked > 30 else 0.2).astype(np.float32)
        self._bg += (v - self._bg) * learn
        changed = float((np.abs(new) > 40).mean())
        if changed > 0.05:
            # lights switched on or off, the camera changed to night mode: learn the new picture quickly
            self._bg = v.copy()
            self._recent.clear()
            return []
        n, _, stats, cents = cv2.connectedComponentsWithStats(flame.astype(np.uint8), connectivity=8)
        area_max = 0.004 * small.shape[0] * small.shape[1]
        spots = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 3 <= area <= area_max:
                x, y, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                         cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                bright = float(val[y:y + h, x:x + w].mean())
                spots.append((float(cents[i][0]), float(cents[i][1]), area, bright, (x, y, w, h)))
        self._recent.append(spots)
        for s in spots:
            if self._alive(s):
                x, y, w, h = s[4]
                box = (x / k, y / k, (x + w) / k, (y + h) / k)
                self.last_spot = f"{int(box[0])},{int(box[1])} {s[2]} px"
                self._alert(ts, frame, box)
                break
        return spots

    def _alive(self, s) -> bool:
        """Seen at about the same place in enough recent pictures, and flickering."""
        same = [t for spots in self._recent for t in spots
                if abs(t[0] - s[0]) <= NEAR and abs(t[1] - s[1]) <= NEAR]
        frames = sum(any(abs(t[0] - s[0]) <= NEAR and abs(t[1] - s[1]) <= NEAR for t in spots) for spots in self._recent)
        if frames < NEED or len(self._recent) < NEED:
            return False
        areas = np.array([t[2] for t in same], np.float32)
        brights = np.array([t[3] for t in same], np.float32)
        area_var = float(areas.std() / max(areas.mean(), 1.0))
        bright_var = float(brights.std() / max(brights.mean(), 1.0))
        appeared = frames < len(self._recent)              # it was not there a moment ago
        # a spot there in every picture must flicker clearly; one that just appeared may flicker a little
        return (area_var >= FLICKER or bright_var >= FLICKER / 4) and (appeared or area_var >= FLICKER)

    def _alert(self, ts: float, frame: np.ndarray, box) -> None:
        pause = float(self.h.get("flame_cooldown_minutes", 2)) * 60
        if ts - self._alerted_at < pause:
            return
        self._alerted_at = ts
        self._recent.clear()
        self.worker.flame_alert(ts, frame, box)
