"""Pure-Python counting logic: line crossings, zones, dwell time.

Kept free of OpenCV/YOLO so it can be unit-tested on its own.
All coordinates here are normalised (0..1) relative to frame width/height.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

Point = tuple[float, float]


def _cross(a: Point, b: Point, c: Point) -> float:
    """z-component of (b-a) x (c-a). Sign tells which side of line a->b point c is on."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def point_in_polygon(pt: Point, poly: list[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


@dataclass
class Crossing:
    line_id: str
    track_id: int
    direction: str  # "in" | "out"
    ts: float


@dataclass
class LineCounter:
    """Counts tracks crossing a virtual line (e.g. a doorway).

    Direction convention: moving from the LEFT side of the line (as you walk from
    point A to point B) to the RIGHT side counts as "in". Set invert=True to flip.
    The dashboard draws an arrow showing the "in" direction.
    """

    line_id: str
    a: Point
    b: Point
    invert: bool = False
    cooldown_s: float = 1.5
    _last_pos: dict[int, Point] = field(default_factory=dict)
    _last_cross_ts: dict[int, float] = field(default_factory=dict)
    count_in: int = 0
    count_out: int = 0

    def update(self, track_id: int, pos: Point, ts: float) -> Crossing | None:
        prev = self._last_pos.get(track_id)
        self._last_pos[track_id] = pos
        if prev is None or not segments_intersect(prev, pos, self.a, self.b):
            return None
        if ts - self._last_cross_ts.get(track_id, -1e9) < self.cooldown_s:
            return None  # jitter on the line; ignore
        self._last_cross_ts[track_id] = ts
        went_right = _cross(self.a, self.b, pos) > 0  # image coords: y grows downward
        is_in = went_right != self.invert
        if is_in:
            self.count_in += 1
        else:
            self.count_out += 1
        return Crossing(self.line_id, track_id, "in" if is_in else "out", ts)

    def forget(self, alive: Iterable[int]) -> None:
        alive = set(alive)
        for tid in list(self._last_pos):
            if tid not in alive:
                self._last_pos.pop(tid, None)
                self._last_cross_ts.pop(tid, None)


@dataclass
class Visit:
    zone_id: str
    track_id: int
    enter_ts: float
    exit_ts: float

    @property
    def dwell_s(self) -> float:
        return self.exit_ts - self.enter_ts


@dataclass
class ZoneTracker:
    """Tracks how long anonymous tracks stay inside a polygon zone."""

    zone_id: str
    polygon: list[Point]
    exit_grace_s: float = 2.0     # tolerate brief detection drop-outs
    min_dwell_s: float = 1.0      # ignore people just walking through a corner
    _inside: dict[int, float] = field(default_factory=dict)     # track -> enter_ts
    _last_seen: dict[int, float] = field(default_factory=dict)  # track -> last ts inside

    def update(self, track_id: int, pos: Point, ts: float) -> None:
        if point_in_polygon(pos, self.polygon):
            self._inside.setdefault(track_id, ts)
            self._last_seen[track_id] = ts

    def flush(self, ts: float, force: bool = False) -> list[Visit]:
        """Close visits for tracks not seen inside for exit_grace_s."""
        done = []
        for tid, last in list(self._last_seen.items()):
            if force or ts - last > self.exit_grace_s:
                enter = self._inside.pop(tid)
                self._last_seen.pop(tid)
                if last - enter >= self.min_dwell_s:
                    done.append(Visit(self.zone_id, tid, enter, last))
        return done

    @property
    def occupancy(self) -> int:
        return len(self._inside)
