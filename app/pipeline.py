"""One worker thread per camera: read frames -> detect & track people -> count -> annotate."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from .analytics import LineCounter, ZoneTracker
from .db import Database

log = logging.getLogger("cctv.pipeline")
PERSON = 0  # COCO class id


def _hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, cfg: dict, db: Database):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam, self.cfg, self.db = cam, cfg, db
        self.id = cam["id"]
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status = "starting"
        self.fps = 0.0
        self.people_now = 0
        self.lines = [LineCounter(l["id"], tuple(l["points"][0]), tuple(l["points"][1]), l.get("invert", False))
                      for l in cam["lines"]]
        self.zones = [ZoneTracker(z["id"], [tuple(p) for p in z["points"]]) for z in cam["zones"]]
        self._model = None
        self._last_occ_write = 0.0
        self._last_alert: dict[str, float] = {}

    # ---------- public ----------
    def snapshot(self) -> dict:
        return {
            "id": self.id, "name": self.cam["name"], "status": self.status, "fps": round(self.fps, 1),
            "people_now": self.people_now,
            "lines": [{"id": l.line_id, "name": c["name"], "in": l.count_in, "out": l.count_out}
                      for l, c in zip(self.lines, self.cam["lines"])],
            "zones": [{"id": z.zone_id, "name": c["name"], "occupancy": z.occupancy}
                      for z, c in zip(self.zones, self.cam["zones"])],
        }

    def latest_jpeg(self) -> bytes | None:
        with self.lock:
            return self.jpeg

    def stop(self):
        self.stop_event.set()

    # ---------- internals ----------
    def _load_model(self):
        from ultralytics import YOLO  # imported lazily: heavy
        self._model = YOLO(self.cfg["detector"]["model"])

    def _open(self):
        src = self.cam["source"]
        src = int(src) if isinstance(src, int) or (isinstance(src, str) and src.isdigit()) else src
        cap = cv2.VideoCapture(src)
        if isinstance(src, str) and src.startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        try:
            self._load_model()
        except Exception as e:  # noqa: BLE001
            self.status = f"model error: {e}"
            log.exception("model load failed")
            return
        backoff = 2
        while not self.stop_event.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.status = "offline"
                log.warning("camera %s offline, retrying in %ss", self.id, backoff)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 2
            self.status = "online"
            self._loop(cap)
            cap.release()
            if self.cam.get("loop") and not self.stop_event.is_set():
                # restart sample video: reset tracker so IDs don't carry across
                self._load_model()
                continue
            if not self.stop_event.is_set():
                if self._is_file():
                    self.status = "ended"
                    for z in self.zones:
                        for v in z.flush(time.time(), force=True):
                            self.db.add_visit(self.id, v.zone_id, v.track_id, v.enter_ts, v.exit_ts)
                    return
                self.status = "reconnecting"
                self.stop_event.wait(2)

    def _is_file(self) -> bool:
        src = str(self.cam["source"])
        return not src.isdigit() and "://" not in src

    def _loop(self, cap):
        det = self.cfg["detector"]
        target_dt = 1.0 / max(det["process_fps"], 1)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        is_file = self._is_file()
        skip = max(int(round(src_fps / det["process_fps"])), 1) if is_file else 1
        frame_i, t_prev = 0, time.time()
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                return
            frame_i += 1
            if is_file and frame_i % skip:
                continue
            t0 = time.time()
            self._process(frame, t0)
            # pace: real-time for files, rate-limit for live streams
            elapsed = time.time() - t0
            if is_file:
                time.sleep(max(skip / src_fps - elapsed, 0))
            elif elapsed < target_dt:
                # drain buffered frames so we always analyse the freshest one
                cap.grab()
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

    def _process(self, frame: np.ndarray, ts: float):
        det = self.cfg["detector"]
        h, w = frame.shape[:2]
        res = self._model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON],
                                conf=det["confidence"], imgsz=det["imgsz"], device=det.get("device"),
                                verbose=False)[0]
        tracks = []
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist()):
                x1, y1, x2, y2 = box
                foot = ((x1 + x2) / 2 / w, y2 / h)  # bottom-centre = where the person stands
                tracks.append((tid, (x1, y1, x2, y2), foot))
        self.people_now = len(tracks)

        for tid, _, foot in tracks:
            for lc in self.lines:
                c = lc.update(tid, foot, ts)
                if c:
                    self.db.add_crossing(ts, self.id, c.line_id, c.direction, tid)
            for z in self.zones:
                z.update(tid, foot, ts)
        alive = [t[0] for t in tracks]
        for lc in self.lines:
            lc.forget(alive)
        for z in self.zones:
            for v in z.flush(ts):
                self.db.add_visit(self.id, v.zone_id, v.track_id, v.enter_ts, v.exit_ts)

        if ts - self._last_occ_write >= 10:
            self._last_occ_write = ts
            self.db.add_occupancy(ts, self.id, self.people_now)
        self._check_alerts(ts)
        self._annotate(frame, tracks)

    def _check_alerts(self, ts: float):
        al = self.cfg["alerts"]
        cooldown = al["cooldown_minutes"] * 60

        def fire(kind, msg):
            if ts - self._last_alert.get(kind, 0) >= cooldown:
                self._last_alert[kind] = ts
                self.db.add_alert(ts, self.id, kind, msg)
                log.info("ALERT %s %s: %s", self.id, kind, msg)

        if al["after_hours_presence"] and self.people_now > 0 and not self._in_working_hours(ts):
            fire("after_hours", f"تم رصد {self.people_now} شخص خارج أوقات الدوام في {self.cam['name']}")
        if al["max_occupancy"] and self.people_now > al["max_occupancy"]:
            fire("crowding", f"ازدحام: {self.people_now} أشخاص في {self.cam['name']} (الحد {al['max_occupancy']})")

    def _in_working_hours(self, ts: float) -> bool:
        wh = self.cfg["working_hours"]
        d: datetime = self.db.local(ts)
        if d.weekday() not in wh["days"]:
            return False
        sh, sm = _hhmm(wh["start"])
        eh, em = _hhmm(wh["end"])
        return (sh, sm) <= (d.hour, d.minute) < (eh, em)

    def _annotate(self, frame: np.ndarray, tracks):
        h, w = frame.shape[:2]
        P = lambda p: (int(p[0] * w), int(p[1] * h))
        blur = self.cfg["privacy"]["blur_people"]
        for tid, (x1, y1, x2, y2), _ in tracks:
            x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)
            if blur and x2 > x1 and y2 > y1:
                hy2 = y1 + max(int((y2 - y1) * 0.25), 1)  # head region
                roi = frame[y1:hy2, x1:x2]
                if roi.size:
                    k = max((x2 - x1) // 3 | 1, 15)
                    frame[y1:hy2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120), 2)
            cv2.putText(frame, f"#{tid}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 2)
        for zc, z in zip(self.cam["zones"], self.zones):
            pts = np.array([P(p) for p in zc["points"]], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (200, 140, 40))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], True, (200, 140, 40), 2)
            cv2.putText(frame, f"{z.zone_id}: {z.occupancy}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 200, 80), 2)
        for lc in self.lines:
            a, b = P(lc.a), P(lc.b)
            cv2.line(frame, a, b, (0, 200, 255), 3)
            # arrow showing the "in" direction (perpendicular to the line)
            mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = max((dx * dx + dy * dy) ** 0.5, 1)
            sgn = -1 if lc.invert else 1
            nx, ny = int(-dy / n * 40 * sgn), int(dx / n * 40 * sgn)
            cv2.arrowedLine(frame, (mx, my), (mx + nx, my + ny), (0, 200, 255), 2, tipLength=0.4)
            cv2.putText(frame, f"{lc.line_id} IN {lc.count_in} / OUT {lc.count_out}", (a[0], max(a[1] - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"people: {len(tracks)}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
