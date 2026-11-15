"""One worker thread per camera: read frames -> detect & track people -> count -> annotate."""
from __future__ import annotations

import logging
import os
import threading
import time

# IP cameras: TCP transport avoids the broken/grey frames that UDP gives on busy networks
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .analytics import LineCounter, ZoneTracker
from .config import in_working_hours
from .db import Database

log = logging.getLogger("cctv.pipeline")
PERSON = 0  # COCO class id


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, cfg: dict, db: Database, faces=None, notifier=None):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam, self.cfg, self.db = cam, cfg, db
        self.faces, self.notifier = faces, notifier     # FaceEngine / Notifier, both optional
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
        self._pending: list[tuple[str, str, str | None]] = []   # alerts waiting for this frame's picture
        self.raw: np.ndarray | None = None      # latest unmarked picture (for enrolling faces from a camera)
        self._last_raw = 0.0
        self._last_face = 0.0
        self.names: dict[int, dict] = {}        # track id -> recognised person (or unknown) for the live view
        self._unknown_seen: dict[int, int] = {}
        self._offline_since: float | None = None
        self._was_online = False

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

    def latest_raw(self) -> np.ndarray | None:
        with self.lock:
            return None if self.raw is None else self.raw.copy()

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
                self._note_offline()
                log.warning("camera %s offline, retrying in %ss", self.id, backoff)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 60)
                continue
            backoff = 2
            self.status = "online"
            self._loop(cap)
            self._note_offline()
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

    def _note_offline(self) -> None:
        """Alert once when a camera that was working stops sending pictures for 2 minutes."""
        now = time.time()
        if self._offline_since is None:
            self._offline_since = now
        elif (self._was_online and now - self._offline_since >= 120 and self.cfg["alerts"].get("camera_offline", True)
              and not self._is_file()):
            self._was_online = False
            self.db.add_alert(now, self.id, "offline", f"الكاميرا {self.cam['name']} مفصولة أو ما تبث صورة")
            if self.notifier:
                self.notifier.queue_alert("offline", f"⚠️ الكاميرا {self.cam['name']} مفصولة أو ما تبث صورة", None)

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
            self._offline_since = None
            self._was_online = True
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
        if ts - self._last_raw >= 1.0:
            self._last_raw = ts
            with self.lock:
                self.raw = frame.copy()
        self._recognise(frame, tracks, ts)

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
        self._flush_alerts(ts)

    # ---------- face recognition (only people enrolled by the owner) ----------
    def _recognise(self, frame: np.ndarray, tracks, ts: float) -> None:
        alive = {t[0] for t in tracks}
        for tid in list(self.names):
            if tid not in alive:
                self.names.pop(tid, None)
                self._unknown_seen.pop(tid, None)
        fc = self.cfg["faces"]
        if not (fc["enabled"] and self.faces and self.faces.ready and tracks):
            return
        if ts - self._last_face < fc.get("interval_s", 0.5):
            return
        # people already named are not checked again while they stay in view
        if all(self.names.get(t[0], {}).get("person_id") for t in tracks):
            return
        self._last_face = ts
        self.faces.threshold = fc["threshold"]
        for f in self.faces.analyse(frame):
            fx, fy = (f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2
            tid = next((t[0] for t in tracks if t[1][0] <= fx <= t[1][2] and t[1][1] <= fy <= t[1][1] + (t[1][3] - t[1][1]) * 0.5),
                       None)
            if tid is None:
                continue
            p = f["person"]
            if p is None:
                if not self.names.get(tid, {}).get("person_id"):
                    self._unknown_seen[tid] = self._unknown_seen.get(tid, 0) + 1
                    self.names[tid] = {"person_id": None, "name": "غير معروف"}
                    if self._unknown_seen[tid] == 3:    # seen clearly a few times, still nobody we know
                        self._unknown_alert(ts)
                continue
            if self.names.get(tid, {}).get("person_id") == p["id"]:
                continue
            self.names[tid] = {"person_id": p["id"], "name": p["name"]}
            self.db.add_sighting(ts, self.id, p["id"], p["name"], f["score"])
            log.info("recognised %s on %s (%.2f)", p["name"], self.id, f["score"])
            if p.get("alert_on_seen"):
                self._queue_alert(ts, f"person:{p['id']}", f"تم رصد {p['name']} في {self.cam['name']}", p["id"])

    def _unknown_alert(self, ts: float) -> None:
        if self.cfg["alerts"].get("unknown_after_hours") and not self._in_working_hours(ts):
            self._queue_alert(ts, "unknown", f"شخص غير معروف خارج أوقات الدوام في {self.cam['name']}")

    # ---------- alerts ----------
    def _queue_alert(self, ts: float, kind: str, msg: str, person_id: str | None = None) -> None:
        cooldown = self.cfg["alerts"]["cooldown_minutes"] * 60
        if ts - self._last_alert.get(kind, 0) >= cooldown:
            self._last_alert[kind] = ts
            self._pending.append((kind, msg, person_id))

    def _flush_alerts(self, ts: float) -> None:
        """Save the alerts of this frame together with its (marked) picture."""
        if not self._pending:
            return
        jpg = self.latest_jpeg() if self.cfg["privacy"].get("save_alert_pictures", True) else None
        for kind, msg, pid in self._pending:
            self.db.add_alert(ts, self.id, kind.split(":", 1)[0], msg, jpg, pid)
            log.info("ALERT %s %s: %s", self.id, kind, msg)
            if self.notifier:
                t = self.db.local(ts).strftime("%H:%M")
                self.notifier.queue_alert(kind, f"🔔 {msg}\n{t}", self.latest_jpeg())
        self._pending.clear()

    def _check_alerts(self, ts: float):
        al = self.cfg["alerts"]
        if al["after_hours_presence"] and self.people_now > 0 and not self._in_working_hours(ts):
            self._queue_alert(ts, "after_hours", f"تم رصد {self.people_now} شخص خارج أوقات الدوام في {self.cam['name']}")
        if al["max_occupancy"] and self.people_now > al["max_occupancy"]:
            self._queue_alert(ts, "crowding", f"ازدحام: {self.people_now} أشخاص في {self.cam['name']} (الحد {al['max_occupancy']})")

    def _in_working_hours(self, ts: float) -> bool:
        return in_working_hours(self.cfg, self.db.local(ts))

    def _annotate(self, frame: np.ndarray, tracks):
        h, w = frame.shape[:2]
        P = lambda p: (int(p[0] * w), int(p[1] * h))
        blur = self.cfg["privacy"]["blur_people"]
        for tid, (x1, y1, x2, y2), _ in tracks:
            x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)
            who = self.names.get(tid)
            if blur and x2 > x1 and y2 > y1 and not (who and who.get("person_id")):
                hy2 = y1 + max(int((y2 - y1) * 0.25), 1)  # head region
                roi = frame[y1:hy2, x1:x2]
                if roi.size:
                    k = max((x2 - x1) // 3 | 1, 15)
                    frame[y1:hy2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            color = (80, 200, 120) if not who else ((255, 170, 60) if who.get("person_id") else (60, 60, 230))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if who:
                _label(frame, who["name"], (x1, max(y1 - 4, 18)), color)
            else:
                cv2.putText(frame, f"#{tid}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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


_FONT = None


def _font(size: int = 18):
    """A font with Arabic letters for names on the picture (Windows fonts; None if unavailable)."""
    global _FONT
    if _FONT is None:
        _FONT = False
        try:
            from PIL import ImageFont
            for f in ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                try:
                    _FONT = ImageFont.truetype(f, size)
                    break
                except OSError:
                    continue
        except ImportError:
            pass
    return _FONT or None


def _shape(text: str) -> str:
    """Join Arabic letters and order them right-to-left so PIL draws them readable."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def _label(frame: np.ndarray, text: str, org: tuple[int, int], color) -> None:
    font = _font()
    ascii_only = all(ord(ch) < 128 for ch in text)
    if font is None or ascii_only:
        cv2.putText(frame, text if ascii_only else "?", org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return
    from PIL import Image, ImageDraw
    x, y = org
    shaped = _shape(text)
    l, t, r, b = font.getbbox(shaped)
    tw, th = r - l + 8, b - t + 6
    y0 = max(y - th, 0)
    x0 = min(max(x, 0), max(frame.shape[1] - tw, 0))
    patch = frame[y0:y0 + th, x0:x0 + tw]
    if patch.size == 0:
        return
    img = Image.new("RGB", (patch.shape[1], patch.shape[0]), (int(color[2]), int(color[1]), int(color[0])))
    ImageDraw.Draw(img).text((4 - l, 3 - t), shaped, font=font, fill=(0, 0, 0))
    frame[y0:y0 + th, x0:x0 + tw] = np.asarray(img)[:, :, ::-1]
