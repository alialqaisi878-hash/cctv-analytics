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
from .discovery import sub_stream_url

log = logging.getLogger("cctv.pipeline")
PERSON = 0  # COCO class id
MIN_PERSON_H = 0.04    # boxes smaller than this part of the picture height are ignored
HOLD_S = 3.0           # a person the detector misses for a moment still counts for this long
GHOST_S = 600          # things seen weakly and never accepted are remembered this long (so they are not "new" again)
VIEW_WIDTH = 1280      # pictures shown/saved are scaled down to this width (cameras often send 4-8 MP)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class _Seen:
    """Something the detector reported in the same place over several analysed pictures."""
    __slots__ = ("box", "first_box", "first", "last", "hits", "strong", "motion", "changed", "ok", "tid")

    def __init__(self, box, ts):
        self.box = self.first_box = box
        self.first = self.last = ts
        self.hits = self.strong = self.motion = 0
        self.changed = self.ok = False
        self.tid = None


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, cfg: dict, db: Database, faces=None, notifier=None, maps=None):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam, self.cfg, self.db = cam, cfg, db
        self.faces, self.notifier, self.maps = faces, notifier, maps    # FaceEngine / Notifier / FloorMaps, optional
        self.id = cam["id"]
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self._analysed_jpeg: bytes | None = None    # marked picture of the last analysed frame (alerts)
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
        self._still_here: dict[str, float] = {}   # person id -> last time a sighting was written for this camera
        self._offline_since: float | None = None
        self._was_online = False
        self._raw_boxes: list = []              # this picture's detections before tracking (set by a model callback)
        self._seen: list[_Seen] = []            # what the detector keeps reporting (people once accepted)
        self._bg = None                         # slow-moving background (small, grey) to tell movement from furniture
        self._dt = 0.125                        # seconds between analyses (smoothed)
        self._last_ts = 0.0
        self.analysed_at = 0.0                  # when the last picture was analysed
        self.best_conf = 0.0                    # how sure the detector was about its best "person" lately
        self._best_at = 0.0
        self.last_error = ""                    # the last thing that went wrong while analysing (shown in the dashboard)
        self.errors = 0
        self._stats_at = time.time()
        self._tracks: list = []                 # latest tracks, used to mark the live picture between analyses
        self._frame: tuple | None = None        # (seq, frame) newest picture from the camera (reader thread)
        self._frame_ready = threading.Event()
        self.viewed_at = 0.0                    # last time the dashboard asked for the live picture
        self._view_seq = 0                      # frame number of the live picture last made
        self._view_lock = threading.Lock()
        self.stream = ""                        # "sub" (light stream) or "main"
        self._opened_for = ""
        self._no_sub = ""
        self.size = ""                          # e.g. "704x576"
        self.read_fps = 0.0                     # pictures per second arriving from the camera

    # ---------- public ----------
    def snapshot(self) -> dict:
        return {
            "id": self.id, "name": self.cam["name"], "status": self.status, "fps": round(self.fps, 1),
            "people_now": self.people_now, "stream": self.stream, "size": self.size,
            "read_fps": round(self.read_fps, 1),
            "analysed_ago": round(time.time() - self.analysed_at, 1) if self.analysed_at else None,
            "best_conf": round(self.best_conf, 2), "error": self.last_error,
            "lines": [{"id": l.line_id, "name": c["name"], "in": l.count_in, "out": l.count_out}
                      for l, c in zip(self.lines, self.cam["lines"])],
            "zones": [{"id": z.zone_id, "name": c["name"], "occupancy": z.occupancy}
                      for z, c in zip(self.zones, self.cam["zones"])],
        }

    def latest_jpeg(self) -> bytes | None:
        with self.lock:
            return self.jpeg

    def live_jpeg(self) -> bytes | None:
        """The newest camera picture for someone watching. Made here, on request, and not in the thread
        that reads the camera, so reading never slows down and pictures never pile up behind."""
        self.viewed_at = time.time()
        got = self._frame
        if got is None:
            return self.latest_jpeg()
        with self._view_lock:
            if got[0] != self._view_seq:
                jpg = self._render(got[1], self._tracks, mark_only=True)
                if jpg:
                    with self.lock:
                        self.jpeg = jpg
                self._view_seq = got[0]
        return self.latest_jpeg()

    def latest_raw(self) -> np.ndarray | None:
        with self.lock:
            return None if self.raw is None else self.raw.copy()

    def stop(self):
        self.stop_event.set()

    # ---------- internals ----------
    def _load_model(self):
        from ultralytics import YOLO  # imported lazily: heavy
        self._model = YOLO(self.cfg["detector"]["model"])
        # keep the plain detections too: the tracker hides a person until it has matched them twice, which never
        # happens when someone walks fast past a busy PC (few analyses a second), so presence must not depend on it
        def keep_raw(predictor):
            try:
                b = predictor.results[0].boxes
                self._raw_boxes = list(zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy().tolist()))
            except Exception:  # noqa: BLE001 - an older/newer library: fall back to the tracker's boxes
                self._raw_boxes = []
        self._model.add_callback("on_predict_postprocess_end", keep_raw)   # added before the tracker's: runs first

    def _want_stream(self) -> str:
        return self.cfg["detector"].get("stream", "fast")

    @staticmethod
    def _capture(src):
        """Open a stream, letting the graphics card decode it when it can (much less CPU)."""
        if isinstance(src, str) and "://" in src:
            params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]   # falls back to the CPU by itself
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000]
            try:
                return cv2.VideoCapture(src, cv2.CAP_FFMPEG, params)
            except (cv2.error, AttributeError, TypeError):
                pass
        return cv2.VideoCapture(src)

    def _open(self):
        src = self.cam["source"]
        src = int(src) if isinstance(src, int) or (isinstance(src, str) and src.isdigit()) else src
        tries = [("main", src)]
        self._opened_for = self._want_stream()
        if isinstance(src, str) and src.startswith("rtsp") and self._want_stream() == "fast":
            sub = sub_stream_url(src)
            if sub and sub != self._no_sub:
                tries.insert(0, ("sub", sub))
        cap = None
        for kind, url in tries:
            cap = self._capture(url)
            if cap.isOpened():
                self.stream = kind
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                self.size = f"{w}x{h}" if w and h else ""
                log.info("camera %s: %s stream %s", self.id, kind, self.size)
                break
            cap.release()
            if kind == "sub":
                self._no_sub = url          # this camera has no second stream: don't wait for it again
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
            try:
                self._loop(cap)
            except Exception as e:  # noqa: BLE001 - a camera must never stop for good because of one error
                self._note_error(e)
                self.stop_event.wait(3)
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

    def _note_error(self, e: Exception) -> None:
        self.errors += 1
        self.last_error = f"{type(e).__name__}: {e}"[:300]
        if self.errors <= 5 or self.errors % 200 == 0:      # the log must not fill up with the same error
            log.exception("camera %s: analysis error (%d so far)", self.id, self.errors)

    def _analyse(self, frame: np.ndarray, ts: float) -> None:
        """One picture through the analysis; an error in one picture never stops the camera."""
        try:
            self._process(frame, ts)
            self.analysed_at = time.time()
        except Exception as e:  # noqa: BLE001
            self._note_error(e)
            if self.errors % 20 == 0:                       # keeps failing: start the detector afresh
                try:
                    self._load_model()
                except Exception:  # noqa: BLE001
                    log.exception("camera %s: detector could not be reloaded", self.id)

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
                self.notifier.queue_alert("offline", f"⚠️ الكاميرا {self.cam['name']} مفصولة أو ما تبث صورة", None,
                                          {"camera_id": self.id, "camera": self.cam["name"]})

    def _is_file(self) -> bool:
        src = str(self.cam["source"])
        return not src.isdigit() and "://" not in src

    def _loop(self, cap):
        if not self._is_file():
            return self._loop_live(cap)
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
            self._analyse(frame, t0)
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

    def _loop_live(self, cap):
        """Cameras: one thread reads every frame as it arrives (so pictures never queue up and lag
        behind), shows the newest to whoever watches, and the analysis takes the newest when it is free."""
        alive = threading.Event()
        alive.set()

        def reader():
            # only reads: anything slower here makes the camera's pictures queue up and the view lag
            seq, t_prev = 0, time.time()
            src = str(self.cam["source"])
            has_sub = src.startswith("rtsp") and sub_stream_url(src) is not None
            while alive.is_set() and not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                seq += 1
                self._frame = (seq, frame)
                self._frame_ready.set()
                now = time.time()
                self.read_fps = 0.9 * self.read_fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now
                if seq % 50 == 0 and has_sub and self._want_stream() != self._opened_for:
                    break               # the quality setting changed: reconnect with the other stream
            alive.clear()
            self._frame_ready.set()

        t = threading.Thread(target=reader, daemon=True, name=f"read-{self.id}")
        t.start()
        target_dt = 1.0 / max(self.cfg["detector"]["process_fps"], 1)
        last_seq, t_prev = 0, time.time()
        try:
            while alive.is_set() and not self.stop_event.is_set():
                self._frame_ready.wait(5)
                self._frame_ready.clear()
                got = self._frame
                if got is None or got[0] == last_seq:
                    if not alive.is_set():
                        break
                    continue
                last_seq, frame = got
                t0 = time.time()
                self._offline_since = None
                self._was_online = True
                self._analyse(frame, t0)
                elapsed = time.time() - t0
                if elapsed < target_dt:
                    self.stop_event.wait(target_dt - elapsed)
                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now
        finally:
            alive.clear()
            t.join(timeout=5)
            self._frame = None

    def _moving(self, frame: np.ndarray):
        """Which parts of the picture changed lately (small mask), or None when that cannot be told:
        the first picture, or the whole picture changed at once (lights, the camera's night mode)."""
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self._bg is None:
            self._bg = g.astype(np.float32)
            return None
        mask = cv2.absdiff(g, cv2.convertScaleAbs(self._bg)) > 18
        cv2.accumulateWeighted(g, self._bg, 0.05)
        return None if mask.mean() > 0.5 else mask

    def _process(self, frame: np.ndarray, ts: float):
        det = self.cfg["detector"]
        h, w = frame.shape[:2]
        sure = det["confidence"]
        # the detector is asked for weaker guesses too: a person half out of the picture, bent over or seen from
        # above scores low. Weak guesses only count when they also move like a person (see below).
        low = min(sure, max(0.2, sure * 0.5))
        self._raw_boxes = []
        res = self._model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[PERSON],
                                conf=low, imgsz=det["imgsz"], device=det.get("device"),
                                verbose=False)[0]
        if self._last_ts:
            self._dt = 0.8 * self._dt + 0.2 * min(max(ts - self._last_ts, 0.01), 10)
        self._last_ts = ts
        hold = max(HOLD_S, 2.5 * self._dt)

        followed = []                           # the tracker's boxes: (id, box)
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist()):
                followed.append((tid, tuple(float(v) for v in box)))
        dets = [(tuple(float(v) for v in b), float(c)) for b, c in self._raw_boxes]
        if not dets:                            # no plain detections (another library version): the tracker's boxes
            dets = [(b, 1.0) for _, b in followed]
        best = max((c for _, c in dets), default=0.0)
        if best >= self.best_conf or ts - self._best_at > 5:
            self.best_conf, self._best_at = best, ts
        moving = self._moving(frame)

        def moved(b) -> bool:
            if moving is None:
                return False
            mh, mw = moving.shape
            x1, y1 = max(int(b[0] / w * mw), 0), max(int(b[1] / h * mh), 0)
            x2, y2 = min(int(b[2] / w * mw) + 1, mw), min(int(b[3] / h * mh) + 1, mh)
            part = moving[y1:y2, x1:x2]
            return bool(part.size) and float(part.mean()) > 0.04

        free = list(self._seen)
        for b, c in sorted(dets, key=lambda d: -d[1]):
            bw, bh = b[2] - b[0], b[3] - b[1]
            if c < low or bh < MIN_PERSON_H * h:
                continue
            o = max(free, key=lambda s: _iou(b, s.box), default=None)
            if o is None or _iou(b, o.box) <= 0.3:
                # walking fast past a slow PC: the next box does not touch the last one, but it is close by
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                near = [s for s in free if ts - s.last <= hold and
                        abs((s.box[0] + s.box[2]) / 2 - cx) < 1.5 * bh and abs((s.box[1] + s.box[3]) / 2 - cy) < bh]
                o = min(near, key=lambda s: abs((s.box[0] + s.box[2]) / 2 - cx), default=None)
            if o is None:
                o = _Seen(b, ts)
                self._seen.append(o)
            else:
                free.remove(o)
                if ts - o.last > max(0.5, 3.5 * self._dt):   # back after a gap: it has to prove itself again from the start
                    o.first, o.hits, o.strong, o.motion, o.changed = ts, 0, 0, 0, False
            o.box, o.last = b, ts
            o.hits += 1
            # sure and shaped like a person (taller than wide: drops desks, bags, shadows)
            if c >= sure and bh >= 0.8 * bw:
                # the tracker only gives an id to what it matched twice already, so with an id once is enough
                o.strong += 2 if any(_iou(b, fb) > 0.5 for _, fb in followed) else 1
            o.motion += moved(b)
            o.changed = o.changed or _iou(b, o.first_box) < 0.5
            # a person: the detector is sure twice, or it is less sure but the thing keeps being seen, moves and
            # is not where/how it first appeared (furniture the detector half-believes in never does that)
            if not o.ok and (o.strong >= 2 or (o.hits >= 3 and o.motion >= 2 and o.changed and ts - o.first >= 0.4)):
                o.ok = True
        self._seen = [s for s in self._seen if ts - s.last <= (hold if s.ok else GHOST_S)][-200:]
        people = [s for s in self._seen if s.ok]

        tracks = []                             # followed people (with an id): used for lines, zones and faces
        for s in people:
            s.tid = None
        for tid, b in followed:
            s = max(people, key=lambda p: _iou(b, p.box), default=None)
            if s is not None and s.tid is None and _iou(b, s.box) > 0.3:
                s.tid = tid
                tracks.append((tid, b, ((b[0] + b[2]) / 2 / w, b[3] / h)))  # bottom-centre = where the person stands
        # everyone accepted as a person, also before the tracker has an id for them (so they are counted and blurred)
        shown = list(tracks) + [(None, s.box, ((s.box[0] + s.box[2]) / 2 / w, s.box[3] / h))
                                for s in people if s.tid is None and ts - s.last <= max(1.0, 2.5 * self._dt)]
        self.people_now = len(people)
        self._log_stats(ts)
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
        self._tracks = shown
        jpg = self._render(frame, shown)
        with self.lock:
            if not (time.time() - self.viewed_at < 3 and self._frame is not None):
                self.jpeg = jpg                 # the reader thread keeps the live picture while it is watched
            self._analysed_jpeg = jpg
        self._flush_alerts(ts)

    # ---------- face recognition (only people enrolled by the owner) ----------
    def _recognise(self, frame: np.ndarray, tracks, ts: float) -> None:
        alive = {t[0] for t in tracks}
        for tid in list(self.names):
            if tid not in alive:
                self.names.pop(tid, None)
                self._unknown_seen.pop(tid, None)
        # people who stay in view keep their place on the map fresh (one row a minute, no new face check)
        for tid in alive:
            n = self.names.get(tid) or {}
            pid = n.get("person_id")
            if pid and ts - self._still_here.get(pid, ts) >= 60:
                self._still_here[pid] = ts
                self.db.add_sighting(ts, self.id, pid, n["name"], n.get("score", 0.0))
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
            self.names[tid] = {"person_id": p["id"], "name": p["name"], "score": f["score"]}
            last = self.db.last_sighting(p["id"])     # a new place, or back after being away, is worth telling
            moved = (not last or last["camera_id"] != self.id
                     or ts - last["ts"] > 60 * self.cfg["faces"].get("present_minutes", 15))
            self.db.add_sighting(ts, self.id, p["id"], p["name"], f["score"])
            self._still_here[p["id"]] = ts
            if moved and self.notifier:
                self.notifier.queue_location(f"{p['name']} انشاف في {self.cam['name']}", self._meta(p["id"]))
            log.info("recognised %s on %s (%.2f)", p["name"], self.id, f["score"])
            if p.get("alert_on_seen"):
                self._queue_alert(ts, f"person:{p['id']}", f"تم رصد {p['name']} في {self.cam['name']}", p["id"])

    def _meta(self, pid: str | None) -> dict:
        """What a webhook (n8n / ERP) gets besides the text: the camera, its place on the map and the person."""
        person = self.faces.person(pid) if (pid and self.faces) else None
        return {"camera_id": self.id, "camera": self.cam["name"],
                "location": self.maps.where(self.id) if self.maps else None,
                "person": {"id": pid, "name": person["name"], "erp_id": person.get("erp_id", ""),
                           "job": person.get("job", "")} if person else None}

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
        jpg = self._analysed_jpeg if self.cfg["privacy"].get("save_alert_pictures", True) else None
        for kind, msg, pid in self._pending:
            self.db.add_alert(ts, self.id, kind.split(":", 1)[0], msg, jpg, pid)
            log.info("ALERT %s %s: %s", self.id, kind, msg)
            if self.notifier:
                t = self.db.local(ts).strftime("%H:%M")
                self.notifier.queue_alert(kind, f"🔔 {msg}\n{t}", self._analysed_jpeg, self._meta(pid))
        self._pending.clear()

    def _log_stats(self, ts: float) -> None:
        """Once a minute: what this camera is doing, so a problem on the owner's PC can be read from app.log."""
        if ts - self._stats_at < 60:
            return
        self._stats_at = ts
        log.info("camera %s: %s stream %s, reading %.1f/s, analysing %.1f/s, people %d, best guess %.2f, errors %d",
                 self.id, self.stream or "-", self.size or "-", self.read_fps, self.fps, self.people_now,
                 self.best_conf, self.errors)

    def _check_alerts(self, ts: float):
        al = self.cfg["alerts"]
        if al["after_hours_presence"] and self.people_now > 0 and not self._in_working_hours(ts):
            self._queue_alert(ts, "after_hours", f"تم رصد {self.people_now} شخص خارج أوقات الدوام في {self.cam['name']}")
        if al["max_occupancy"] and self.people_now > al["max_occupancy"]:
            self._queue_alert(ts, "crowding", f"ازدحام: {self.people_now} أشخاص في {self.cam['name']} (الحد {al['max_occupancy']})")

    def _in_working_hours(self, ts: float) -> bool:
        return in_working_hours(self.cfg, self.db.local(ts))

    def _render(self, frame: np.ndarray, tracks, mark_only: bool = False) -> bytes:
        """The picture people see: scaled down, people boxed (heads blurred) and the lines/zones drawn.
        mark_only: a newer frame than the tracks (live view), so blur a little wider to cover movement."""
        h, w = frame.shape[:2]
        k = min(1.0, VIEW_WIDTH / w)
        small = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
        if k < 1 or mark_only:
            pad = 0.12 if mark_only else 0.0
            scaled = []
            for tid, (x1, y1, x2, y2), foot in tracks:
                bw, bh = (x2 - x1) * pad, (y2 - y1) * pad
                scaled.append((tid, ((x1 - bw) * k, (y1 - bh) * k, (x2 + bw) * k, (y2 + bh) * k), foot))
            tracks = scaled
        self._annotate(small, tracks)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else b""

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
                if tid is not None:
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
