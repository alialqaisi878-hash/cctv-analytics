"""One person detector shared by all cameras.

Every camera used to run its own copy of the model in its own thread. With many cameras (Ali has 32) those
copies fought over the processor: each analysed a picture every few seconds or not at all, so nobody was
ever counted. Now one thread runs one model for everybody, several pictures at a time, and the cameras
where something moves (or where people already are) go first. A camera where nothing happens is still
looked at every few seconds.
"""
from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger("cctv.detector")

PERSON = 0  # COCO class id


class _Request:
    __slots__ = ("cam", "frame", "prio", "at", "done", "result", "error")

    def __init__(self, cam: str, frame, prio: float):
        self.cam, self.frame, self.prio, self.at = cam, frame, prio, time.time()
        self.done = threading.Event()
        self.result = None
        self.error: Exception | None = None


class SharedDetector:
    def __init__(self, cfg: dict, model=None, batch: int | None = None):
        self.cfg = cfg
        self._model = model                     # tests hand in a fake model; the real one loads in the thread
        cores = os.cpu_count() or 4
        self.batch = batch or max(2, min(8, cores // 2))
        self._pending: dict[str, _Request] = {}
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.status = "starting"
        self.rate = 0.0                         # pictures analysed per second, all cameras together
        self.busy = 0.0                         # share of the time the detector is working (1.0 = flat out)
        self.last_error = ""
        self._count = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="detector")
        self._thread.start()

    # ---------- used by the cameras ----------
    def detect(self, cam: str, frame, prio: float, timeout: float = 3.0):
        """[(box, sureness), ...] for this picture, or None when a newer picture of the same camera replaced it
        or nothing was free in time (the camera then simply tries again with its newest picture)."""
        req = _Request(cam, frame, prio)
        with self._cond:
            old = self._pending.pop(cam, None)
            if old is not None:
                old.done.set()                  # an older picture of this camera: not worth analysing any more
            self._pending[cam] = req
            self._cond.notify()
        if not req.done.wait(timeout):
            with self._cond:
                if self._pending.get(cam) is req:
                    del self._pending[cam]
            return None
        if req.error is not None:
            raise req.error
        return req.result

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def snapshot(self) -> dict:
        return {"status": self.status, "rate": round(self.rate, 1), "busy": round(self.busy, 2),
                "batch": self.batch, "waiting": len(self._pending), "error": self.last_error}

    # ---------- the detector thread ----------
    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            # one thread runs the model, so it may use every core; the cameras' own threads only read and draw
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        except Exception:  # noqa: BLE001
            pass
        from ultralytics import YOLO  # imported lazily: heavy
        self._model = YOLO(self.cfg["detector"]["model"])

    def _run(self) -> None:
        try:
            self._load()
            self.status = "ready"
        except Exception as e:  # noqa: BLE001
            self.status = f"model error: {e}"
            self.last_error = str(e)
            log.exception("detector model could not be loaded")
            with self._cond:                    # let the cameras show the error instead of waiting forever
                for r in self._pending.values():
                    r.error = RuntimeError(self.status)
                    r.done.set()
                self._pending.clear()
            return
        t_prev, work = time.time(), 0.0
        while not self._stop.is_set():
            with self._cond:
                while not self._pending and not self._stop.is_set():
                    self._cond.wait(1.0)
                if self._stop.is_set():
                    return
            time.sleep(0.01)                    # a moment for other cameras' pictures to join this batch
            now = time.time()
            with self._cond:
                # what moves or has people first; the longer a camera waited, the higher it climbs
                order = sorted(self._pending.values(), key=lambda r: r.prio + (now - r.at), reverse=True)
                take = order[:self.batch]
                for r in take:
                    del self._pending[r.cam]
            t0 = time.time()
            try:
                out = self._infer([r.frame for r in take])
                for r, dets in zip(take, out):
                    r.result = dets
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                log.exception("detector failed on %d pictures", len(take))
                for r in take:
                    r.error = e
            for r in take:
                r.done.set()
            t1 = time.time()
            work += t1 - t0
            self._count += len(take)
            if t1 - t_prev >= 2:
                self.rate = 0.7 * self.rate + 0.3 * (self._count / (t1 - t_prev))
                self.busy = 0.7 * self.busy + 0.3 * min(work / (t1 - t_prev), 1.0)
                t_prev, work, self._count = t1, 0.0, 0

    def _infer(self, frames) -> list[list]:
        det = self.cfg["detector"]
        sure = det["confidence"]
        low = min(sure, max(0.2, sure * 0.5))   # weaker guesses too: the cameras decide which of them are people
        res = self._model.predict(frames, classes=[PERSON], conf=low, imgsz=det["imgsz"], device=det.get("device"),
                                  verbose=False)
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append([])
                continue
            out.append([(tuple(float(v) for v in xy), float(c))
                        for xy, c in zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy().tolist())])
        return out
