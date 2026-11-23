"""Fire and smoke: a second, small model looks at every camera every few seconds and alerts at once.

It runs in its own thread next to the people detector and takes only the pictures the cameras already have,
so it adds little work: with 32 cameras and a look every 5 seconds that is about 6 pictures a second,
done in batches. One sighting is not enough: flames or smoke must be seen in 2 of the last 3 looks at the
same camera, so a red light or a reflection in one picture does not wake anyone up at night.

Model: YOLOv8n trained on fire and smoke by luminous0219 (github.com/luminous0219/fire-and-smoke-detection-yolov8,
AGPL-3.0, see app/models/fire/LICENSE). Classes: 0 fire, 1 smoke (read from the model itself).
"""
from __future__ import annotations

import collections
import hashlib
import logging
import threading
import time
from pathlib import Path

import numpy as np

from .detector import SharedDetector

log = logging.getLogger("cctv.hazard")

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(__file__).resolve().parent / "models" / "fire" / "fire_smoke.pt"
MODEL_SHA256 = "ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16"
NEED, OF = 2, 3          # seen in 2 of the last 3 looks at a camera before alerting
WORDS = {"fire": "🔥 نار", "smoke": "💨 دخان"}


def _kind(name: str) -> str | None:
    n = str(name).lower()
    return "fire" if ("fire" in n or "flame" in n) else ("smoke" if "smoke" in n else None)


class HazardWatch:
    def __init__(self, cfg: dict, model=None):
        self.cfg = cfg
        self._model = model                 # tests pass a fake model
        self._pending: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._looked: dict[str, float] = {}
        self._seen: dict[str, collections.deque] = {}
        self.status = "off"
        self.engine = ""
        self.last_error = ""
        self.looks = 0
        self.rate = 0.0
        self._thread: threading.Thread | None = None

    @property
    def h(self) -> dict:
        return self.cfg["hazard"]

    @property
    def enabled(self) -> bool:
        return bool(self.h.get("enabled"))

    # ---------- called by the cameras ----------
    def offer(self, worker, frame: np.ndarray, ts: float) -> None:
        """A camera's newest picture. Taken only when that camera is due a look; never waits."""
        if not self.enabled or frame is None:
            return
        cam = worker.id
        if ts - self._looked.get(cam, 0.0) < float(self.h.get("every_s", 5)):
            return
        self._looked[cam] = ts
        with self._lock:
            self._pending[cam] = (worker, frame, ts)
        self._wake.set()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="hazard")
            self._thread.start()

    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "status": self.status, "engine": self.engine, "looks": self.looks,
                "rate": round(self.rate, 1), "error": self.last_error}

    # ---------- the watching thread ----------
    def _load(self):
        if self._model is not None:
            self.engine = self.engine or "test"
            return
        path = Path(self.h.get("model") or MODEL)
        if not path.is_absolute():
            path = ROOT / path
        if path.resolve() == MODEL.resolve():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != MODEL_SHA256:      # the file is a pickle: never load a changed or broken copy
                raise RuntimeError("fire model file is damaged or changed (checksum does not match)")
        from ultralytics import YOLO
        imgsz = int(self.h.get("imgsz", 640))
        self._model, engine = YOLO(str(path), task="detect"), "pytorch"
        if SharedDetector._wants_openvino(self.cfg["detector"]):
            self.status = "preparing the fast engine (first start only)"
            try:
                fast = YOLO(str(SharedDetector._openvino_model(self._model, {"model": str(path), "imgsz": imgsz})),
                            task="detect")
                fast.predict([np.zeros((imgsz, imgsz, 3), np.uint8)], imgsz=imgsz, verbose=False)
                self._model, engine = fast, "openvino"
            except Exception as e:  # noqa: BLE001
                self.last_error = f"fast engine not used: {type(e).__name__}: {e}"[:300]
                log.exception("fire model: OpenVINO not used")
        self.engine = engine

    def _run(self) -> None:
        self.status = "loading"
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            self.status, self.last_error = "not working", f"{type(e).__name__}: {e}"[:300]
            log.exception("fire model could not be loaded")
            self._thread = None
            return
        self.status = "watching"
        while True:
            self._wake.wait(5)
            self._wake.clear()
            if not self.enabled:
                with self._lock:
                    self._pending.clear()
                continue
            with self._lock:
                batch = list(self._pending.values())[:8]
                for w, _, _ in batch:
                    self._pending.pop(w.id, None)
                if self._pending:
                    self._wake.set()
            if not batch:
                continue
            t0 = time.time()
            try:
                found = self._infer([f for _, f, _ in batch])
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                log.exception("fire model failed on a batch")
                time.sleep(2)
                continue
            each = (time.time() - t0) / len(batch)
            self.rate = 0.8 * self.rate + 0.2 * (1.0 / max(each, 1e-3))
            self.looks += len(batch)
            for (w, frame, ts), hits in zip(batch, found):
                self._judge(w, frame, ts, hits)

    def _infer(self, frames) -> list[list[tuple]]:
        """For each picture: [(kind, confidence, (x1, y1, x2, y2)), ...] above that kind's threshold."""
        h = self.h
        sure = {"fire": float(h.get("fire_confidence", 0.45)), "smoke": float(h.get("smoke_confidence", 0.5))}
        want = set(k for k in ("fire", "smoke") if h.get(k, True))
        res = self._model.predict(frames, conf=min(sure.values()), imgsz=int(h.get("imgsz", 640)), verbose=False)
        out = []
        for frame, r in zip(frames, res):
            names = getattr(r, "names", None) or getattr(self._model, "names", {}) or {}
            fh, fw = frame.shape[:2]
            hits = []
            b = r.boxes
            if b is not None and len(b):
                for xyxy, c, k in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                    kind = _kind(names.get(int(k), ""))
                    if kind not in want or float(c) < sure[kind]:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in xyxy)
                    if (x2 - x1) * (y2 - y1) < float(h.get("min_area", 0.0005)) * fw * fh:
                        continue                    # a few pixels: noise
                    hits.append((kind, float(c), (x1, y1, x2, y2)))
            out.append(hits)
        return out

    def _judge(self, worker, frame, ts: float, hits: list[tuple]) -> None:
        seen = self._seen.setdefault(worker.id, collections.deque(maxlen=OF))
        seen.append({k for k, _, _ in hits})
        for kind in ("fire", "smoke"):
            if sum(kind in s for s in seen) >= NEED and any(k == kind for k, _, _ in hits):
                best = max(c for k, c, _ in hits if k == kind)
                try:
                    worker.hazard_alert(ts, kind, best, frame, [b for k, _, b in hits if k == kind])
                except Exception:  # noqa: BLE001
                    log.exception("fire alert failed")
