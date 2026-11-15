"""Floor plans of the workplace with the cameras placed on them.

The owner uploads a picture of each floor (a drawing, a photo of the plan or a screenshot) and puts
each camera where it hangs. An enrolled person is then shown at the camera that recognised them last.
Positions are fractions of the picture (0..1), so they stay right whatever size the page shows it at.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cctv.map")
MAX_SIDE = 2400


class FloorMaps:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir) / "maps"
        self._lock = threading.Lock()
        self.maps: list[dict] = []          # [{id, name, w, h}]
        self.cams: dict[str, dict] = {}     # camera id -> {map, x, y}
        self._load()

    def _file(self) -> Path:
        return self.dir / "maps.json"

    def _load(self) -> None:
        try:
            if self._file().exists():
                d = json.loads(self._file().read_text(encoding="utf-8"))
                self.maps, self.cams = list(d.get("maps", [])), dict(d.get("cams", {}))
        except (OSError, ValueError):
            log.exception("could not read %s", self._file())

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._file().with_suffix(".tmp")
        tmp.write_text(json.dumps({"maps": self.maps, "cams": self.cams}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._file())

    def public(self) -> dict:
        return {"maps": [dict(m) for m in self.maps], "cams": {k: dict(v) for k, v in self.cams.items()}}

    def get(self, map_id: str) -> dict | None:
        return next((m for m in self.maps if m["id"] == map_id), None)

    def add(self, name: str, img: np.ndarray) -> dict:
        """Add a floor plan. Raises ValueError with an Arabic message."""
        if img is None or not img.size:
            raise ValueError("الصورة غير صالحة. اختر صورة JPG أو PNG للمخطط.")
        name = (name or "").strip()[:60] or f"مخطط {len(self.maps) + 1}"
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            k = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])   # also drops anything hidden in the file
        if not ok:
            raise ValueError("ما گدرت أحفظ صورة المخطط.")
        m = {"id": uuid.uuid4().hex[:10], "name": name, "w": int(w), "h": int(h)}
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{m['id']}.jpg").write_bytes(buf.tobytes())
            self.maps.append(m)
            self._save()
        return m

    def rename(self, map_id: str, name: str) -> dict:
        with self._lock:
            m = self.get(map_id)
            if m is None:
                raise ValueError("المخطط غير موجود.")
            if name.strip():
                m["name"] = name.strip()[:60]
            self._save()
        return m

    def remove(self, map_id: str) -> bool:
        with self._lock:
            if self.get(map_id) is None:
                return False
            self.maps = [m for m in self.maps if m["id"] != map_id]
            self.cams = {k: v for k, v in self.cams.items() if v.get("map") != map_id}
            self._save()
        (self.dir / f"{map_id}.jpg").unlink(missing_ok=True)
        return True

    def image(self, map_id: str) -> Path | None:
        p = self.dir / f"{map_id}.jpg"
        return p if self.get(map_id) and p.exists() else None

    def place(self, cam_id: str, map_id: str | None, x: float = 0.5, y: float = 0.5) -> None:
        """Put a camera on a floor plan, or take it off when map_id is empty."""
        with self._lock:
            if not map_id:
                self.cams.pop(cam_id, None)
            else:
                if self.get(map_id) is None:
                    raise ValueError("المخطط غير موجود.")
                self.cams[cam_id] = {"map": map_id, "x": round(min(max(float(x), 0.0), 1.0), 4),
                                     "y": round(min(max(float(y), 0.0), 1.0), 4)}
            self._save()

    def forget_camera(self, cam_id: str) -> None:
        if cam_id in self.cams:
            with self._lock:
                self.cams.pop(cam_id, None)
                self._save()

    def where(self, cam_id: str | None) -> dict | None:
        """The floor and point of a camera: {map_id, map, x, y}, or None if it is not on a plan."""
        c = self.cams.get(cam_id or "")
        m = self.get(c["map"]) if c else None
        return {"map_id": m["id"], "map": m["name"], "x": c["x"], "y": c["y"]} if m else None
