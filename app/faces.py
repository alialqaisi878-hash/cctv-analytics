"""Face recognition of people the owner enrolled (off by default).

Uses OpenCV's YuNet face detector and SFace recogniser (small ONNX models, MIT / Apache-2.0),
downloaded once into data/models when the feature is switched on. Only the people added on the
"الأشخاص" page are recognised; everyone else is shown as unknown. Nothing is sent to the internet.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cctv.faces")

_ZOO = "opencv/opencv_zoo/main/models"
MODELS = {
    "yunet": ("face_detection_yunet/face_detection_yunet_2023mar.onnx",
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "sface": ("face_recognition_sface/face_recognition_sface_2021dec.onnx",
              "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
MIRRORS = ["https://media.githubusercontent.com/media/" + _ZOO + "/{path}",
           "https://github.com/" + _ZOO.replace("/main/", "/raw/main/") + "/{path}"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_image(data: str | bytes) -> np.ndarray | None:
    """A picture from the dashboard: raw bytes or a data: URL / base64 text."""
    if isinstance(data, str):
        data = base64.b64decode(data.split(",", 1)[1] if data.startswith("data:") else data)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img


class FaceEngine:
    """Detector + recogniser + the list of enrolled people. Safe to share between camera threads."""

    def __init__(self, data_dir: str | Path, threshold: float = 0.40):
        self.dir = Path(data_dir)
        self.model_dir = self.dir / "models"
        self.people_dir = self.dir / "faces"
        self.threshold = threshold
        self.status = "off"            # off | downloading | ready | error: ...
        self._lock = threading.Lock()
        self._det = self._rec = None
        self._det_size = None
        self.people: list[dict] = []
        self._matrix = np.zeros((0, 128), np.float32)
        self._owners: list[int] = []
        self._load_people()

    # ---------- models ----------
    def ensure_ready(self, background: bool = True) -> None:
        if self.status in ("ready", "downloading"):
            return
        if background:
            self.status = "downloading"
            threading.Thread(target=self._prepare, daemon=True, name="face-models").start()
        else:
            self._prepare()

    def _prepare(self) -> None:
        try:
            self.status = "downloading"
            self.model_dir.mkdir(parents=True, exist_ok=True)
            paths = {k: self._fetch(k) for k in MODELS}
            with self._lock:
                # loaded from memory: OpenCV cannot open files in folders with Arabic names on Windows
                empty = np.array([], np.uint8)
                self._det = cv2.FaceDetectorYN.create("onnx", np.fromfile(paths["yunet"], np.uint8), empty,
                                                      (320, 320), 0.75, 0.3, 500)
                self._rec = cv2.FaceRecognizerSF.create("onnx", np.fromfile(paths["sface"], np.uint8), empty)
                self._det_size = None
            self.status = "ready"
            log.info("face recognition ready (%d people)", len(self.people))
        except Exception as e:  # noqa: BLE001
            self.status = f"error: {e}"
            log.exception("face models could not be prepared")

    def _fetch(self, key: str) -> Path:
        rel, sha = MODELS[key]
        dest = self.model_dir / Path(rel).name
        if dest.exists() and _sha256(dest) == sha:
            return dest
        last = None
        for mirror in MIRRORS:
            url = mirror.format(path=rel)
            tmp = dest.with_suffix(".part")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "cctv-analytics"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
                if _sha256(tmp) != sha:
                    raise ValueError("downloaded file is damaged")
                tmp.replace(dest)
                return dest
            except Exception as e:  # noqa: BLE001
                last = e
                tmp.unlink(missing_ok=True)
        raise RuntimeError(f"تعذر تنزيل نموذج الوجوه ({last})")

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    # ---------- detection / features ----------
    def _detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self._det_size != (w, h):
            self._det.setInputSize((w, h))
            self._det_size = (w, h)
        _, faces = self._det.detect(img)
        return faces if faces is not None else np.zeros((0, 15), np.float32)

    def analyse(self, img: np.ndarray, min_size: int = 24) -> list[dict]:
        """Every face in a picture: box (x1,y1,x2,y2 in pixels), best match and its score."""
        if not self.ready:
            return []
        out = []
        with self._lock:
            for f in self._detect(img):
                x, y, w, h = (float(v) for v in f[:4])
                if min(w, h) < min_size:
                    continue      # too small to recognise reliably
                feat = self._rec.feature(self._rec.alignCrop(img, f)).flatten()
                feat = feat / (np.linalg.norm(feat) + 1e-9)
                person, score = self._match(feat)
                out.append({"box": (x, y, x + w, y + h), "person": person, "score": score, "feature": feat,
                            "det_score": float(f[14])})
        return out

    def _match(self, feat: np.ndarray) -> tuple[dict | None, float]:
        if not len(self._owners):
            return None, 0.0
        sims = self._matrix @ feat
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (self.people[self._owners[i]] if score >= self.threshold else None), score

    # ---------- enrolled people ----------
    def _db_file(self) -> Path:
        return self.people_dir / "people.json"

    def _load_people(self) -> None:
        f = self._db_file()
        try:
            self.people = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        except (OSError, ValueError):
            log.exception("could not read %s", f)
            self.people = []
        self._rebuild()

    def _save_people(self) -> None:
        self.people_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._db_file().with_suffix(".tmp")
        tmp.write_text(json.dumps(self.people, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._db_file())

    def _rebuild(self) -> None:
        feats, owners = [], []
        for i, p in enumerate(self.people):
            for e in p.get("embeddings", []):
                feats.append(e)
                owners.append(i)
        self._matrix = np.array(feats, np.float32).reshape(-1, 128)
        self._owners = owners

    def public_people(self) -> list[dict]:
        return [{"id": p["id"], "name": p["name"], "photos": len(p.get("embeddings", [])),
                 "alert_on_seen": p.get("alert_on_seen", False), "note": p.get("note", ""),
                 "erp_id": p.get("erp_id", ""), "job": p.get("job", ""), "added": p.get("added")} for p in self.people]

    def person(self, pid: str) -> dict | None:
        return next((p for p in self.people if p["id"] == pid), None)

    def _best_face(self, img: np.ndarray) -> dict:
        faces = self.analyse(img, min_size=40)
        if not faces:
            raise ValueError("ما لگيت وجه واضح بالصورة. استخدم صورة قريبة والوجه مقابل الكاميرا.")
        return max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))

    def enroll(self, name: str, img: np.ndarray, person_id: str | None = None, alert_on_seen: bool = False,
               note: str = "", erp_id: str = "", job: str = "") -> dict:
        """Add a person (or one more photo of an existing person). Raises ValueError with an Arabic message."""
        if not self.ready:
            raise ValueError("التعرف على الوجه غير جاهز بعد. فعّله من الإعدادات وانتظر دقيقة.")
        face = self._best_face(img)
        with self._lock:
            if person_id:
                p = self.person(person_id)
                if p is None:
                    raise ValueError("الشخص غير موجود.")
                if len(p["embeddings"]) >= 20:
                    raise ValueError("وصلت الحد (20 صورة للشخص).")
            else:
                name = name.strip()[:60]
                if not name:
                    raise ValueError("اكتب الاسم.")
                p = {"id": uuid.uuid4().hex[:10], "name": name, "embeddings": [], "alert_on_seen": bool(alert_on_seen),
                     "note": note.strip()[:120], "erp_id": str(erp_id or "").strip()[:40],
                     "job": str(job or "").strip()[:80], "added": time.strftime("%Y-%m-%d")}
                self.people.append(p)
            p["embeddings"].append([round(float(v), 5) for v in face["feature"]])
            self._save_thumb(p["id"], img, face["box"], first=len(p["embeddings"]) == 1)
            self._rebuild()
            self._save_people()
        return {"id": p["id"], "name": p["name"], "photos": len(p["embeddings"])}

    def _save_thumb(self, pid: str, img: np.ndarray, box, first: bool) -> None:
        if not first:
            return
        x1, y1, x2, y2 = box
        m = (x2 - x1) * 0.35
        h, w = img.shape[:2]
        crop = img[max(int(y1 - m), 0):min(int(y2 + m), h), max(int(x1 - m), 0):min(int(x2 + m), w)]
        if crop.size:
            crop = cv2.resize(crop, (160, int(160 * crop.shape[0] / crop.shape[1])))
            self.people_dir.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", crop)    # (cv2.imwrite fails in non-English folders on Windows)
            if ok:
                (self.people_dir / f"{pid}.jpg").write_bytes(buf.tobytes())

    def thumb(self, pid: str) -> Path | None:
        p = self.people_dir / f"{pid}.jpg"
        return p if p.exists() and self.person(pid) else None

    def update_person(self, pid: str, changes: dict) -> dict:
        with self._lock:
            p = self.person(pid)
            if p is None:
                raise ValueError("الشخص غير موجود.")
            if "name" in changes and str(changes["name"]).strip():
                p["name"] = str(changes["name"]).strip()[:60]
            if "alert_on_seen" in changes:
                p["alert_on_seen"] = bool(changes["alert_on_seen"])
            if "note" in changes:
                p["note"] = str(changes["note"]).strip()[:120]
            if "erp_id" in changes:     # the employee's number in the company's ERP, used to match records
                p["erp_id"] = str(changes["erp_id"] or "").strip()[:40]
            if "job" in changes:
                p["job"] = str(changes["job"] or "").strip()[:80]
            self._save_people()
        return p

    def remove(self, pid: str) -> bool:
        with self._lock:
            before = len(self.people)
            self.people = [p for p in self.people if p["id"] != pid]
            if len(self.people) == before:
                return False
            self._rebuild()
            self._save_people()
        (self.people_dir / f"{pid}.jpg").unlink(missing_ok=True)
        return True
