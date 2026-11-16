"""FastAPI web dashboard + JSON API."""
from __future__ import annotations

import base64
import copy
import csv
import io
import ipaddress
import logging
import secrets
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import discovery
from .agent import Analyst
from .analytics import LineCounter
from .config import (add_local_camera, apply_settings, in_working_hours, local_camera_ids, normalize_camera,
                     notice_text, public_settings, remove_local_camera, set_camera_lines)
from .db import Database
from .faces import FaceEngine, decode_image
from .floormap import FloorMaps
from .notify import Notifier
from .pipeline import CameraWorker
from .updater import Updater
from .version import __version__

log = logging.getLogger("cctv.server")
STATIC = Path(__file__).parent / "static"


class App:
    def __init__(self, cfg: dict, start_cameras: bool = True):
        self.cfg = cfg
        self.db = Database(cfg["database"], cfg["timezone"])
        self.analyst = Analyst(self.db, cfg)
        self.faces = FaceEngine(Path(cfg["database"]).parent, cfg["faces"]["threshold"])
        if cfg["faces"]["enabled"]:
            self.faces.ensure_ready()
        self.notifier = Notifier(cfg)
        self.maps = FloorMaps(Path(cfg["database"]).parent)
        self.workers: dict[str, CameraWorker] = {}
        if start_cameras:
            for cam in cfg["cameras"]:
                if cam["enabled"]:
                    self.start_worker(cam)
        threading.Thread(target=self._housekeeping, daemon=True, name="housekeeping").start()
        self.last_seen = time.time()      # last request from a dashboard (see /api/window/state)
        self.closed_at = 0.0              # when a dashboard page last said it was being closed
        self.updater = Updater(cfg)
        self.updater.run_background(lambda: in_working_hours(cfg, self.db.local(time.time())))

    def start_worker(self, cam: dict) -> CameraWorker:
        w = CameraWorker(cam, self.cfg, self.db, self.faces, self.notifier, self.maps)
        self.workers[cam["id"]] = w
        w.start()
        return w

    def _housekeeping(self):
        """Retention cleanup every hour + scheduled daily report."""
        last_purge, last_report_day = 0.0, None
        while True:
            now = time.time()
            if now - last_purge > 3600:
                n = self.db.purge_older_than(self.cfg["privacy"]["retention_days"])
                if n:
                    log.info("retention: deleted %d old rows", n)
                last_purge = now
            rt = self.cfg["agent"].get("daily_report_time")
            if rt:
                local = self.db.local(now)
                if local.strftime("%H:%M") >= rt and last_report_day != local.strftime("%Y-%m-%d"):
                    last_report_day = local.strftime("%Y-%m-%d")
                    if not any(r["day"] == last_report_day and r["kind"] == "daily" for r in self.db.reports(50)):
                        try:
                            self.analyst.daily_report(last_report_day)
                        except Exception:  # noqa: BLE001
                            log.exception("scheduled report failed")
            time.sleep(30)


def create_app(cfg: dict, start_cameras: bool = True) -> FastAPI:
    core = App(cfg, start_cameras)
    db = core.db
    api = FastAPI(title="CCTV Analytics")
    api.state.core = core
    api.mount("/static", StaticFiles(directory=STATIC), name="static")

    @api.middleware("http")
    async def basic_auth(request: Request, call_next):
        if request.url.path.startswith("/api/window/"):
            # used by the desktop launcher on this PC only; not counted as dashboard activity
            if not (request.client and request.client.host in ("127.0.0.1", "::1")):
                return Response("local only", 403)
            return await call_next(request)
        core.last_seen = time.time()
        password = cfg["server"].get("password") or ""   # read every time: it can be changed in the settings
        if password:
            auth = request.headers.get("authorization", "")
            ok = False
            if auth.startswith("Basic "):
                try:
                    _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                    ok = secrets.compare_digest(pw.encode(), password.encode())
                except Exception:  # noqa: BLE001
                    ok = False
            if not ok:
                return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="CCTV"'})
        return await call_next(request)

    def day_or_today(day: str | None) -> str:
        return day or db.today()

    @api.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @api.get("/api/info")
    def info():
        p = cfg["privacy"]
        return {"site_name": cfg["site_name"], "today": db.today(), "ai_enabled": core.analyst.ai_enabled,
                "ai_provider": core.analyst.provider, "version": __version__,
                "unseen_alerts": db.unseen_count(), "last_alert_id": db.last_alert_id(),
                "faces": {"enabled": cfg["faces"]["enabled"], "status": core.faces.status,
                          "people": len(core.faces.people)},
                "notify": {"enabled": core.notifier.enabled, "last_error": core.notifier.last_error},
                "privacy": {"show_notice": p["show_notice"], "notice_text": notice_text(cfg),
                            "retention_days": p["retention_days"], "blur_people": p["blur_people"]}}

    cam_lock = threading.Lock()

    @api.get("/api/cameras")
    def cameras():
        removable = local_camera_ids(cfg["_cameras_file"]) if cfg.get("_cameras_file") else set()
        out = []
        for cam in list(cfg["cameras"]):
            w = core.workers.get(cam["id"])
            snap = w.snapshot() if w else {"id": cam["id"], "name": cam["name"], "status": "disabled"}
            snap["removable"] = cam["id"] in removable
            out.append(snap)
        return out

    @api.post("/api/cameras/add")
    def add_camera(payload: dict = Body(...)):
        """Add a network camera: finds its video address with the given login, saves it and starts it."""
        try:
            ip = ipaddress.ip_address(str(payload.get("ip", "")).strip())
            port = int(payload.get("port") or 554)
            channel = int(payload.get("channel") or 1)
        except ValueError:
            raise HTTPException(400, "عنوان الكاميرا (IP) أو المنفذ غير صحيح.")
        if not (ip.is_private or ip.is_loopback) or not (1 <= port <= 65535) or not (1 <= channel <= 64):
            raise HTTPException(400, "يمكن إضافة كاميرات الشبكة المحلية فقط.")
        user = str(payload.get("username") or "").strip()
        pw = str(payload.get("password") or "")
        cam_id = "cam_" + str(ip).replace(".", "_").replace(":", "_") + (f"_ch{channel}" if channel > 1 else "")
        with cam_lock:
            if any(c["id"] == cam_id for c in cfg["cameras"]):
                raise HTTPException(409, "هذه الكاميرا مضافة من قبل.")
            found = discovery.find_stream(str(ip), port, user, pw, channel)
            if found["status"] == "auth":
                raise HTTPException(401, "اسم المستخدم أو كلمة المرور غير صحيحة.")
            if found["status"] == "unreachable":
                raise HTTPException(502, "الكاميرا لا تستجيب. تأكد أنها تعمل وعلى نفس الشبكة.")
            if found["status"] != "ok":
                raise HTTPException(404, "تم الدخول للكاميرا لكن لم يُعثر على عنوان البث. جرّب رقم قناة آخر.")
            name = (str(payload.get("name") or "").strip() or f"كاميرا {ip}")[:60]
            cam = {"id": cam_id, "name": name, "source": found["url"],
                   "lines": [{"id": "door1", "name": "خط العد", "points": [[0.05, 0.55], [0.95, 0.55]]}],
                   "zones": []}
            add_local_camera(cfg["_cameras_file"], copy.deepcopy(cam))
            normalize_camera(cam)
            cfg["cameras"].append(cam)
            core.start_worker(cam)
        log.info("camera %s added (%s)", cam_id, found["vendor"])
        return {"id": cam_id, "name": name, "vendor": found["vendor"]}

    @api.delete("/api/cameras/{cam_id}")
    def delete_camera(cam_id: str):
        with cam_lock:
            if not remove_local_camera(cfg["_cameras_file"], cam_id):
                raise HTTPException(404, "هذه الكاميرا غير مضافة من اللوحة (احذفها من config.yaml).")
            w = core.workers.pop(cam_id, None)
            if w:
                w.stop()
            cfg["cameras"][:] = [c for c in cfg["cameras"] if c["id"] != cam_id]
        core.maps.forget_camera(cam_id)
        return {"ok": True}

    @api.get("/frame/{cam_id}")
    def frame(cam_id: str):
        """Latest analysed picture of a camera (the dashboard refreshes it a few times per second)."""
        w = core.workers.get(cam_id)
        jpg = w.latest_jpeg() if w else None
        if jpg is None:
            raise HTTPException(404, "no picture yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @api.get("/stream/{cam_id}")
    def stream(cam_id: str):
        w = core.workers.get(cam_id)
        if not w:
            raise HTTPException(404, "unknown camera")

        def gen():
            last = None
            while True:
                jpg = w.latest_jpeg()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.05)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @api.get("/api/stats/summary")
    def summary(day: str | None = None):
        return db.summary(day_or_today(day))

    @api.get("/api/stats/hourly")
    def hourly(day: str | None = None):
        return db.hourly(day_or_today(day))

    @api.get("/api/stats/trend")
    def trend(days: int = 7):
        return db.daily_trend(max(1, min(days, 90)))

    @api.get("/api/stats/zones")
    def zones(day: str | None = None):
        return db.zone_dwell(day_or_today(day))

    @api.get("/api/alerts")
    def alerts(day: str | None = None, kind: str | None = None, limit: int = 50):
        return db.alerts(day, max(1, min(limit, 500)), kind or None)

    @api.get("/api/alerts/feed")
    def alerts_feed(after: int = 0):
        """New alerts since the given id (the dashboard pops them up)."""
        return {"alerts": db.alerts_after(after), "unseen": db.unseen_count()}

    @api.get("/api/alerts/{alert_id}/image")
    def alert_image(alert_id: int):
        p = db.alert_image(alert_id)
        if p is None:
            raise HTTPException(404, "لا توجد صورة لهذا التنبيه.")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @api.post("/api/alerts/seen")
    def alerts_seen(payload: dict = Body(default={})):
        ids = payload.get("ids")
        db.mark_seen([int(i) for i in ids] if isinstance(ids, list) else None)
        return {"unseen": db.unseen_count()}

    # ---------- face recognition: enrolled people ----------
    def _picture(payload: dict):
        if payload.get("image"):
            try:
                img = decode_image(payload["image"])
            except (ValueError, TypeError):
                img = None
            if img is None:
                raise HTTPException(400, "الصورة غير صالحة. اختر صورة JPG أو PNG.")
            return img
        w = core.workers.get(str(payload.get("camera_id") or ""))
        img = w.latest_raw() if w else None
        if img is None:
            raise HTTPException(400, "ما وصلت صورة من هذه الكاميرا بعد.")
        return img

    def _need_faces():
        if not cfg["faces"]["enabled"]:
            raise HTTPException(400, "فعّل التعرف على الوجه من الإعدادات أولاً.")
        if not core.faces.ready:
            raise HTTPException(503, "جاري تجهيز التعرف على الوجه (تنزيل النموذج أول مرة). انتظر دقيقة وحاول."
                                if core.faces.status == "downloading" else f"التعرف على الوجه غير جاهز: {core.faces.status}")

    @api.get("/api/people")
    def people():
        seen = db.last_seen()
        return {"status": core.faces.status, "enabled": cfg["faces"]["enabled"],
                "people": [dict(p, last_seen=seen.get(p["id"])) for p in core.faces.public_people()]}

    @api.post("/api/people")
    def add_person(payload: dict = Body(...)):
        _need_faces()
        try:
            return core.faces.enroll(str(payload.get("name") or ""), _picture(payload),
                                     alert_on_seen=bool(payload.get("alert_on_seen")), note=str(payload.get("note") or ""),
                                     erp_id=str(payload.get("erp_id") or ""), job=str(payload.get("job") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.post("/api/people/{pid}/photos")
    def add_photo(pid: str, payload: dict = Body(...)):
        _need_faces()
        try:
            return core.faces.enroll("", _picture(payload), person_id=pid)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.patch("/api/people/{pid}")
    def edit_person(pid: str, payload: dict = Body(...)):
        try:
            core.faces.update_person(pid, payload)
        except ValueError as e:
            raise HTTPException(404, str(e))
        return {"ok": True}

    @api.delete("/api/people/{pid}")
    def delete_person(pid: str):
        if not core.faces.remove(pid):
            raise HTTPException(404, "الشخص غير موجود.")
        db.forget_person(pid)
        return {"ok": True}

    @api.get("/api/people/{pid}/photo")
    def person_photo(pid: str):
        p = core.faces.thumb(pid)
        if p is None:
            raise HTTPException(404, "no photo")
        return FileResponse(p, media_type="image/jpeg")

    # ---------- floor plans: where each enrolled person was last seen ----------
    def _locations() -> list[dict]:
        seen, now = db.last_seen(), time.time()
        fresh = 60 * cfg["faces"].get("present_minutes", 15)
        names = {c["id"]: c["name"] for c in cfg["cameras"]}
        out = []
        for p in core.faces.public_people():
            s = seen.get(p["id"])
            row = {"person_id": p["id"], "name": p["name"], "erp_id": p["erp_id"], "job": p["job"],
                   "camera_id": None, "camera": None, "location": None, "time": None, "minutes_ago": None,
                   "present": False}
            if s:
                ago = max(0, int((now - s["ts"]) // 60))
                row.update(camera_id=s["camera_id"], camera=names.get(s["camera_id"], s["camera_id"]),
                           location=core.maps.where(s["camera_id"]), time=s["time"], minutes_ago=ago,
                           present=now - s["ts"] <= fresh)
            out.append(row)
        return sorted(out, key=lambda r: (not r["present"], r["minutes_ago"] is None, r["minutes_ago"] or 0, r["name"]))

    core.analyst.locations = _locations

    @api.get("/api/locations")
    def locations(erp_id: str | None = None):
        """Also meant for the ERP / n8n: GET /api/locations?erp_id=E-102"""
        rows = _locations()
        if erp_id:
            rows = [r for r in rows if r["erp_id"] == erp_id.strip()]
        return {"present_minutes": cfg["faces"].get("present_minutes", 15), "people": rows}

    @api.get("/api/maps")
    def maps():
        return core.maps.public()

    @api.post("/api/maps")
    def add_map(payload: dict = Body(...)):
        try:
            img = decode_image(payload.get("image") or "")
        except Exception:  # noqa: BLE001  (bad base64 or not a picture)
            img = None
        try:
            return core.maps.add(str(payload.get("name") or ""), img)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.patch("/api/maps/{map_id}")
    def edit_map(map_id: str, payload: dict = Body(...)):
        try:
            return core.maps.rename(map_id, str(payload.get("name") or ""))
        except ValueError as e:
            raise HTTPException(404, str(e))

    @api.delete("/api/maps/{map_id}")
    def delete_map(map_id: str):
        if not core.maps.remove(map_id):
            raise HTTPException(404, "المخطط غير موجود.")
        return {"ok": True}

    @api.get("/api/maps/{map_id}/image")
    def map_image(map_id: str):
        p = core.maps.image(map_id)
        if p is None:
            raise HTTPException(404, "no picture")
        return FileResponse(p, media_type="image/jpeg")

    @api.put("/api/maps/cameras/{cam_id}")
    def place_camera(cam_id: str, payload: dict = Body(...)):
        if not any(c["id"] == cam_id for c in cfg["cameras"]):
            raise HTTPException(404, "الكاميرا غير موجودة.")
        try:
            core.maps.place(cam_id, payload.get("map") or None, payload.get("x", 0.5), payload.get("y", 0.5))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @api.get("/api/attendance")
    def attendance(day: str | None = None):
        return db.attendance(day_or_today(day))

    @api.post("/api/notify/test")
    def notify_test():
        if not core.notifier.enabled:
            raise HTTPException(400, "أكمل إعدادات الإرسال (الطريقة والرمز والرقم) واحفظها أولاً.")
        jpg = next((w.latest_jpeg() for w in core.workers.values() if w.latest_jpeg()), None)
        try:
            core.notifier.send("✅ رسالة تجربة من برنامج الكاميرات. التنبيهات راح توصلك هنا.", jpg)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"ما وصلت الرسالة: {e}")
        return {"ok": True, "with_image": jpg is not None}

    @api.get("/api/cameras/{cam_id}/snapshot.jpg")
    def camera_snapshot(cam_id: str):
        w = core.workers.get(cam_id)
        jpg = w.latest_jpeg() if w else None
        if jpg is None:
            raise HTTPException(404, "no picture yet")
        name = time.strftime("%Y-%m-%d_%H-%M-%S")
        return Response(jpg, media_type="image/jpeg",
                        headers={"Content-Disposition": f'attachment; filename="{cam_id}_{name}.jpg"'})

    @api.get("/api/export.csv")
    def export(day: str | None = None):
        rows = db.crossings_csv_rows(day_or_today(day))
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=["time", "camera_id", "line_id", "direction"])
        wr.writeheader()
        wr.writerows(rows)
        return PlainTextResponse("﻿" + buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename=crossings-{day_or_today(day)}.csv"})

    @api.post("/api/cameras/{cam_id}/line")
    def set_line(cam_id: str, payload: dict = Body(...)):
        """Move the counting line of a camera (two points as fractions of the picture)."""
        try:
            pts = [[min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)] for x, y in payload["points"]]
            if len(pts) != 2 or pts[0] == pts[1]:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "حدّد نقطتين مختلفتين على الصورة.")
        with cam_lock:
            cam = next((c for c in cfg["cameras"] if c["id"] == cam_id), None)
            if cam is None:
                raise HTTPException(404, "unknown camera")
            old = cam["lines"][0] if cam["lines"] else {}
            line = {"id": old.get("id", "door1"), "name": str(payload.get("name") or old.get("name") or "خط العد")[:60],
                    "points": pts, "invert": bool(payload.get("invert", False))}
            lines = [line] + cam["lines"][1:]
            set_camera_lines(cfg["_cameras_file"], cam_id, copy.deepcopy(lines))   # no-op for config.yaml cameras
            w = core.workers.get(cam_id)
            if w:
                w.lines = [LineCounter(l["id"], tuple(l["points"][0]), tuple(l["points"][1]), l.get("invert", False))
                           for l in lines]
            cam["lines"] = lines
        return {"ok": True, "line": line}

    @api.get("/api/cameras/{cam_id}/setup")
    def camera_setup(cam_id: str):
        cam = next((c for c in cfg["cameras"] if c["id"] == cam_id), None)
        if cam is None:
            raise HTTPException(404, "unknown camera")
        return {"id": cam_id, "name": cam["name"], "lines": cam["lines"]}

    @api.post("/api/window/closed")
    def window_closed():
        core.closed_at = time.time()
        return {"ok": True}

    @api.get("/api/window/state")
    def window_state():
        """For the desktop launcher: is a dashboard still open? (It stops the program when none is.)"""
        now = time.time()
        closed = core.closed_at >= core.last_seen   # a page said goodbye and nothing was requested since
        return {"idle": now - core.last_seen, "closed": (now - core.closed_at) if closed else None}

    @api.get("/api/settings")
    def get_settings():
        return public_settings(cfg)

    @api.post("/api/settings")
    def post_settings(payload: dict = Body(...)):
        try:
            apply_settings(cfg, payload)
        except ValueError as e:
            raise HTTPException(400, f"قيمة غير صحيحة: {e}")
        if cfg["faces"]["enabled"]:
            core.faces.ensure_ready()       # downloads the face models the first time
        core.analyst._openai_model_cache = None
        return public_settings(cfg)

    @api.post("/api/discover")
    def discover(payload: dict = Body(default={})):
        try:
            return discovery.discover(payload.get("subnet") or None)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @api.post("/api/agent/ask")
    def ask(payload: dict = Body(...)):
        q = (payload.get("question") or "").strip()
        if not q:
            raise HTTPException(400, "question is required")
        return {"answer": core.analyst.ask(q[:2000])}

    @api.post("/api/reports/daily")
    def make_report(payload: dict = Body(default={})):
        return {"report": core.analyst.daily_report(payload.get("day") or None)}

    @api.get("/api/update")
    def update_status():
        return core.updater.state

    @api.post("/api/update/check")
    def update_check():
        return core.updater.check()

    @api.post("/api/update/install")
    def update_install():
        if not core.updater.state["available"]:
            raise HTTPException(400, "no update available")
        threading.Thread(target=core.updater.install, daemon=True).start()
        return {"ok": True}

    @api.get("/api/reports")
    def reports():
        return db.reports()

    return api
