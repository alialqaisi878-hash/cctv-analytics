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
                     public_settings, remove_local_camera, set_camera_lines)
from .db import Database
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
        self.workers: dict[str, CameraWorker] = {}
        if start_cameras:
            for cam in cfg["cameras"]:
                if cam["enabled"]:
                    w = CameraWorker(cam, cfg, self.db)
                    self.workers[cam["id"]] = w
                    w.start()
        threading.Thread(target=self._housekeeping, daemon=True, name="housekeeping").start()
        self.last_seen = time.time()      # last request from a dashboard (see /api/window/state)
        self.closed_at = 0.0              # when a dashboard page last said it was being closed
        self.updater = Updater(cfg)
        self.updater.run_background(lambda: in_working_hours(cfg, self.db.local(time.time())))

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
                "version": __version__,
                "privacy": {"show_notice": p["show_notice"], "notice_text": p["notice_text"],
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
            w = CameraWorker(cam, cfg, db)
            core.workers[cam_id] = w
            w.start()
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
    def alerts(day: str | None = None):
        return db.alerts(day)

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
