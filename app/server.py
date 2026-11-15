"""FastAPI web dashboard + JSON API."""
from __future__ import annotations

import base64
import csv
import io
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
from .config import in_working_hours
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

    password = cfg["server"].get("password") or ""
    if password:
        @api.middleware("http")
        async def basic_auth(request: Request, call_next):
            auth = request.headers.get("authorization", "")
            ok = False
            if auth.startswith("Basic "):
                try:
                    _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                    ok = secrets.compare_digest(pw, password)
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

    @api.get("/api/cameras")
    def cameras():
        out = []
        for cam in cfg["cameras"]:
            w = core.workers.get(cam["id"])
            out.append(w.snapshot() if w else {"id": cam["id"], "name": cam["name"], "status": "disabled"})
        return out

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
