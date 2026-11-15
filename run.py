"""Entry point.

    python run.py                        # start dashboard + cameras from config.yaml
    python run.py --discover             # list cameras found on the network and exit
    python run.py --demo-data            # fill the database with a sample week, to try the dashboard/agent
    python run.py --report               # print today's report and exit
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import socket
import threading
import time
import webbrowser
from pathlib import Path

from app.config import load_config

ROOT = Path(__file__).parent


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def seed_demo(cfg: dict, days: int = 7) -> None:
    """Generate realistic-looking aggregate data (no video involved)."""
    from app.db import Database
    db = Database(cfg["database"], cfg["timezone"])
    rnd = random.Random(42)
    now = time.time()
    lines = [(c["id"], l["id"]) for c in cfg["cameras"] for l in c["lines"]] or [("entrance", "door1")]
    zones = [(c["id"], z["id"]) for c in cfg["cameras"] for z in c["zones"]] or [("entrance", "reception")]
    # visitors per hour profile for a typical office/shop day (08:00-17:00)
    profile = {8: 14, 9: 22, 10: 18, 11: 20, 12: 26, 13: 30, 14: 18, 15: 15, 16: 20, 17: 12}
    tid = 1
    for d in range(days, -1, -1):
        base_day = db.local(now - d * 86400).replace(minute=0, second=0, microsecond=0)
        if base_day.weekday() == 4:  # Friday: closed
            continue
        for hour, n in profile.items():
            start = base_day.replace(hour=hour).timestamp()
            if start > now:
                continue
            n = max(int(n * rnd.uniform(0.7, 1.3)), 0)
            for _ in range(n):
                cam, line = rnd.choice(lines)
                t_in = start + rnd.uniform(0, 3600)
                if t_in > now:
                    continue
                db.add_crossing(t_in, cam, line, "in", tid)
                stay = rnd.uniform(5, 50) * 60
                if t_in + stay < now:
                    db.add_crossing(t_in + stay, cam, line, "out", tid)
                zcam, zone = rnd.choice(zones)
                dwell = rnd.uniform(30, 900)
                db.add_visit(zcam, zone, tid, t_in + 20, t_in + 20 + dwell)
                tid += 1
            for m in range(0, 60, 5):
                t = start + m * 60
                if t < now:
                    db.add_occupancy(t, lines[0][0], max(int(n / 3 * rnd.uniform(0.5, 1.5)), 0))
        late = base_day.replace(hour=21, minute=40).timestamp()
        if d % 3 == 1 and late < now:
            db.add_alert(late, lines[0][0], "after_hours", "تم رصد 1 شخص خارج أوقات الدوام في الباب الرئيسي")
    print(f"demo data written to {cfg['database']}")


def main():
    ap = argparse.ArgumentParser(description="CCTV people analytics")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--discover", action="store_true", help="scan the network for cameras and exit")
    ap.add_argument("--subnet", help="subnet to scan, e.g. 192.168.1.0/24")
    ap.add_argument("--demo-data", action="store_true", help="write a sample week of data and exit")
    ap.add_argument("--report", action="store_true", help="print today's report and exit")
    ap.add_argument("--no-cameras", action="store_true", help="start dashboard without camera workers")
    ap.add_argument("--open-browser", action="store_true", help="open the dashboard in the web browser")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        shutil.copy(ROOT / "config.example.yaml", cfg_path)
        print(f"created {cfg_path.name} from config.example.yaml - edit it to add your cameras")
    cfg = load_config(cfg_path)
    cfg["database"] = str(ROOT / cfg["database"])
    for cam in cfg["cameras"]:  # resolve relative video paths against the project folder
        src = str(cam["source"])
        if not src.isdigit() and "://" not in src and (ROOT / src).exists():
            cam["source"] = str(ROOT / src)
    if cfg["detector"]["model"] and (ROOT / "models" / cfg["detector"]["model"]).exists():
        cfg["detector"]["model"] = str(ROOT / "models" / cfg["detector"]["model"])

    if args.discover:
        from app.discovery import discover
        print(json.dumps(discover(args.subnet), ensure_ascii=False, indent=2))
        return
    if args.demo_data:
        seed_demo(cfg)
        return
    if args.report:
        from app.agent import Analyst
        from app.db import Database
        print(Analyst(Database(cfg["database"], cfg["timezone"]), cfg).daily_report())
        return

    host, port = cfg["server"]["host"], cfg["server"]["port"]
    url = f"http://localhost:{port}"
    if _port_in_use(port):
        # already running (e.g. started from the desktop shortcut twice): just show it
        print(f"CCTV Analytics is already running: {url}")
        if args.open_browser:
            webbrowser.open(url)
        return

    import uvicorn
    from app.server import create_app
    api = create_app(cfg, start_cameras=not args.no_cameras)
    print(f"\n  Dashboard: {url}\n  (close this window to stop the program)\n")
    if args.open_browser:
        threading.Timer(2.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(api, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
