"""Entry point.

    python run.py                        # start dashboard + cameras from config.yaml
    python run.py --discover             # list cameras found on the network and exit
    python run.py --demo-data            # fill the database with a sample week, to try the dashboard/agent
    python run.py --report               # print today's report and exit
    python run.py --app-window           # desktop mode: own window, no console (used by the Windows shortcut)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
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


def _find_app_browser() -> str | None:
    """Edge or Chrome: both can show a web page as a plain application window (--app=)."""
    candidates = []
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates.append(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe")
            candidates.append(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")


def _open_app_window(url: str) -> subprocess.Popen | None:
    """Open the dashboard in its own window. Returns the window process (ends when the window is closed)."""
    exe = _find_app_browser()
    if not exe:
        webbrowser.open(url)
        return None
    profile = ROOT / "data" / "window"   # own profile: the process then lives exactly as long as the window
    profile.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
                             "--no-default-browser-check", "--window-size=1400,900"])


def _message_box(text: str) -> None:
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, "CCTV Analytics", 0x10)
    else:
        print(text)


def run_windowed(config: str, port: int, url: str) -> int:
    """Desktop mode: the server runs hidden in a child process, the dashboard shows in its own window.

    Closing the window stops the program. When the server exits with RESTART_EXIT_CODE (an online
    update was installed) it is started again and the window stays open.
    """
    from app.updater import RESTART_EXIT_CODE
    if _port_in_use(port):          # already running (auto-start, or opened twice): just show it
        _open_app_window(url)
        return 0
    (ROOT / "data").mkdir(exist_ok=True)
    log_path = ROOT / "data" / "app.log"
    if log_path.exists() and log_path.stat().st_size > 5_000_000:
        log_path.unlink()
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":   # the child needs real output streams (they go to the log file)
        exe = exe.with_name("python.exe")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW

    def start_server() -> subprocess.Popen:
        log = open(log_path, "ab")
        return subprocess.Popen([str(exe), str(ROOT / "run.py"), "--config", config], cwd=str(ROOT), env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags)

    server = start_server()
    deadline = time.time() + 180
    while not _port_in_use(port):
        if server.poll() is not None or time.time() > deadline:
            if server.poll() is None:
                server.kill()
            _message_box(f"CCTV Analytics could not start.\nDetails: {log_path}")
            return 1
        time.sleep(0.3)
    window = _open_app_window(url)
    try:
        while True:
            if window is not None and window.poll() is not None:
                return 0                      # window closed by the user
            rc = server.poll()
            if rc == RESTART_EXIT_CODE:
                server = start_server()
            elif rc is not None:
                _message_box(f"CCTV Analytics stopped unexpectedly.\nDetails: {log_path}")
                return 1
            time.sleep(0.5)
    finally:
        if server.poll() is None:
            server.kill()


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
    ap.add_argument("--app-window", action="store_true", help="desktop mode: show the dashboard in its own window")
    args = ap.parse_args()

    if sys.stdout is None or sys.stderr is None:   # started with pythonw.exe: no console
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")

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
    if args.app_window:
        sys.exit(run_windowed(args.config, port, url))
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
