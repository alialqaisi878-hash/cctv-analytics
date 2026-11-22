"""Entry point.

    python run.py                        # start dashboard + cameras from config.yaml
    python run.py --discover             # list cameras found on the network and exit
    python run.py --demo-data            # fill the database with a sample week, to try the dashboard/agent
    python run.py --report               # print today's report and exit
    python run.py --app-window           # desktop mode: own window, no console (used by the Windows shortcut)
    python run.py --service              # the program in the background, day and night (started with Windows)
    python run.py --stop [--forget]      # stop it (and no longer start with Windows)
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


WINDOW_PROFILE = ROOT / "data" / "window"


def _open_app_window(url: str) -> bool:
    """Open the dashboard in its own window (Edge/Chrome app mode), or in the default browser.
    Returns True when it is our own app window (with its own browser profile)."""
    exe = _find_app_browser()
    if not exe:
        webbrowser.open(url)
        return False
    profile = WINDOW_PROFILE   # own profile: no tabs, extensions or sign-in prompts of the user's browser
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
                      "--no-default-browser-check", "--window-size=1400,900"])
    return True


def _message_box(text: str) -> None:
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, "CCTV Analytics", 0x10)
    else:
        print(text)


def _windowless_python() -> Path:
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return w if os.name == "nt" and w.exists() else exe


def _alive(port: int) -> bool:
    """Does the server answer? (A password page or an error still means it is alive.)"""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/info", timeout=10):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _single_instance(path: Path):
    """A lock held for as long as this service runs: a second copy (auto-start + shortcut) gives way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        return None


class Service:
    """The program running day and night: the server runs in a child process that is started again when it
    stops unexpectedly, stops answering, or installed an online update. Closing the window does not stop it;
    the icon next to the clock, the settings page or the uninstaller does."""

    HEALTH_EVERY = 20          # seconds between checks that the server still answers
    HEALTH_GRACE = 300         # a fresh server may be busy preparing for this long
    HEALTH_FAILS = 9           # this many missed checks in a row (3 minutes): start it again

    def __init__(self, config: str, port: int, url: str):
        self.config, self.port, self.url = config, port, url
        self.log_path = ROOT / "data" / "app.log"
        self.stopping = threading.Event()
        self.server: subprocess.Popen | None = None
        self.restarts = 0

    def note(self, msg: str) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " service " + msg + "\n")
        except OSError:
            pass

    def start_server(self) -> None:
        if self.log_path.exists() and self.log_path.stat().st_size > 5_000_000:
            self.log_path.replace(self.log_path.with_suffix(".old.log"))
        exe = Path(sys.executable)
        if exe.name.lower() == "pythonw.exe":   # the child needs real output streams (they go to the log file)
            exe = exe.with_name("python.exe")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        log = open(self.log_path, "ab")
        self.server = subprocess.Popen([str(exe), str(ROOT / "run.py"), "--config", self.config], cwd=str(ROOT),
                                       env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       creationflags=0x08000000 if os.name == "nt" else 0)   # CREATE_NO_WINDOW
        self.started_at = time.time()

    def kill_server(self) -> None:
        if self.server and self.server.poll() is None:
            self.server.kill()
            try:
                self.server.wait(10)
            except subprocess.TimeoutExpired:
                pass

    def _keep_awake(self) -> None:
        from app.autostart import keep_awake
        try:
            keep_awake(bool(load_config(ROOT / self.config)["startup"]["keep_awake"]))
        except Exception:  # noqa: BLE001
            pass

    def loop(self) -> int:
        from app.autostart import STOP_EXIT_CODE
        from app.updater import RESTART_EXIT_CODE
        self.start_server()
        fails, last_check, last_awake, backoff = 0, time.time(), 0.0, 5
        try:
            while not self.stopping.wait(1):
                now = time.time()
                if now - last_awake > 60:
                    self._keep_awake()          # asked again every minute, from this same thread
                    last_awake = now
                rc = self.server.poll()
                if rc == STOP_EXIT_CODE:
                    self.note("stopped on request")
                    return 0
                if rc == RESTART_EXIT_CODE:
                    self.note("restarting after an online update")
                    self.start_server()
                    fails = 0
                    continue
                if rc is not None:
                    self.restarts += 1
                    self.note(f"the server stopped unexpectedly (code {rc}): starting it again in {backoff}s")
                    if self.stopping.wait(backoff):
                        break
                    backoff = min(backoff * 2, 60)
                    self.start_server()
                    fails = 0
                    continue
                if now - self.started_at > 600:
                    backoff = 5                 # it ran fine for a while: a later crash restarts quickly again
                if now - last_check >= self.HEALTH_EVERY and now - self.started_at > self.HEALTH_GRACE:
                    last_check = now
                    fails = 0 if _alive(self.port) else fails + 1
                    if fails >= self.HEALTH_FAILS:
                        self.restarts += 1
                        self.note("the server stopped answering for 3 minutes: starting it again")
                        self.kill_server()
                        self.start_server()
                        fails = 0
            return 0
        finally:
            self.kill_server()
            self.stopping.set()

    def tray(self) -> bool:
        """The icon next to the clock (Windows): open the window, or stop the program. False if not possible."""
        if os.name != "nt":
            return False
        try:
            import pystray
            from PIL import Image, ImageDraw
        except Exception:  # noqa: BLE001
            return False
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 60, 60), fill=(29, 78, 216, 255))
        d.ellipse((22, 22, 42, 42), fill=(255, 255, 255, 255))
        d.ellipse((28, 28, 36, 36), fill=(29, 78, 216, 255))

        def show(icon=None, item=None):
            _open_app_window(self.url)

        def stop(icon, item=None):
            self.note("stopped from the icon next to the clock")
            self.stopping.set()
            icon.stop()

        icon = pystray.Icon("CCTV Analytics", img, "CCTV Analytics: شغال", menu=pystray.Menu(
            pystray.MenuItem("افتح البرنامج", show, default=True),
            pystray.MenuItem("أوقف البرنامج (الكاميرات ما تتحلل)", stop)))

        def watch():
            self.stopping.wait()
            icon.stop()
        threading.Thread(target=watch, daemon=True).start()
        try:
            icon.run()
            return True
        except Exception:  # noqa: BLE001
            self.note("the icon next to the clock could not be shown")
            return False


def run_service(config: str, port: int, url: str) -> int:
    """python run.py --service: the program in the background (started with Windows, or by the shortcut)."""
    lock = _single_instance(ROOT / "data" / "service.lock")
    if lock is None or _port_in_use(port):
        return 0                            # already running
    svc = Service(config, port, url)
    svc.note("started")
    result: list[int] = []
    worker = threading.Thread(target=lambda: result.append(svc.loop()), name="service")
    worker.start()
    if not svc.tray():                      # no icon: just run
        worker.join()
    else:
        svc.stopping.set()
        worker.join()
    return result[0] if result else 0


def run_windowed(config: str, port: int, url: str) -> int:
    """The desktop shortcut: start the background program if it is not running yet, then open its window.
    Closing the window only closes the window: the cameras keep being watched."""
    if not _port_in_use(port):
        flags = 0x00000008 | 0x00000200 | 0x08000000 if os.name == "nt" else 0   # detached, own group, no window
        subprocess.Popen([str(_windowless_python()), str(ROOT / "run.py"), "--config", config, "--service"],
                         cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True,
                         start_new_session=os.name != "nt")
        deadline = time.time() + 180
        while not _port_in_use(port):
            if time.time() > deadline:
                _message_box(f"CCTV Analytics could not start.\nDetails: {ROOT / 'data' / 'app.log'}")
                return 1
            time.sleep(0.5)
    _open_app_window(url)
    return 0


def stop_program(port: int, forget: bool) -> int:
    """python run.py --stop [--forget]: stop the background program (and no longer start with Windows)."""
    import urllib.request
    if forget:
        from app.autostart import set_start_with_windows
        set_start_with_windows(ROOT, False)
    if not _port_in_use(port):
        return 0
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/window/quit", method="POST"),
                               timeout=10).read()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(60):
        if not _port_in_use(port):
            return 0
        time.sleep(0.5)
    return 1


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
    ap.add_argument("--service", action="store_true", help="run in the background day and night (started with Windows)")
    ap.add_argument("--stop", action="store_true", help="stop the program running in the background")
    ap.add_argument("--forget", action="store_true", help="with --stop: no longer start with Windows")
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
    if args.stop:
        sys.exit(stop_program(port, args.forget))
    if args.service:
        sys.exit(run_service(args.config, port, url))
    if args.app_window:
        sys.exit(run_windowed(args.config, port, url))
    if _port_in_use(port):
        # already running (e.g. started from the desktop shortcut twice): just show it
        print(f"CCTV Analytics is already running: {url}")
        if args.open_browser:
            webbrowser.open(url)
        return

    import uvicorn
    from app import autostart
    from app.server import create_app
    autostart.apply(cfg, ROOT)               # start with Windows (or not), as set in the settings
    api = create_app(cfg, start_cameras=not args.no_cameras)
    print(f"\n  Dashboard: {url}\n  (close this window to stop the program)\n")
    if args.open_browser:
        threading.Timer(2.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(api, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
