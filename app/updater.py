"""Online updates from GitHub Releases.

Every release (made by .github/workflows/windows-installer.yml when a tag like v1.2.0 is pushed)
carries `cctv-analytics-update.zip` with the program files. The app checks the latest release,
downloads that zip, replaces its own code (keeping config.yaml, data/ and the Python environment),
installs any new libraries, and restarts itself. No uninstall/reinstall needed.

Works with a public repo out of the box; for a private repo put a read-only GitHub token in
config.yaml -> updates.token.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .version import __version__

log = logging.getLogger("cctv.updater")
ROOT = Path(__file__).resolve().parent.parent
ASSET_NAME = "cctv-analytics-update.zip"
RESTART_EXIT_CODE = 3  # launch.bat / start.sh restart the app when they see this
# never touched by an update
# (launchers are protected because Windows reads a running .bat file while it executes)
PROTECTED = {"config.yaml", "data", ".venv", ".git", "install.log", "models", "samples",
             "launch.bat", "start.bat", "start.sh", "python", "tools"}


def parse_version(v: str) -> tuple[int, ...]:
    v = v.strip().lstrip("vV")
    parts = []
    for p in v.split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts + [0] * (3 - len(parts)))


class Updater:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("updates", {})
        self.state = {"current": __version__, "latest": None, "available": False, "notes": "",
                      "checked_at": None, "error": None, "installing": False}
        self._asset_url: str | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True) and self.cfg.get("repo"))

    def _request(self, url: str, accept: str) -> urllib.request.Request:
        headers = {"Accept": accept, "User-Agent": "cctv-analytics-updater"}
        token = self.cfg.get("token") or os.environ.get("CCTV_UPDATE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return urllib.request.Request(url, headers=headers)

    def check(self) -> dict:
        if not self.enabled:
            self.state["error"] = "updates disabled"
            return self.state
        # CCTV_UPDATE_API points the updater at another server: used by the automated update test
        api = os.environ.get("CCTV_UPDATE_API") or "https://api.github.com"
        url = f"{api.rstrip('/')}/repos/{self.cfg['repo']}/releases/latest"
        try:
            with urllib.request.urlopen(self._request(url, "application/vnd.github+json"), timeout=15) as r:
                rel = json.load(r)
        except urllib.error.HTTPError as e:
            msg = {404: "no release found (or the repo is private and needs updates.token)",
                   401: "invalid updates.token", 403: "GitHub rate limit, will retry later"}.get(e.code, str(e))
            self.state.update(error=msg, checked_at=time.time())
            return self.state
        except Exception as e:  # noqa: BLE001 - offline etc.
            self.state.update(error=f"cannot reach GitHub: {e}", checked_at=time.time())
            return self.state
        asset = next((a for a in rel.get("assets", []) if a.get("name") == ASSET_NAME), None)
        latest = rel.get("tag_name", "")
        self._asset_url = asset["url"] if asset else None
        self.state.update(
            latest=latest, notes=(rel.get("body") or "")[:4000], checked_at=time.time(), error=None,
            available=bool(asset) and parse_version(latest) > parse_version(__version__))
        return self.state

    def install(self) -> None:
        """Download and apply the update, then exit so the launcher restarts the app."""
        with self._lock:
            if self.state["installing"]:
                return
            self.state["installing"] = True
        try:
            if not self._asset_url:
                self.check()
            if not self.state["available"] or not self._asset_url:
                raise RuntimeError("no update available")
            log.info("downloading update %s", self.state["latest"])
            with urllib.request.urlopen(self._request(self._asset_url, "application/octet-stream"), timeout=300) as r:
                data = r.read()
            apply_update_zip(data, ROOT)
            log.info("update applied, restarting")
        except Exception as e:  # noqa: BLE001
            log.exception("update failed")
            self.state.update(installing=False, error=f"update failed: {e}")
            return
        threading.Timer(1.5, lambda: os._exit(RESTART_EXIT_CODE)).start()

    def run_background(self, in_working_hours) -> None:
        """Periodic check; auto-install only outside working hours so counting isn't interrupted."""
        def loop():
            time.sleep(20)
            while True:
                self.check()
                if (self.state["available"] and self.cfg.get("auto_install", True)
                        and not in_working_hours()):
                    self.install()
                time.sleep(max(float(self.cfg.get("check_hours", 6)), 0.5) * 3600)
        if self.enabled:
            threading.Thread(target=loop, daemon=True, name="updater").start()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def apply_update_zip(data: bytes, root: Path) -> None:
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    if "run.py" not in names or not any(n.startswith("app/") for n in names):
        raise RuntimeError("update package looks wrong")
    for n in names:  # refuse path traversal
        if n.startswith("/") or ".." in Path(n).parts:
            raise RuntimeError(f"unsafe path in update: {n}")
    old_req = _sha(root / "requirements.txt")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zf.extractall(tmp)
        # replace the app package wholesale (removes deleted modules), then copy top-level files
        if (tmp / "app").is_dir():
            backup = root / "app.old"
            shutil.rmtree(backup, ignore_errors=True)
            if (root / "app").exists():
                (root / "app").rename(backup)
            shutil.copytree(tmp / "app", root / "app")
            shutil.rmtree(backup, ignore_errors=True)
        for item in tmp.iterdir():
            if item.name == "app" or item.name in PROTECTED:
                continue
            dest = root / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    if _sha(root / "requirements.txt") != old_req:
        log.info("requirements changed, installing libraries")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt"),
                        "--disable-pip-version-check", "-q"], check=True)
