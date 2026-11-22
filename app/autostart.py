"""Running all day and night on Windows: start with Windows, keep the PC awake, stop on request.

The program is a background service now: closing its window only closes the window. It starts by itself when
the user signs in to Windows (a per-user "Run" entry, no administrator rights needed) and is stopped from the
icon next to the clock, from the settings page or by the uninstaller.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("cctv.autostart")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "CCTV Analytics"
STOP_EXIT_CODE = 4      # the server exits with this when asked to stop for good: the service then stops too


def service_command(root: Path) -> str | None:
    """How Windows starts the service at sign-in: the app's own windowless Python running run.py --service."""
    pyw = root / ".venv" / "Scripts" / "pythonw.exe"
    if not pyw.exists():
        return None                     # not an installed copy (a developer's checkout): leave Windows alone
    return f'"{pyw}" "{root / "run.py"}" --service'


def set_start_with_windows(root: Path, on: bool) -> bool:
    """Add or remove the sign-in entry. True when Windows now does what was asked."""
    if os.name != "nt":
        return False
    import winreg
    cmd = service_command(root)
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            if on and cmd:
                try:
                    if winreg.QueryValueEx(k, RUN_NAME)[0] == cmd:
                        return True
                except FileNotFoundError:
                    pass
                winreg.SetValueEx(k, RUN_NAME, 0, winreg.REG_SZ, cmd)
                log.info("starts with Windows: %s", cmd)
                # older versions could start with Windows through a shortcut that opened a console window
                old = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup/CCTV Analytics.lnk"
                if os.environ.get("APPDATA") and old.exists():
                    old.unlink()
            elif not on:
                try:
                    winreg.DeleteValue(k, RUN_NAME)
                    log.info("no longer starts with Windows")
                except FileNotFoundError:
                    pass
        return True
    except OSError:
        log.exception("could not change starting with Windows")
        return False


def starts_with_windows() -> bool | None:
    if os.name != "nt":
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, RUN_NAME)
            return True
    except OSError:
        return False


def keep_awake(on: bool) -> None:
    """Ask Windows not to go to sleep while the program runs (the screen may still turn off)."""
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def apply(cfg: dict, root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parent.parent
    set_start_with_windows(root, bool(cfg["startup"]["with_windows"]))
