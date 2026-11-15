"""Load and validate config.yaml, filling in safe defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "site_name": "My workplace",
    "demo": False,                  # True keeps the sample-video camera of old installs
    "timezone": "Asia/Baghdad",
    "database": "data/cctv.db",
    "detector": {
        "model": "yolo11n.pt",      # downloaded automatically on first run
        "confidence": 0.35,
        "process_fps": 8,           # frames analysed per second per camera
        "imgsz": 640,
        "device": None,             # None = auto (GPU if available), or "cpu", "0"
    },
    "privacy": {
        "show_notice": True,
        "notice_text": (
            "تنبيه: هذا المكان مراقب بالكاميرات لأغراض السلامة وتنظيم العمل. "
            "يتم حفظ إحصائيات مجمّعة فقط (أعداد الدخول والخروج والازدحام)، "
            "ولا يتم التعرف على الوجوه أو تحديد هوية الأشخاص."
        ),
        "blur_people": True,        # blur heads in the live view
        "retention_days": 30,       # delete aggregate events older than this
        "store_snapshots": False,   # never save frames to disk unless enabled
    },
    "working_hours": {
        "start": "08:00",
        "end": "17:00",
        "days": [6, 0, 1, 2, 3],    # Python weekday(): Mon=0 ... Sun=6 (default Sun-Thu)
    },
    "alerts": {
        "after_hours_presence": True,
        "max_occupancy": 0,         # 0 = disabled; otherwise alert when exceeded
        "cooldown_minutes": 15,
    },
    "agent": {
        "model": "claude-opus-5-5",
        "language": "ar",
        "daily_report_time": "17:30",   # local time; "" disables the scheduled report
        "api_key": "",                  # Claude API key (or set the ANTHROPIC_API_KEY environment variable)
    },
    "server": {"host": "0.0.0.0", "port": 8000, "password": ""},
    "updates": {
        "enabled": True,
        "repo": "alialqaisi878-hash/cctv-analytics",
        "token": "",                # only needed if the GitHub repo is private
        "auto_install": True,       # installs automatically outside working hours
        "check_hours": 6,
    },
    "cameras": [],
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = _merge(DEFAULTS, raw or {})
    # settings changed in the program live in their own file, so config.yaml keeps its comments
    cfg["_settings_file"] = str(path.parent / "data" / "settings.yaml")
    cfg = _merge(cfg, read_settings(cfg["_settings_file"]))
    if not cfg["demo"]:   # the sample video of early versions is no longer shown
        cfg["cameras"] = [c for c in cfg["cameras"] if not (c.get("id") == "demo" and "samples/" in str(c.get("source")))]
    # cameras added from the dashboard live in their own file, so config.yaml keeps its comments
    cfg["_cameras_file"] = str(path.parent / "data" / "cameras.yaml")
    cfg["cameras"] = list(cfg["cameras"]) + _read_local_cameras(cfg["_cameras_file"])
    seen = set()
    for cam in cfg["cameras"]:
        if "id" not in cam or "source" not in cam:
            raise ValueError("every camera needs an 'id' and a 'source'")
        if cam["id"] in seen:
            raise ValueError(f"duplicate camera id: {cam['id']}")
        seen.add(cam["id"])
        normalize_camera(cam)
    return cfg


def normalize_camera(cam: dict) -> dict:
    cam.setdefault("name", cam["id"])
    cam.setdefault("enabled", True)
    cam.setdefault("loop", False)
    cam.setdefault("lines", [])
    cam.setdefault("zones", [])
    for ln in cam["lines"]:
        ln.setdefault("name", ln["id"])
        ln.setdefault("invert", False)
    for z in cam["zones"]:
        z.setdefault("name", z["id"])
    return cam


def _read_local_cameras(file: str | Path) -> list[dict]:
    file = Path(file)
    if not file.exists():
        return []
    return list((yaml.safe_load(file.read_text(encoding="utf-8")) or {}).get("cameras") or [])


def _write_local_cameras(file: str | Path, cams: list[dict]) -> None:
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(yaml.safe_dump({"cameras": cams}, allow_unicode=True, sort_keys=False), encoding="utf-8")


def local_camera_ids(file: str | Path) -> set[str]:
    return {c["id"] for c in _read_local_cameras(file)}


def add_local_camera(file: str | Path, cam: dict) -> None:
    cams = [c for c in _read_local_cameras(file) if c["id"] != cam["id"]]
    _write_local_cameras(file, cams + [cam])


def remove_local_camera(file: str | Path, cam_id: str) -> bool:
    cams = _read_local_cameras(file)
    kept = [c for c in cams if c["id"] != cam_id]
    if len(kept) == len(cams):
        return False
    _write_local_cameras(file, kept)
    return True


def save_config(cfg: dict, path: str | Path) -> None:
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def in_working_hours(cfg: dict, d) -> bool:
    """d: timezone-aware local datetime."""
    wh = cfg["working_hours"]
    if d.weekday() not in wh["days"]:
        return False
    sh, sm = (int(x) for x in wh["start"].split(":"))
    eh, em = (int(x) for x in wh["end"].split(":"))
    return (sh, sm) <= (d.hour, d.minute) < (eh, em)


# ---------- settings edited in the program ----------
# section -> key -> validator returning the cleaned value (raises ValueError when invalid)
def _time(v) -> str:
    v = str(v).strip()
    if v == "":
        return v
    h, m = v.split(":")
    if not (0 <= int(h) <= 23 and 0 <= int(m) <= 59):
        raise ValueError(v)
    return f"{int(h):02d}:{int(m):02d}"


def _required_time(v) -> str:
    v = _time(v)
    if not v:
        raise ValueError("time required")
    return v


def _int(lo: int, hi: int):
    def check(v):
        v = int(v)
        if not lo <= v <= hi:
            raise ValueError(v)
        return v
    return check


def _float(lo: float, hi: float):
    def check(v):
        v = float(v)
        if not lo <= v <= hi:
            raise ValueError(v)
        return v
    return check


def _days(v) -> list[int]:
    out = sorted({int(x) for x in v})
    if not out or any(not 0 <= x <= 6 for x in out):
        raise ValueError(v)
    return out


def _text(n: int):
    return lambda v: str(v).strip()[:n]


EDITABLE = {
    None: {"site_name": _text(80)},
    "working_hours": {"start": _required_time, "end": _required_time, "days": _days},
    "privacy": {"retention_days": _int(1, 365), "blur_people": bool, "show_notice": bool},
    "alerts": {"after_hours_presence": bool, "max_occupancy": _int(0, 1000), "cooldown_minutes": _int(1, 1440)},
    "agent": {"daily_report_time": _time, "api_key": _text(300)},
    "server": {"password": _text(100)},
    "detector": {"process_fps": _int(1, 30), "confidence": _float(0.05, 0.95)},
    "updates": {"auto_install": bool},
}
SECRET = {("agent", "api_key"), ("server", "password")}


def read_settings(file: str | Path) -> dict:
    file = Path(file)
    if not file.exists():
        return {}
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def clean_settings(changes: dict) -> dict:
    """Keep only editable keys, validated. Raises ValueError naming the bad field."""
    out: dict = {}
    for section, keys in EDITABLE.items():
        src = changes if section is None else changes.get(section)
        if not isinstance(src, dict):
            continue
        for key, check in keys.items():
            if key not in src:
                continue
            try:
                value = check(src[key])
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"{section + '.' if section else ''}{key}") from None
            if section is None:
                out[key] = value
            else:
                out.setdefault(section, {})[key] = value
    return out


def apply_settings(cfg: dict, changes: dict) -> None:
    """Validate, apply to the running configuration (in place) and save to data/settings.yaml."""
    clean = clean_settings(changes)
    for k, v in clean.items():
        if isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    file = Path(cfg["_settings_file"])
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(yaml.safe_dump(_merge(read_settings(file), clean), allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def public_settings(cfg: dict) -> dict:
    """Editable settings for the dashboard; secrets are reported only as set / not set."""
    out: dict = {}
    for section, keys in EDITABLE.items():
        for key in keys:
            if (section, key) in SECRET:
                out.setdefault(section, {})["has_" + key] = bool(cfg[section].get(key))
            elif section is None:
                out[key] = cfg[key]
            else:
                out.setdefault(section, {})[key] = cfg[section][key]
    return out


def set_camera_lines(file: str | Path, cam_id: str, lines: list[dict]) -> bool:
    cams = _read_local_cameras(file)
    for c in cams:
        if c["id"] == cam_id:
            c["lines"] = lines
            _write_local_cameras(file, cams)
            return True
    return False
