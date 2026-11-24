"""Load and validate config.yaml, filling in safe defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROVIDERS = ("", "greenapi", "ultramsg", "telegram", "webhook")

DEFAULTS: dict[str, Any] = {
    "site_name": "My workplace",
    "demo": False,                  # True keeps the sample-video camera of old installs
    "timezone": "Asia/Baghdad",
    "database": "data/cctv.db",
    "detector": {
        "model": "yolo11n.pt",      # downloaded automatically on first run
        "confidence": 0.5,          # how sure the detector must be that it sees a person
        "process_fps": 8,           # frames analysed per second per camera
        "stream": "fast",           # fast = the camera's light second stream when it has one, high = full quality
        "imgsz": 640,
        "device": None,             # None = auto (GPU if available), or "cpu", "0"
    },
    "privacy": {
        "show_notice": True,
        "notice_text": "",          # "" = written automatically from the settings below (see notice_text())
        "blur_people": True,        # blur heads in the live view
        "retention_days": 30,       # delete events and alert pictures older than this
        "save_alert_pictures": True,  # keep one picture per alert (no video is ever saved)
    },
    "faces": {
        "enabled": False,           # recognise the people enrolled on the "الأشخاص" page
        "threshold": 0.40,          # similarity needed to call it a match (higher = stricter)
        "interval_s": 0.5,          # seconds between face checks per camera
        "present_minutes": 15,      # on the map, someone seen within this many minutes counts as "here now"
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
        "camera_offline": True,     # alert when a camera stops sending pictures
        "unknown_after_hours": False,  # (face recognition) alert for unknown faces outside working hours
    },
    "hazard": {                     # fire and smoke (see app/hazard.py): alerts at any hour
        "enabled": True,
        "fire": True,
        "smoke": True,
        "every_s": 5,               # each camera is looked at this often
        "fire_confidence": 0.45,
        "smoke_confidence": 0.5,
        "cooldown_minutes": 5,      # the same camera alerts again for the same thing after this long
        "imgsz": 640,
        "model": "",                # "" = the model that comes with the program
        "flame_cameras": [],        # up to 4 camera ids watched picture by picture for small flames (lighters)
        "flame_cooldown_minutes": 2,
    },
    "notify": {                     # send alerts with their picture to WhatsApp / Telegram (see app/notify.py)
        "provider": "",             # "" | greenapi | ultramsg | telegram | webhook (n8n...)
        "instance": "",
        "token": "",
        "to": "",                   # WhatsApp number (07xx... or 9647xx...) or Telegram chat id
        "api_url": "",              # only if the provider gave a special address
        "send_images": True,
        "kinds": [],                # [] = every alert; or e.g. ["after_hours", "person"]
        "send_locations": False,    # webhook only: also send an event when an enrolled person shows up on another camera
    },
    "agent": {
        "model": "claude-opus-5-5",
        "language": "ar",
        "daily_report_time": "17:30",   # local time; "" disables the scheduled report
        "api_key": "",                  # Claude (sk-ant-...) or OpenAI (sk-...) key; the kind is detected
        "provider": "auto",             # auto | claude | openai
        "openai_model": "",             # "" = pick a current GPT model automatically
    },
    "server": {"host": "0.0.0.0", "port": 8000, "password": ""},
    "startup": {
        "with_windows": True,           # the program starts by itself when Windows starts (runs day and night)
        "keep_awake": True,             # Windows does not go to sleep while the program runs
    },
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
    if str(cfg["agent"].get("openai_model") or "").startswith("sk-"):   # a key saved in the model box by mistake
        cfg["agent"]["openai_model"] = ""
    if cfg["detector"].get("confidence") == 0.35:   # the old default marked things that are not people
        cfg["detector"]["confidence"] = 0.5
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


def notice_text(cfg: dict) -> str:
    """The camera notice shown in the program: written from the settings unless config.yaml sets its own."""
    p = cfg["privacy"]
    if p.get("notice_text"):
        return p["notice_text"]
    parts = ["تنبيه: هذا المكان مراقب بالكاميرات لأغراض السلامة وتنظيم العمل."]
    if cfg["faces"]["enabled"]:
        parts.append("يتم التعرف على وجوه الأشخاص المسجّلين في النظام فقط.")
    else:
        parts.append("لا يتم التعرف على الوجوه.")
    if p.get("save_alert_pictures", True):
        parts.append(f"تُحفظ صورة واحدة لكل تنبيه لمدة {p['retention_days']} يوم، ولا يُحفظ أي فيديو.")
    else:
        parts.append("لا تُحفظ أي صور أو فيديو، فقط إحصائيات مجمّعة.")
    return " ".join(parts)


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


def _camera_ids(v) -> list[str]:
    out = []
    for x in v or []:
        x = str(x).strip()[:60]
        if x and x not in out:
            out.append(x)
    if len(out) > 4:
        raise ValueError("4 cameras at most")
    return out


def _text(n: int):
    return lambda v: str(v).strip()[:n]


def _choice(*options):
    def check(v):
        v = str(v or "").strip()
        if v not in options:
            raise ValueError(v)
        return v
    return check


ALERT_KINDS = ("after_hours", "crowding", "offline", "person", "unknown")


def _kinds(v) -> list[str]:
    out = [str(x) for x in (v or [])]
    if any(x not in ALERT_KINDS for x in out):
        raise ValueError(v)
    return out


EDITABLE = {
    None: {"site_name": _text(80)},
    "working_hours": {"start": _required_time, "end": _required_time, "days": _days},
    "privacy": {"retention_days": _int(1, 365), "blur_people": bool, "show_notice": bool,
                "save_alert_pictures": bool},
    "alerts": {"after_hours_presence": bool, "max_occupancy": _int(0, 1000), "cooldown_minutes": _int(1, 1440),
               "camera_offline": bool, "unknown_after_hours": bool},
    "agent": {"daily_report_time": _time, "api_key": _text(300), "provider": _choice("auto", "claude", "openai"),
              "openai_model": _text(60)},
    "hazard": {"enabled": bool, "fire": bool, "smoke": bool, "every_s": _int(2, 60),
               "fire_confidence": _float(0.2, 0.9), "smoke_confidence": _float(0.2, 0.9),
               "cooldown_minutes": _int(1, 120), "flame_cameras": _camera_ids, "flame_cooldown_minutes": _int(1, 60)},
    "faces": {"enabled": bool, "threshold": _float(0.25, 0.8), "present_minutes": _int(1, 720)},
    "notify": {"provider": _choice(*PROVIDERS), "instance": _text(60), "token": _text(200), "to": _text(60),
               "api_url": _text(200), "send_images": bool, "kinds": _kinds, "send_locations": bool},
    "server": {"password": _text(100)},
    "detector": {"process_fps": _int(1, 30), "confidence": _float(0.05, 0.95), "stream": _choice("fast", "high")},
    "updates": {"auto_install": bool},
    "startup": {"with_windows": bool, "keep_awake": bool},
}
SECRET = {("agent", "api_key"), ("server", "password"), ("notify", "token")}


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
    # an API key pasted into the model box: use it as the key, and let the model be chosen automatically
    agent = out.get("agent") or {}
    if str(agent.get("openai_model", "")).startswith("sk-"):
        raw = str(changes["agent"]["openai_model"]).strip()
        agent["openai_model"] = ""
        if not agent.get("api_key"):
            agent["api_key"] = EDITABLE["agent"]["api_key"](raw)
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
