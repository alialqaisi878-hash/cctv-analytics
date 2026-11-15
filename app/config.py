"""Load and validate config.yaml, filling in safe defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "site_name": "My workplace",
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
    },
    "server": {"host": "0.0.0.0", "port": 8000, "password": ""},
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
    seen = set()
    for cam in cfg["cameras"]:
        if "id" not in cam or "source" not in cam:
            raise ValueError("every camera needs an 'id' and a 'source'")
        if cam["id"] in seen:
            raise ValueError(f"duplicate camera id: {cam['id']}")
        seen.add(cam["id"])
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
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
