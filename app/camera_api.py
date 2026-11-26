"""What a camera can do by itself: many Dahua cameras find people and count them on the camera.

This reads only (model, firmware, which smart rules it has and which are switched on), once per camera, with the
same user and password the camera's video already uses. If that login is refused it stops at once and does not try
again, so a camera never locks the account. The answers are shown in the program check, so the next version can take
people and counts from the cameras that do it themselves and leave the computer free.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from urllib.parse import unquote, urlsplit

log = logging.getLogger("cctv.camera_api")

TIMEOUT = 4.0
# rules a Dahua camera can run itself that tell about people
PEOPLE_RULES = ("NumberStat", "ManNumDetection", "CrossLineDetection", "CrossRegionDetection", "SmartMotionHuman",
                "FaceDetection", "StereoNumber", "QueueDetection", "CrowdDistriMap")


def login_of(source: str) -> tuple[str, str, str] | None:
    """(host, user, password) of a Dahua RTSP address, or None for anything else."""
    s = str(source or "")
    if not s.startswith("rtsp://") or "realmonitor" not in s:
        return None
    u = urlsplit(s)
    if not u.hostname:
        return None
    return u.hostname, unquote(u.username or ""), unquote(u.password or "")


def _kv(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        k, sep, v = line.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


def probe(host: str, user: str, password: str, get=None) -> dict:
    """Ask the camera what it is and what it can do. Never raises."""
    import httpx
    res = {"checked_at": time.time(), "status": "ok", "model": "", "firmware": "", "can": [], "on": [], "human_motion": None}
    own = get is None
    client = httpx.Client(auth=httpx.DigestAuth(user, password), timeout=TIMEOUT) if own else None
    get = get or (lambda path: client.get(f"http://{host}{path}"))
    try:
        r = get("/cgi-bin/magicBox.cgi?action=getDeviceType")
        if r.status_code == 401:
            return {**res, "status": "login refused (not tried again)"}
        if r.status_code != 200:
            return {**res, "status": f"not a Dahua camera or no web access (HTTP {r.status_code})"}
        res["model"] = _kv(r.text).get("type", r.text.strip()[:60])
        r = get("/cgi-bin/magicBox.cgi?action=getSoftwareVersion")
        if r.status_code == 200:
            res["firmware"] = _kv(r.text).get("version", "")[:60]
        r = get("/cgi-bin/devVideoAnalyse.cgi?action=getcaps&channel=1")
        if r.status_code == 200:
            found = set(re.findall(r"\b(" + "|".join(PEOPLE_RULES) + r")\b", r.text))
            res["can"] = sorted(found)
        r = get("/cgi-bin/configManager.cgi?action=getConfig&name=VideoAnalyseRule")
        if r.status_code == 200:
            kv = _kv(r.text)
            on = set()
            for k, v in kv.items():
                m = re.match(r"table\.VideoAnalyseRule\[0\]\[(\d+)\]\.Type$", k)
                if m and kv.get(f"table.VideoAnalyseRule[0][{m.group(1)}].Enable", "false") == "true":
                    on.add(v)
            res["on"] = sorted(on)
            res["can"] = sorted(set(res["can"]) | on)
        r = get("/cgi-bin/configManager.cgi?action=getConfig&name=SmartMotionDetect")
        if r.status_code == 200 and "SmartMotionDetect" in r.text:
            kv = _kv(r.text)
            res["human_motion"] = kv.get("table.SmartMotionDetect[0].Enable") == "true" and \
                kv.get("table.SmartMotionDetect[0].ObjectTypes.Human", "true") == "true"
    except Exception as e:  # noqa: BLE001
        res["status"] = f"no answer ({type(e).__name__})"
    finally:
        if own:
            client.close()
    return res


class CameraAbilities:
    """Checks each Dahua camera once, one after another, in the background."""

    def __init__(self):
        self.found: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._todo: list[tuple[str, str]] = []
        self._thread: threading.Thread | None = None

    def check(self, cam_id: str, source: str) -> None:
        if cam_id in self.found or login_of(source) is None:
            return
        with self._lock:
            if any(c == cam_id for c, _ in self._todo):
                return
            self._todo.append((cam_id, source))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="camera-abilities")
                self._thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._todo:
                    return
                cam_id, source = self._todo.pop(0)
            host, user, pw = login_of(source)
            self.found[cam_id] = probe(host, user, pw)
            log.info("camera %s abilities: %s", cam_id, {k: v for k, v in self.found[cam_id].items() if k != "checked_at"})
            time.sleep(0.5)

    def describe(self, cam_id: str) -> str:
        a = self.found.get(cam_id)
        if not a:
            return ""
        if a["status"] != "ok":
            return f"camera's own abilities: {a['status']}"
        hm = {None: "?", True: "on", False: "off"}[a["human_motion"]]
        return (f"camera's own abilities: model {a['model'] or '?'}, firmware {a['firmware'] or '?'}, "
                f"people rules it has {','.join(a['can']) or 'none'}, switched on {','.join(a['on']) or 'none'}, "
                f"human motion detection {hm}")
