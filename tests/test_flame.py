"""Small flames (a lighter) on chosen cameras: caught in a fraction of a second, not fooled by lamps or lights."""
import time

import cv2
import numpy as np

from tests.test_v11 import cfg  # noqa: F401  (fixture)

RNG = np.random.default_rng(3)


class Catch:
    id = "cam"

    def __init__(self):
        self.alerts = []

    def flame_alert(self, ts, frame, box):
        self.alerts.append((ts, box))


def _scene(i):
    """A plain room with a little camera noise and someone walking past."""
    f = np.full((432, 768, 3), (90, 100, 110), np.uint8)
    cv2.rectangle(f, (50 + 4 * i % 600, 150), (110 + 4 * i % 600, 400), (60, 70, 140), -1)
    noise = RNG.integers(-4, 5, f.shape)
    return np.clip(f.astype(int) + noise, 0, 255).astype(np.uint8)


def _lighter(f, x=384, y=140):
    h = int(RNG.uniform(5, 10))
    cv2.ellipse(f, (x, y), (max(2, h // 2), h), 0, 0, 360, (0, 140, 255), -1)
    cv2.ellipse(f, (x, y + h // 3), (1, max(1, h // 2)), 0, 0, 360, (210, 250, 255), -1)


def _watch(cfg, draw, n=120, fps=12.0):  # noqa: F811
    from app.flame import FlameWatch
    w = Catch()
    fw = FlameWatch(w, cfg)
    fw.stop()
    for i in range(n):
        f = _scene(i)
        draw(f, i / fps)
        fw.check(cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], 1), 1000 + i / fps)
    return w.alerts


def test_lighter_for_a_third_of_a_second_is_caught(cfg):  # noqa: F811
    alerts = _watch(cfg, lambda f, t: _lighter(f) if 5.0 <= t < 5.34 else None)
    assert len(alerts) == 1
    ts, (x1, y1, x2, y2) = alerts[0]
    assert 1005.0 <= ts <= 1005.4 and x1 <= 384 <= x2 + 1 and y1 - 12 <= 140 <= y2 + 12


def test_lamps_leds_screens_and_lights_do_not_alert(cfg):  # noqa: F811
    lamp = lambda f, t: cv2.circle(f, (60, 60), 5, (150, 220, 255), -1)                                   # noqa: E731
    lamp_on = lambda f, t: cv2.circle(f, (60, 60), 5, (150, 220, 255), -1) if t >= 5 else None             # noqa: E731
    led = lambda f, t: cv2.circle(f, (700, 40), 3, (40, 40, 255), -1) if int(t * 2) % 2 else None          # noqa: E731
    phone = lambda f, t: cv2.rectangle(f, (int(100 + 40 * t), 300), (int(112 + 40 * t), 320), (255, 230, 220), -1)  # noqa: E731
    lights = lambda f, t: cv2.add(f, (60, 60, 60, 0), dst=f) if t >= 5 else None                          # noqa: E731
    for draw in (lambda f, t: None, lamp, lamp_on, led, phone, lights):
        assert _watch(cfg, draw) == []


def test_only_chosen_cameras_use_the_full_picture_and_alert_with_a_close_up(cfg):  # noqa: F811
    from app.db import Database
    from app.pipeline import CameraWorker

    class Notes:
        sent = []

        def queue_alert(self, kind, text, jpeg, meta=None):
            self.sent.append((kind, text, jpeg))

    cam = cfg["cameras"][0]
    w = CameraWorker(cam, cfg, Database(cfg["database"], cfg["timezone"]), notifier=Notes())
    cfg["detector"]["stream"] = "fast"
    assert not w.flame_chosen() and w._want_stream() == "fast"
    cfg["hazard"]["flame_cameras"] = [cam["id"]]
    assert w.flame_chosen() and w._want_stream() == "high"
    w._sync_flame()
    assert w._flame is not None
    f = _scene(0)
    _lighter(f)
    w.flame_alert(time.time(), f, (380, 130, 388, 150))
    kind, text, jpeg = Notes.sent[0]
    assert kind == "flame" and "جداحة" in text and jpeg[:2] == b"\xff\xd8"
    assert [a["kind"] for a in w.db.alerts_after(0)] == ["flame"]
    cfg["hazard"]["flame_cameras"] = []
    w._sync_flame()
    assert w._flame is None


def test_at_most_four_cameras(cfg):  # noqa: F811
    from fastapi.testclient import TestClient
    from app.server import create_app
    c = TestClient(create_app(cfg, start_cameras=False))
    ok = c.post("/api/settings", json={"hazard": {"flame_cameras": ["a", "b"]}}).json()
    assert ok["hazard"]["flame_cameras"] == ["a", "b"]
    assert c.post("/api/settings", json={"hazard": {"flame_cameras": list("abcde")}}).status_code == 400
