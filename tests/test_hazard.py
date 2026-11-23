"""Fire and smoke: the watcher, its alert (picture + WhatsApp, at any hour) and its settings."""
import time

import numpy as np
import pytest
import torch

from tests.test_v11 import cfg  # noqa: F401  (fixture)


class _Boxes:
    def __init__(self, rows):
        t = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = t[:, :4], t[:, 4], t[:, 5]

    def __len__(self):
        return len(self.xyxy)


class _R:
    names = {0: "fire", 1: "smoke"}

    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class FakeFireModel:
    """Says "fire" on pictures whose top-left pixel is red, "smoke" when it is grey."""
    names = _R.names

    def __init__(self):
        self.calls = 0

    def predict(self, frames, conf, imgsz, verbose):
        self.calls += 1
        out = []
        for f in frames:
            px = tuple(int(v) for v in f[0, 0])
            rows = []
            if px == (0, 0, 255):
                rows.append([100, 100, 300, 300, 0.8, 0])
                rows.append([0, 0, 3, 3, 0.9, 0])          # a few pixels: ignored
            if px == (128, 128, 128):
                rows.append([50, 50, 400, 300, 0.4, 1])     # below the smoke threshold (0.5)
            out.append(_R(rows))
        return out


class Notes:
    def __init__(self):
        self.sent = []

    def queue_alert(self, kind, text, jpeg, meta=None):
        self.sent.append((kind, text, jpeg, meta))


def _frame(color, flicker=True):
    """A plain picture; with flicker the picture changes from look to look like real flames do."""
    f = np.zeros((480, 640, 3), np.uint8)
    f[:] = color
    if flicker:
        _frame.n = getattr(_frame, "n", 0) + 1
        f[100:300, 100:300, 1] = 150 * (_frame.n % 2)              # the flames' brightness moves
    return f


def _setup(cfg):  # noqa: F811
    from app.db import Database
    from app.hazard import HazardWatch
    from app.pipeline import CameraWorker
    cfg["hazard"].update(enabled=True, every_s=0)
    hz = HazardWatch(cfg, model=FakeFireModel())
    notes = Notes()
    w = CameraWorker(cfg["cameras"][0], cfg, Database(cfg["database"], cfg["timezone"]), notifier=notes, hazard=hz)
    return hz, w, notes


def _wait(cond, s=5.0):
    end = time.time() + s
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_fire_needs_two_of_three_looks_then_alerts_with_picture(cfg):  # noqa: F811
    hz, w, notes = _setup(cfg)
    hz.offer(w, _frame((0, 0, 255)), time.time())
    assert _wait(lambda: hz.looks == 1)
    time.sleep(0.1)
    assert notes.sent == []                                     # one sighting is not enough
    hz.offer(w, _frame((0, 0, 255)), time.time())
    assert _wait(lambda: hz.looks == 2)
    hz.offer(w, _frame((0, 0, 255)), time.time())
    assert _wait(lambda: len(notes.sent) == 1)
    kind, text, jpeg, meta = notes.sent[0]
    assert kind == "fire" and "نار" in text and "80%" in text and jpeg[:2] == b"\xff\xd8"
    assert meta["camera_id"] == w.id
    saved = w.db.alerts_after(0)
    assert [a["kind"] for a in saved] == ["fire"] and w.db.alert_image(saved[0]["id"]) is not None
    # the same fire again within the pause: no second message
    hz.offer(w, _frame((0, 0, 255)), time.time())
    assert _wait(lambda: hz.looks == 4)
    time.sleep(0.1)
    assert len(notes.sent) == 1
    assert hz.snapshot()["status"] == "watching"


def test_fire_that_never_changes_is_a_picture_not_a_fire(cfg):  # noqa: F811
    hz, w, notes = _setup(cfg)
    still = _frame((0, 0, 255), flicker=False)
    for i in range(4):
        hz.offer(w, still.copy(), time.time())
        assert _wait(lambda: hz.looks == i + 1)
    time.sleep(0.1)
    assert notes.sent == []


def test_weak_smoke_and_switched_off_do_nothing(cfg):  # noqa: F811
    hz, w, notes = _setup(cfg)
    for _ in range(3):
        hz.offer(w, _frame((128, 128, 128)), time.time())
        time.sleep(0.15)
    assert _wait(lambda: hz.looks >= 1)
    time.sleep(0.2)
    assert notes.sent == []
    cfg["hazard"]["enabled"] = False
    n = hz.looks
    hz.offer(w, _frame((0, 0, 255)), time.time())
    time.sleep(0.3)
    assert hz.looks == n


def test_each_camera_looked_at_only_every_few_seconds(cfg):  # noqa: F811
    hz, w, _ = _setup(cfg)
    cfg["hazard"]["every_s"] = 5
    t = time.time()
    hz.offer(w, _frame((0, 0, 0)), t)
    hz.offer(w, _frame((0, 0, 0)), t + 1)
    assert _wait(lambda: hz.looks == 1)
    time.sleep(0.2)
    assert hz.looks == 1


def test_fire_alert_goes_out_whatever_kinds_were_picked(cfg):  # noqa: F811
    from app.notify import Notifier
    cfg["notify"].update(provider="webhook", api_url="http://x/y", kinds=["after_hours"])
    n = Notifier(cfg)
    assert n.wants("fire") and n.wants("smoke") and not n.wants("crowding")


def test_changed_model_file_is_never_loaded(cfg, monkeypatch):  # noqa: F811
    from app import hazard
    monkeypatch.setattr(hazard, "MODEL_SHA256", "0" * 64)
    hz = hazard.HazardWatch(cfg)
    with pytest.raises(RuntimeError, match="checksum"):
        hz._load()
    assert hazard.MODEL.exists() and hazard.MODEL.stat().st_size > 1_000_000


def test_settings_and_program_check(cfg):  # noqa: F811
    from fastapi.testclient import TestClient
    from app.server import create_app
    c = TestClient(create_app(cfg, start_cameras=False))
    s = c.get("/api/settings").json()
    assert s["hazard"]["enabled"] is True and s["hazard"]["every_s"] == 5
    s = c.post("/api/settings", json={"hazard": {"enabled": False, "smoke_confidence": 0.6}}).json()
    assert s["hazard"]["enabled"] is False and s["hazard"]["smoke_confidence"] == 0.6
    assert "fire and smoke: on False" in c.get("/api/diagnostics").text
