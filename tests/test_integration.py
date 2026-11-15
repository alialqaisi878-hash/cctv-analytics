"""Pipeline + agent tests that need no camera, no model weights and no API key."""
import json
import time

import httpx2 as httpx
import numpy as np
import pytest

from app.config import load_config


class _Boxes:
    def __init__(self, xyxy, ids):
        import torch
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.id = torch.tensor(ids)


class _Res:
    def __init__(self, xyxy, ids):
        self.boxes = _Boxes(xyxy, ids) if ids else None


class FakeModel:
    """Emits one person walking from top to bottom of the frame across the door line."""

    def __init__(self):
        self.step = 0

    def track(self, frame, **kw):
        y = 50 + self.step * 40  # bottom of box moves 50 -> 450 px on a 480px frame
        self.step += 1
        return [_Res([[300, y - 120, 340, y]], [7])]


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config.example.yaml")
    c["database"] = str(tmp_path / "t.db")
    return c


def test_pipeline_counts_and_stores(cfg):
    from app.db import Database
    from app.pipeline import CameraWorker
    db = Database(cfg["database"], cfg["timezone"])
    w = CameraWorker(cfg["cameras"][0], cfg, db)
    w._model = FakeModel()
    frame = np.zeros((480, 640, 3), np.uint8)
    t = time.time()
    for i in range(11):
        w._process(frame.copy(), t + i * 0.5)
    assert w.lines[0].count_in == 1
    s = db.summary(db.today())
    assert s["total_in"] == 1 and s["total_out"] == 0
    assert w.latest_jpeg()[:2] == b"\xff\xd8"  # valid JPEG produced


def _fake_claude():
    """Mock Messages API: first asks for a tool, then answers."""
    calls = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body)
        if len(calls) == 1:
            content = [{"type": "tool_use", "id": "tu_1", "name": "get_day_summary", "input": {"day": "2026-01-01"}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "تقرير تجريبي"}]
            stop = "end_turn"
        return httpx.Response(200, json={"id": f"msg_{len(calls)}", "type": "message", "role": "assistant",
                                         "model": body["model"], "content": content, "stop_reason": stop,
                                         "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

    import anthropic
    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, calls


def test_agent_tool_loop(cfg):
    from app.agent import Analyst
    from app.db import Database
    client, calls = _fake_claude()
    a = Analyst(Database(cfg["database"], cfg["timezone"]), cfg, client=client)
    assert a.ask("كم واحد دخل؟") == "تقرير تجريبي"
    assert len(calls) == 2
    assert calls[0]["fallbacks"] == "default"
    assert {t["name"] for t in calls[0]["tools"]} >= {"get_day_summary", "get_hourly", "get_zone_activity"}
    tool_result = calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and '"total_in": 0' in tool_result["content"][0]["text"] \
        if isinstance(tool_result["content"], list) else '"total_in": 0' in tool_result["content"]
