"""v1.1: alert pictures, face recognition of enrolled people, WhatsApp/Telegram alerts, OpenAI assistant."""
import base64
import json
import shutil
import time
import urllib.request
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app.config import load_config, notice_text

SAMPLES = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
CACHE = Path(__file__).parent / ".cache"


def _download(name: str) -> Path:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / name
    if not p.exists():
        try:
            urllib.request.urlretrieve(SAMPLES + name, p)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"test picture not available: {e}")
    return p


@pytest.fixture
def cfg(tmp_path):
    c = load_config("tests/demo-config.yaml")
    c["database"] = str(tmp_path / "data" / "t.db")
    c["_settings_file"] = str(tmp_path / "data" / "settings.yaml")
    c["_cameras_file"] = str(tmp_path / "data" / "cameras.yaml")
    return c


@pytest.fixture(scope="session")
def engine_models(tmp_path_factory):
    """Face models, downloaded once per test run (or copied from tests/.cache)."""
    from app.faces import FaceEngine
    d = tmp_path_factory.mktemp("faces")
    cache = CACHE / "models"
    if cache.is_dir():
        shutil.copytree(cache, d / "models")
    e = FaceEngine(d)
    try:
        e.ensure_ready(background=False)
    except Exception:  # noqa: BLE001
        pass
    if not e.ready:
        pytest.skip(f"face models not available: {e.status}")
    cache.mkdir(parents=True, exist_ok=True)
    for f in (d / "models").glob("*.onnx"):
        if not (cache / f.name).exists():
            shutil.copy2(f, cache / f.name)
    return d / "models"


def _engine(tmp_path, models):
    from app.faces import FaceEngine
    # Arabic folder name on purpose: the program is installed in folders like that
    shutil.copytree(models, tmp_path / "برامج" / "models", dirs_exist_ok=True)
    e = FaceEngine(tmp_path / "برامج")
    e.ensure_ready(background=False)
    assert e.ready
    return e


# ---------- alerts with pictures ----------
def test_alert_picture_seen_and_retention(cfg):
    from app.db import Database
    db = Database(cfg["database"], cfg["timezone"])
    old = time.time() - 40 * 86400
    a_old = db.add_alert(old, "cam1", "after_hours", "old", b"\xff\xd8old")
    a_new = db.add_alert(time.time(), "cam1", "crowding", "new", b"\xff\xd8new")
    assert db.alert_image(a_new).read_bytes() == b"\xff\xd8new"
    assert db.unseen_count() == 2
    rows = db.alerts(db.today())
    assert rows[0]["has_image"] and not rows[0]["seen"] and "image" not in rows[0]
    db.mark_seen([a_new])
    assert db.unseen_count() == 1
    assert [a["id"] for a in db.alerts_after(a_old)] == [a_new]
    old_file = db.alert_image(a_old)
    db.purge_older_than(30)
    assert not old_file.exists() and db.alert_image(a_new) is not None
    assert db.alerts(kind="crowding")[0]["id"] == a_new


def test_old_database_gets_new_columns(cfg):
    import sqlite3
    from app.db import Database
    Path(cfg["database"]).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(cfg["database"])
    con.execute("CREATE TABLE alerts (id INTEGER PRIMARY KEY, ts REAL, day TEXT, camera_id TEXT, kind TEXT, message TEXT)")
    con.execute("INSERT INTO alerts(ts,day,camera_id,kind,message) VALUES (1,'2026-01-01','c','after_hours','x')")
    con.commit()
    con.close()
    db = Database(cfg["database"], cfg["timezone"])
    assert db.alerts("2026-01-01")[0]["has_image"] is False
    assert db.unseen_count() == 1


def test_notice_follows_settings(cfg):
    assert "لا يتم التعرف على الوجوه" in notice_text(cfg)
    cfg["faces"]["enabled"] = True
    assert "المسجّلين" in notice_text(cfg)


def test_alert_saved_with_marked_picture(cfg, monkeypatch):
    monkeypatch.setattr("app.pipeline.CONFIRM_HITS", 1)   # one picture is enough here
    from app.db import Database
    from app.pipeline import CameraWorker
    from tests.test_integration import FakeModel
    cfg["alerts"]["max_occupancy"] = 0
    cfg["alerts"]["after_hours_presence"] = True
    cfg["working_hours"]["days"] = []          # always "after hours"
    db = Database(cfg["database"], cfg["timezone"])
    sent = []

    class N:
        def queue_alert(self, kind, text, jpeg, meta=None):
            sent.append((kind, text, jpeg))

    w = CameraWorker(cfg["cameras"][0], cfg, db, notifier=N())
    w._model = FakeModel()
    w._process(np.zeros((480, 640, 3), np.uint8), time.time())
    a = db.alerts(db.today())
    assert len(a) == 1 and a[0]["kind"] == "after_hours" and a[0]["has_image"]
    assert db.alert_image(a[0]["id"]).read_bytes()[:2] == b"\xff\xd8"
    assert sent and sent[0][0] == "after_hours" and sent[0][2][:2] == b"\xff\xd8"


# ---------- face recognition ----------
def test_enroll_and_recognise(tmp_path, engine_models):
    e = _engine(tmp_path, engine_models)
    lena = cv2.imread(str(_download("lena.jpg")))
    messi = cv2.imread(str(_download("messi5.jpg")))
    p = e.enroll("لينا", lena, alert_on_seen=True)
    assert p["photos"] == 1 and e.thumb(p["id"]) is not None
    other = cv2.resize(cv2.flip(lena, 1), None, fx=0.7, fy=0.7)
    found = e.analyse(other)
    assert found and found[0]["person"]["name"] == "لينا" and found[0]["score"] > 0.6
    stranger = e.analyse(messi)
    assert stranger and stranger[0]["person"] is None
    with pytest.raises(ValueError):
        e.enroll("x", np.zeros((300, 300, 3), np.uint8))        # no face
    e.enroll("", other, person_id=p["id"])
    assert e.public_people()[0]["photos"] == 2
    # saved to disk and loaded again
    from app.faces import FaceEngine
    e2 = FaceEngine(tmp_path / "برامج")
    assert e2.public_people()[0]["name"] == "لينا"
    assert e.remove(p["id"]) and not e.people


def test_pipeline_names_person_and_alerts(cfg, tmp_path, engine_models, monkeypatch):
    monkeypatch.setattr("app.pipeline.CONFIRM_HITS", 1)   # one picture is enough here
    from app.db import Database
    from app.pipeline import CameraWorker
    e = _engine(tmp_path, engine_models)
    lena = cv2.imread(str(_download("lena.jpg")))
    p = e.enroll("Lena", lena, alert_on_seen=True)
    cfg["faces"]["enabled"] = True
    cfg["alerts"]["after_hours_presence"] = False
    db = Database(cfg["database"], cfg["timezone"])

    frame = np.zeros((720, 1280, 3), np.uint8)
    face = cv2.resize(lena, (240, 240))
    frame[60:300, 500:740] = face

    class Still:
        def track(self, frame, **kw):
            from tests.test_integration import _Res
            return [_Res([[460, 40, 780, 700]], [3])]

    w = CameraWorker(cfg["cameras"][0], cfg, db, faces=e)
    w._model = Still()
    w._process(frame.copy(), time.time())
    assert w.names[3]["name"] == "Lena"
    att = db.attendance(db.today())
    assert att and att[0]["person_id"] == p["id"]
    a = db.alerts(db.today())
    assert a and a[0]["kind"] == "person" and a[0]["person_id"] == p["id"] and a[0]["has_image"]
    assert w.latest_raw() is not None


def test_arabic_label_drawn():
    from app.pipeline import _label
    img = np.zeros((100, 300, 3), np.uint8)
    _label(img, "أحمد", (10, 60), (255, 170, 60))
    assert img.any()


# ---------- WhatsApp / Telegram ----------
def _mock(calls, status=200, body=None):
    def handler(req: httpx.Request):
        calls.append(req)
        return httpx.Response(status, json=body if body is not None else {"idMessage": "1", "sent": "true", "ok": True})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_whatsapp_number():
    from app.notify import whatsapp_number
    assert whatsapp_number("0770 123 4567") == "9647701234567"
    assert whatsapp_number("+964 770-123-4567") == "9647701234567"
    assert whatsapp_number("00964770") == "964770"


@pytest.mark.parametrize("provider,path", [("greenapi", "/waInstance1101/sendFileByUpload/TOK"),
                                           ("ultramsg", "/instance9/messages/image"),
                                           ("telegram", "/botTOK/sendPhoto")])
def test_notify_sends_picture(cfg, provider, path):
    from app.notify import Notifier
    calls = []
    cfg["notify"].update(provider=provider, instance="1101" if provider == "greenapi" else "instance9", token="TOK",
                         to="07701234567" if provider != "telegram" else "555")
    n = Notifier(cfg, client=_mock(calls))
    assert n.enabled and n.wants("person:abc")
    n.send("تنبيه تجربة", b"\xff\xd8jpeg")
    assert calls[0].url.path == path
    body = calls[0].content
    from urllib.parse import quote
    assert b"jpeg" in body or quote(base64.b64encode(b"\xff\xd8jpeg").decode(), safe="").encode() in body
    if provider == "greenapi":
        assert b"9647701234567@c.us" in body
    n.send("نص فقط")
    assert calls[1].url.path.endswith(("sendMessage/TOK", "/messages/chat", "/sendMessage"))


def test_notify_error_and_kinds(cfg):
    from app.notify import Notifier
    calls = []
    cfg["notify"].update(provider="greenapi", instance="1", token="bad", to="0770", kinds=["person"])
    n = Notifier(cfg, client=_mock(calls, 401, {"error": "Unauthorized"}))
    assert n.wants("person:x") and not n.wants("crowding")
    with pytest.raises(RuntimeError, match="401"):
        n.send("x")


def test_notify_webhook_n8n(cfg):
    import json as _json
    import time as _time
    from app.notify import Notifier
    calls = []
    cfg["site_name"] = "المخزن"
    cfg["notify"].update(provider="webhook", api_url="", token="s3", to="07701234567", instance="")
    n = Notifier(cfg, client=_mock(calls))
    assert not n.enabled
    cfg["notify"]["api_url"] = "https://n8n.test/webhook/cctv"
    assert n.enabled and n.wants("crowding")
    n.queue_alert("person:ab", "شوهد أحمد", b"\xff\xd8jpeg", {"camera_id": 2, "camera": "الباب", "person": "أحمد"})
    for _ in range(50):
        if calls:
            break
        _time.sleep(0.05)
    req = calls[0]
    assert str(req.url) == "https://n8n.test/webhook/cctv" and req.headers["X-CCTV-Token"] == "s3"
    body = _json.loads(req.content)
    assert body["message"] == "شوهد أحمد" and body["site"] == "المخزن" and body["kind"] == "person"
    assert body["camera"] == "الباب" and body["person"] == "أحمد" and body["to"] == "07701234567"
    assert base64.b64decode(body["image_base64"]) == b"\xff\xd8jpeg" and body["has_image"]
    n.send("تجربة")
    body = _json.loads(calls[1].content)
    assert body["kind"] == "test" and body["image_base64"] is None


# ---------- OpenAI assistant ----------
def test_openai_tool_loop(cfg):
    from app.agent import Analyst
    from app.db import Database
    calls = []

    def handler(req: httpx.Request):
        calls.append(req)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-4o"}, {"id": "gpt-5"}, {"id": "whisper-1"}]})
        body = json.loads(req.content)
        if len([m for m in body["messages"] if m["role"] == "tool"]) == 0:
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_day_summary", "arguments": '{"day": "2026-01-01"}'}}]}}]})
        assert '"total_in": 0' in body["messages"][-1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "جواب OpenAI"}}]})

    cfg["agent"]["api_key"] = "sk-proj-test"
    http = httpx.Client(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    a = Analyst(Database(cfg["database"], cfg["timezone"]), cfg, openai_http=http)
    assert a.provider == "openai"
    assert a.ask("كم واحد دخل؟") == "جواب OpenAI"
    chat = [json.loads(c.content) for c in calls if c.url.path.endswith("/chat/completions")]
    assert chat[0]["model"] == "gpt-5"
    assert {t["function"]["name"] for t in chat[0]["tools"]} >= {"get_day_summary", "get_attendance"}


def test_provider_detection(cfg):
    from app.agent import Analyst
    from app.db import Database
    a = Analyst(Database(cfg["database"], cfg["timezone"]), cfg)
    cfg["agent"]["api_key"] = "sk-ant-api03-x"
    assert a.provider == "claude"
    cfg["agent"]["api_key"] = "sk-abc"
    assert a.provider == "openai"
    cfg["agent"]["provider"] = "claude"
    assert a.provider == "claude"


def test_openai_bad_key(cfg):
    from app.agent import Analyst
    from app.db import Database
    cfg["agent"]["api_key"] = "sk-wrong"
    cfg["agent"]["openai_model"] = "gpt-x"
    http = httpx.Client(base_url="https://api.openai.com/v1",
                        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "bad"}})))
    a = Analyst(Database(cfg["database"], cfg["timezone"]), cfg, openai_http=http)
    assert "OpenAI" in a.ask("x") and "غير صحيح" in a.ask("x")


# ---------- web API ----------
def test_api_alerts_people_and_settings(cfg, tmp_path, engine_models):
    from fastapi.testclient import TestClient
    from app.server import create_app
    shutil.copytree(engine_models, tmp_path / "data" / "models", dirs_exist_ok=True)
    api = create_app(cfg, start_cameras=False)
    core = api.state.core
    c = TestClient(api)
    i = c.get("/api/info").json()
    assert i["unseen_alerts"] == 0 and i["faces"]["enabled"] is False
    aid = core.db.add_alert(time.time(), "demo", "after_hours", "تجربة", b"\xff\xd8x")
    feed = c.get(f"/api/alerts/feed?after={i['last_alert_id']}").json()
    assert feed["unseen"] == 1 and feed["alerts"][0]["id"] == aid
    assert c.get(f"/api/alerts/{aid}/image").content == b"\xff\xd8x"
    assert c.post("/api/alerts/seen", json={"ids": [aid]}).json()["unseen"] == 0
    assert c.get("/api/alerts/999/image").status_code == 404

    # face recognition: off -> clear message; on -> enroll from an uploaded picture
    jpg = base64.b64encode(_download("lena.jpg").read_bytes()).decode()
    r = c.post("/api/people", json={"name": "Lena", "image": "data:image/jpeg;base64," + jpg})
    assert r.status_code == 400 and "فعّل" in r.json()["detail"]
    s = c.post("/api/settings", json={"faces": {"enabled": True}}).json()
    assert s["faces"]["enabled"] is True
    for _ in range(100):
        if core.faces.ready:
            break
        time.sleep(0.1)
    r = c.post("/api/people", json={"name": "Lena", "image": "data:image/jpeg;base64," + jpg, "alert_on_seen": True})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert c.get(f"/api/people/{pid}/photo").status_code == 200
    ppl = c.get("/api/people").json()
    assert ppl["people"][0]["name"] == "Lena" and ppl["people"][0]["alert_on_seen"]
    assert c.patch(f"/api/people/{pid}", json={"alert_on_seen": False}).status_code == 200
    bad = c.post("/api/people", json={"name": "x", "image": "data:image/png;base64,AAAA"})
    assert bad.status_code == 400
    assert c.delete(f"/api/people/{pid}").status_code == 200
    assert "المسجّلين" in c.get("/api/info").json()["privacy"]["notice_text"]

    # notification settings: token never shown back; test button reports errors clearly
    s = c.post("/api/settings", json={"notify": {"provider": "greenapi", "instance": "1", "token": "T", "to": "0770"}}).json()
    assert s["notify"]["has_token"] and "token" not in s["notify"]
    calls = []
    core.notifier._client = _mock(calls)
    assert c.post("/api/notify/test").json()["ok"] is True and calls
    core.notifier._client = _mock(calls, 400, {"error": "wrong instance"})
    r = c.post("/api/notify/test")
    assert r.status_code == 502 and "wrong instance" in r.json()["detail"]
    assert c.post("/api/settings", json={"notify": {"provider": "sms"}}).status_code == 400
    assert c.post("/api/settings", json={"agent": {"provider": "openai", "openai_model": "gpt-5"}}).status_code == 200


# ---------- floor map: where each employee is ----------
def test_floor_maps_and_places(tmp_path):
    from app.floormap import FloorMaps
    fm = FloorMaps(tmp_path / "بيانات")
    with pytest.raises(ValueError):
        fm.add("x", None)
    m = fm.add("الطابق الأرضي", np.full((3000, 1500, 3), 200, np.uint8))
    assert (m["w"], m["h"]) == (1200, 2400) and fm.image(m["id"]).exists()
    fm.place("cam1", m["id"], 1.7, 0.25)
    assert fm.where("cam1") == {"map_id": m["id"], "map": "الطابق الأرضي", "x": 1.0, "y": 0.25}
    with pytest.raises(ValueError):
        fm.place("cam1", "nope")
    again = FloorMaps(tmp_path / "بيانات")          # kept on disk
    assert again.where("cam1")["map"] == "الطابق الأرضي"
    fm.place("cam1", None)
    assert fm.where("cam1") is None
    fm.place("cam2", m["id"], 0.5, 0.5)
    assert fm.remove(m["id"]) and fm.where("cam2") is None and fm.image(m["id"]) is None


def test_location_sent_to_webhook_when_person_moves(cfg, tmp_path, engine_models, monkeypatch):
    monkeypatch.setattr("app.pipeline.CONFIRM_HITS", 1)   # one picture is enough here
    import json as _json
    from app.db import Database
    from app.floormap import FloorMaps
    from app.notify import Notifier
    from app.pipeline import CameraWorker
    e = _engine(tmp_path, engine_models)
    lena = cv2.imread(str(_download("lena.jpg")))
    p = e.enroll("Lena", lena, erp_id="E-7", job="محاسبة")
    cfg["faces"]["enabled"] = True
    cfg["alerts"]["after_hours_presence"] = False
    cfg["notify"].update(provider="webhook", api_url="https://n8n.test/webhook/cctv", send_locations=True)
    db = Database(cfg["database"], cfg["timezone"])
    fm = FloorMaps(tmp_path / "data")
    m = fm.add("المكتب", np.zeros((400, 600, 3), np.uint8))
    fm.place(cfg["cameras"][0]["id"], m["id"], 0.3, 0.6)
    calls = []
    n = Notifier(cfg, client=_mock(calls))
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[60:300, 500:740] = cv2.resize(lena, (240, 240))

    class Still:
        def track(self, frame, **kw):
            from tests.test_integration import _Res
            return [_Res([[460, 40, 780, 700]], [3])]

    w = CameraWorker(cfg["cameras"][0], cfg, db, faces=e, notifier=n, maps=fm)
    w._model = Still()
    t0 = time.time()
    w._process(frame.copy(), t0)
    for _ in range(60):
        if calls:
            break
        time.sleep(0.05)
    body = _json.loads(calls[0].content)
    assert body["kind"] == "location" and body["person"]["erp_id"] == "E-7" and body["person"]["job"] == "محاسبة"
    assert body["location"]["map"] == "المكتب" and body["location"]["x"] == 0.3 and body["image_base64"] is None
    # staying in view keeps the place fresh, without sending again
    w._process(frame.copy(), t0 + 61)
    rows = db._query("SELECT COUNT(*) n FROM sightings WHERE person_id=?", (p["id"],))
    assert rows[0]["n"] == 2
    time.sleep(0.2)
    assert len(calls) == 1


def test_api_map_and_locations(cfg, tmp_path, engine_models):
    from fastapi.testclient import TestClient
    from app.server import create_app
    shutil.copytree(engine_models, tmp_path / "data" / "models", dirs_exist_ok=True)
    cfg["faces"]["enabled"] = True
    api = create_app(cfg, start_cameras=False)
    core = api.state.core
    for _ in range(100):
        if core.faces.ready:
            break
        time.sleep(0.1)
    c = TestClient(api)
    cam = cfg["cameras"][0]["id"]
    _, png = cv2.imencode(".png", np.full((300, 500, 3), 90, np.uint8))
    assert c.post("/api/maps", json={"name": "x", "image": "data:image/png;base64,AAAA"}).status_code == 400
    m = c.post("/api/maps", json={"name": "الطابق الأول", "image": "data:image/png;base64," + base64.b64encode(png).decode()}).json()
    assert c.get(f"/api/maps/{m['id']}/image").headers["content-type"] == "image/jpeg"
    assert c.put(f"/api/maps/cameras/{cam}", json={"map": m["id"], "x": 0.2, "y": 0.8}).status_code == 200
    assert c.put("/api/maps/cameras/nocam", json={"map": m["id"]}).status_code == 404
    assert c.get("/api/maps").json()["cams"][cam] == {"map": m["id"], "x": 0.2, "y": 0.8}

    jpg = base64.b64encode(_download("lena.jpg").read_bytes()).decode()
    pid = c.post("/api/people", json={"name": "Lena", "image": "data:image/jpeg;base64," + jpg,
                                      "job": "استقبال", "erp_id": "E-1"}).json()["id"]
    loc = c.get("/api/locations").json()["people"]
    assert loc[0]["camera"] is None and loc[0]["present"] is False
    core.db.add_sighting(time.time() - 120, cam, pid, "Lena", 0.8)
    loc = c.get("/api/locations?erp_id=E-1").json()["people"]
    assert loc[0]["present"] and loc[0]["minutes_ago"] == 2 and loc[0]["location"]["map"] == "الطابق الأول"
    assert loc[0]["job"] == "استقبال"
    assert c.get("/api/locations?erp_id=nobody").json()["people"] == []
    assert c.patch(f"/api/people/{pid}", json={"erp_id": "E-9", "job": "مدير"}).status_code == 200
    assert c.get("/api/people").json()["people"][0]["erp_id"] == "E-9"
    assert json.loads(core.analyst.locations and json.dumps(core.analyst.locations()))[0]["erp_id"] == "E-9"
    assert c.post("/api/settings", json={"notify": {"send_locations": True}, "faces": {"present_minutes": 30}}).status_code == 200
    assert c.delete(f"/api/maps/{m['id']}").status_code == 200
    assert c.get("/api/locations").json()["people"][0]["location"] is None


# ---------- live picture: no lag, scaled down, fast while watched ----------
def test_live_camera_never_lags_and_shows_fast(cfg):
    import threading
    from app.db import Database
    from app.pipeline import CameraWorker, VIEW_WIDTH

    class Cam:
        """A 4K camera at 25 pictures a second; each picture carries its number in the first pixel."""
        def __init__(self):
            self.n = 0
        def read(self):
            time.sleep(0.04)
            self.n += 1
            f = np.zeros((2160, 3840, 3), np.uint8)
            f[0, 0, 0] = self.n % 256
            f[0, 0, 1] = self.n // 256
            return (self.n <= 150), f

    class Slow:
        """Analysis slower than the camera (0.3 s per picture, like a busy CPU)."""
        def track(self, frame, **kw):
            time.sleep(0.3)
            from tests.test_integration import _Res
            return [_Res([[1000, 500, 1400, 1800]], [1])]

    cam = dict(cfg["cameras"][0], source="rtsp://test/stream")
    cfg["alerts"]["after_hours_presence"] = False
    w = CameraWorker(cam, cfg, Database(cfg["database"], cfg["timezone"]))
    w._model = Slow()
    seen = []
    orig = w._process
    def process(frame, ts):
        seen.append((int(frame[0, 0, 1]) * 256 + int(frame[0, 0, 0]), w_cam.n))
        orig(frame, ts)
    w._process = process
    w_cam = Cam()
    shots = []
    t = threading.Thread(target=w._loop, args=(w_cam,))
    t.start()
    while t.is_alive():
        j = w.live_jpeg()                   # someone is watching the whole time
        if j and (not shots or shots[-1] is not j):
            shots.append(j)
        time.sleep(0.02)
    t.join()
    # the analysis always got the newest picture: never more than a couple of pictures behind the camera
    assert len(seen) >= 5 and all(cam_n - got <= 2 for got, cam_n in seen), seen
    # pictures were skipped instead of queued up
    assert seen[-1][0] - seen[0][0] > len(seen)
    # the live picture changed far more often than the analysis ran, and is scaled down
    assert len(shots) > 2 * len(seen), (len(shots), len(seen))
    img = cv2.imdecode(np.frombuffer(shots[-1], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == VIEW_WIDTH
    assert w._analysed_jpeg is not None
    # making the live picture never slowed down reading the camera (25 a second)
    assert w.read_fps > 18, w.read_fps


def test_page_is_never_cached_and_knows_its_version(cfg):
    """After an update the app window must load the new page, not keep the old one."""
    from fastapi.testclient import TestClient
    from app.server import create_app
    from app.version import __version__
    c = TestClient(create_app(cfg, start_cameras=False))
    r = c.get("/")
    assert "no-store" in r.headers["cache-control"]
    assert f'const PAGE_VERSION = "{__version__}"' in r.text and "__PAGE_VERSION__" not in r.text
    assert "data-view-cam" in r.text


def test_light_stream_is_used_and_falls_back_to_the_main_one(cfg):
    from app.db import Database
    from app.discovery import sub_stream_url
    from app.pipeline import CameraWorker
    assert sub_stream_url("rtsp://u:p@10.0.0.5:554/Streaming/Channels/101") == "rtsp://u:p@10.0.0.5:554/Streaming/Channels/102"
    assert sub_stream_url("rtsp://u:p@10.0.0.5:554/cam/realmonitor?channel=2&subtype=0").endswith("channel=2&subtype=1")
    assert sub_stream_url("rtsp://u:p@10.0.0.5:554/h264Preview_01_main").endswith("h264Preview_01_sub")
    assert sub_stream_url("rtsp://u:p@10.0.0.5:554/h264") is None

    class Cap:
        def __init__(self, ok): self.ok = ok
        def isOpened(self): return self.ok
        def get(self, prop): return {cv2.CAP_PROP_FRAME_WIDTH: 704, cv2.CAP_PROP_FRAME_HEIGHT: 576}.get(prop, 0)
        def set(self, *a): return True
        def release(self): pass

    main = "rtsp://u:p@10.0.0.5:554/Streaming/Channels/101"
    w = CameraWorker(dict(cfg["cameras"][0], source=main), cfg, Database(cfg["database"], cfg["timezone"]))
    opened = []
    w._capture = lambda url: (opened.append(url), Cap(True))[1]
    w._open()
    assert opened == [main.replace("101", "102")] and w.stream == "sub" and w.size == "704x576"
    assert w.snapshot()["stream"] == "sub"

    opened.clear()                                  # camera without a second stream: the main one
    w._capture = lambda url: (opened.append(url), Cap(url == main))[1]
    w._open()
    assert opened == [main.replace("101", "102"), main] and w.stream == "main"
    opened.clear()                                  # and it does not wait for the missing stream again
    w._open()
    assert opened == [main]

    opened.clear()                                  # owner asked for full quality
    cfg["detector"]["stream"] = "high"
    w._open()
    assert opened == [main] and w.stream == "main"


def test_things_that_are_not_people_are_ignored(cfg):
    """Small specks, wide objects and a box seen only once are not counted or drawn as people."""
    from app.db import Database
    from app.pipeline import CameraWorker, CONFIRM_HITS
    from tests.test_integration import _Res

    boxes = [[10, 10, 14, 15],          # a speck
             [100, 300, 400, 400],      # wide: a desk, a car, a shadow
             [500, 100, 560, 400]]      # a person
    class M:
        def track(self, frame, **kw):
            return [_Res(boxes, [1, 2, 3])]
    cfg["alerts"]["after_hours_presence"] = False
    w = CameraWorker(cfg["cameras"][0], cfg, Database(cfg["database"], cfg["timezone"]))
    w._model = M()
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(CONFIRM_HITS):
        w._process(frame, time.time())
        assert w.people_now == (1 if i == CONFIRM_HITS - 1 else 0)
    assert [t[0] for t in w._tracks] == [3]
