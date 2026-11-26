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
    c["_names_file"] = str(tmp_path / "data" / "camera_names.yaml")
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


def test_alert_saved_with_marked_picture(cfg):
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
    for _ in range(2):                         # a person counts once seen in two analyses
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


def test_pipeline_names_person_and_alerts(cfg, tmp_path, engine_models):
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
    assert w.names[1]["name"] == "Lena"          # the first person this camera found is number 1
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


def test_location_sent_to_webhook_when_person_moves(cfg, tmp_path, engine_models):
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
    def process(frame, ts, *a):
        seen.append((int(frame[0, 0, 1]) * 256 + int(frame[0, 0, 0]), w_cam.n))
        orig(frame, ts, *a)
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


def _raw_worker(cfg, frames):
    """A worker whose detector reports the given plain detections [(box, sureness), ...] picture by picture."""
    from app.db import Database
    from app.pipeline import CameraWorker
    from tests.test_integration import _Res
    w = CameraWorker(cfg["cameras"][0], cfg, Database(cfg["database"], cfg["timezone"]))

    class M:
        i = 0
        def track(self, frame, **kw):
            assert kw["conf"] == 0.25               # weaker guesses are asked for too (the setting is 0.5)
            w._raw_boxes = [(list(b), c) for b, c in frames[min(M.i, len(frames) - 1)]]
            M.i += 1
            return [_Res([], [])]
    w._model = M()
    return w


def test_things_that_are_not_people_are_ignored(cfg):
    """Specks and wide objects are never people; a box seen in one picture only is not counted;
    furniture the detector half-believes in is never counted, even when someone walks in front of it."""
    cfg["alerts"]["after_hours_presence"] = False
    person, chair = [500, 100, 560, 400], [200, 200, 280, 380]
    speck, desk = ([10, 10, 14, 15], 0.9), ([100, 300, 400, 400], 0.9)
    frames = [[speck, desk, (person, 0.9), (chair, 0.3)],        # a ghost: sure, but in one picture only
              [speck, desk, (chair, 0.3)], [speck, (chair, 0.35)], [(chair, 0.3)], [(chair, 0.3)],
              [(person, 0.9), (chair, 0.3)], [(person, 0.9), (chair, 0.3)], [(person, 0.9), (chair, 0.3)]]
    w = _raw_worker(cfg, frames)
    seen, t = [], time.time()
    for i in range(len(frames)):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[150:420, 180:300] = 40 * (i % 2)       # things move in front of the chair the whole time
        w._process(frame, t + i * 0.2)
        seen.append(w.people_now)
    assert seen == [0, 0, 0, 0, 0, 0, 1, 1], seen
    assert [t[1] for t in w._tracks] == [tuple(float(v) for v in person)]
    assert w.snapshot()["best_conf"] == 0.9


def test_person_at_the_edge_of_the_picture_is_counted(cfg):
    """Ali stood at the very edge of the picture, partly cut off: the detector was only 0.2-0.3 sure, under the
    0.5 setting, so he was never counted. A weak detection that comes in and moves like a person now counts,
    and one missed picture does not make the person disappear."""
    cfg["alerts"]["after_hours_presence"] = True
    cfg["working_hours"]["days"] = []
    boxes = [[0, 200 - 10 * i, 40 + 12 * i, 420] for i in range(6)]         # walking in from the left edge
    frames = [[]] * 2 + [[(b, 0.3)] for b in boxes] + [[]] + [[(boxes[-1], 0.28)]] * 2 + [[]] * 20
    w = _raw_worker(cfg, frames)
    seen, t = [], time.time()
    for i, f in enumerate(frames):
        frame = np.full((480, 640, 3), 30, np.uint8)
        for b, _ in f:
            frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 200            # the person really is in the picture
        w._process(frame, t + i * 0.2)
        seen.append(w.people_now)
    assert seen[:5] == [0] * 5 and seen[5:11] == [1] * 6, seen             # counted from the third time seen
    assert seen[-1] == 0                                                     # and gone a moment after leaving
    assert [a["kind"] for a in w.db.alerts(w.db.today())] == ["after_hours"]


def test_one_error_never_stops_a_camera(cfg):
    """An error while analysing a picture is written down and shown, and the camera carries on."""
    cfg["alerts"]["after_hours_presence"] = False
    w = _raw_worker(cfg, [[([500, 100, 560, 400], 0.9)]])
    good = w._model.track
    def bad(frame, **kw):
        raise RuntimeError("boom")
    frame = np.zeros((480, 640, 3), np.uint8)
    w._model.track = bad
    w._analyse(frame, time.time())
    assert "boom" in w.snapshot()["error"] and w.errors == 1 and w.snapshot()["analysed_ago"] is None
    w._model.track = good
    w._analyse(frame, time.time())
    w._analyse(frame, time.time())
    assert w.people_now == 1 and w.snapshot()["analysed_ago"] is not None


def test_api_key_typed_in_the_model_box_is_moved_to_the_key(cfg):
    from app.config import apply_settings, load_config
    key = "sk-proj-" + "x" * 150
    apply_settings(cfg, {"agent": {"openai_model": key}})
    assert cfg["agent"]["openai_model"] == "" and cfg["agent"]["api_key"] == key
    # a key already given in its own box wins; the model box is just cleared
    apply_settings(cfg, {"agent": {"openai_model": "sk-other", "api_key": "sk-proj-right"}})
    assert cfg["agent"]["openai_model"] == "" and cfg["agent"]["api_key"] == "sk-proj-right"
    # an old install that already saved a key there is cleaned when it starts
    import shutil as _sh
    import yaml
    d = Path(cfg["_settings_file"]).parent.parent / "inst"
    (d / "data").mkdir(parents=True)
    _sh.copy("tests/demo-config.yaml", d / "config.yaml")
    (d / "data" / "settings.yaml").write_text(yaml.safe_dump({"agent": {"openai_model": "sk-proj-abc"}}), encoding="utf-8")
    assert load_config(d / "config.yaml")["agent"]["openai_model"] == ""


def test_person_counted_and_alerted_before_the_tracker_follows_them(cfg):
    """Someone walking fast past a busy PC: the tracker never gives them an id, but they are still
    counted, blurred and alerted on (Ali walked past a camera and got no alert in 1.1.3)."""
    from app.db import Database
    from app.pipeline import CameraWorker
    from tests.test_integration import _Res
    cfg["alerts"]["after_hours_presence"] = True
    cfg["working_hours"]["days"] = []
    db = Database(cfg["database"], cfg["timezone"])
    w = CameraWorker(cfg["cameras"][0], cfg, db)

    class M:
        x = 0
        def track(self, frame, **kw):
            M.x += 120                                   # a big jump between analyses: no id from the tracker
            w._raw_boxes = [([M.x, 100, M.x + 60, 400], 0.9)]
            return [_Res([], [])]
    w._model = M()
    for _ in range(3):
        w._process(np.zeros((480, 640, 3), np.uint8), time.time())
    assert w.people_now == 1 and w._tracks and w._tracks[0][0] == 1   # one person, the same one each time
    assert [a["kind"] for a in db.alerts(db.today())] == ["after_hours"]


def test_diagnostics_show_what_the_program_does_without_secrets(cfg):
    """The owner can copy one text that tells what every camera is doing and what went wrong."""
    from pathlib import Path as _P
    from fastapi.testclient import TestClient
    from app.pipeline import CameraWorker
    from app.server import create_app
    from app.version import __version__
    api = create_app(cfg, start_cameras=False)
    core = api.state.core
    cam = cfg["cameras"][0]
    w = CameraWorker(cam, cfg, core.db)
    w.status, w.last_error, w.errors, w.best_conf = "online", "RuntimeError: boom", 3, 0.31
    core.workers[cam["id"]] = w
    log = _P(cfg["database"]).parent / "app.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("opening rtsp://admin:hunter2@192.168.1.9/s\nkey sk-proj-abcdefghijklmnop\nTraceback: نص عربي\n", encoding="utf-8")
    c = TestClient(api)
    t = c.get("/api/diagnostics").text
    assert f"CCTV Analytics {__version__}" in t and cam["name"] in t
    assert "last error: RuntimeError: boom" in t and "best guess 0.31" in t and "inside working hours now" in t
    assert "hunter2" not in t and "rtsp://***@192.168.1.9" in t and "abcdefghijklmnop" not in t and "نص عربي" in t
    assert "attachment" in c.get("/api/diagnostics?download=1").headers["content-disposition"]
    snap = [x for x in c.get("/api/cameras").json() if x["id"] == cam["id"]][0]
    assert snap["error"] == "RuntimeError: boom" and snap["best_conf"] == 0.31 and snap["analysed_ago"] is None
    page = c.get("/").text
    assert "diag-open" in page and "vw-health" in page


def test_one_shared_detector_serves_many_cameras_busiest_first(cfg):
    """Ali has 32 cameras: each with its own model, none got analysed. One model now serves all of them,
    a few pictures at a time, and a camera where something moves goes before the quiet ones."""
    import threading
    from app.detector import SharedDetector

    calls = []

    class Boxes:
        def __init__(self, n):
            import torch
            self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 150.0]] * n)
            self.conf = torch.tensor([0.9] * n)
        def __len__(self):
            return len(self.conf)

    class Res:
        def __init__(self, n):
            self.boxes = Boxes(n)

    class Model:
        def predict(self, frames, **kw):
            calls.append([int(f[0, 0, 0]) for f in frames])
            time.sleep(0.05)
            return [Res(int(f[0, 0, 0]) == 7) for f in frames]

    d = SharedDetector(cfg, model=Model(), batch=2)
    out = {}

    def ask(cam, prio):
        f = np.zeros((10, 10, 3), np.uint8)
        f[0, 0, 0] = cam
        out[cam] = d.detect(f"c{cam}", f, prio)

    ask(1, 0)                                           # warm up: the detector thread is running
    calls.clear()
    ts = [threading.Thread(target=ask, args=(c, 3 if c == 7 else 0)) for c in (2, 3, 4, 5, 6, 7)]
    with d._cond:                                       # all six wait together before the detector picks
        for t in ts:
            t.start()
        while len(d._pending) < 6:
            d._cond.wait(0.01)
    for t in ts:
        t.join()
    assert sorted(sum(calls, [])) == [2, 3, 4, 5, 6, 7] and max(len(c) for c in calls) == 2
    assert 7 in calls[0]                                # the camera where something moves went first
    assert out[7] == [((10.0, 10.0, 50.0, 150.0), 0.8999999761581421)] and out[2] == []
    d.stop()
    # a newer picture of the same camera replaces the older one still waiting
    gate = threading.Event()

    class Busy(Model):
        def predict(self, frames, **kw):
            gate.wait(5)
            return super().predict(frames, **kw)

    d = SharedDetector(cfg, model=Busy(), batch=1)
    threading.Thread(target=ask, args=(9, 0)).start()            # keeps the detector busy
    while d._pending:
        time.sleep(0.01)
    t = threading.Thread(target=ask, args=(8, 0)); t.start()
    while "c8" not in d._pending:
        time.sleep(0.01)
    newer = {}
    t2 = threading.Thread(target=lambda: newer.setdefault("r", d.detect("c8", np.zeros((10, 10, 3), np.uint8), 0)))
    t2.start()
    t.join(3)
    assert not t.is_alive() and out[8] is None       # the old picture is dropped: the camera goes on with its newest
    gate.set()
    t2.join(3)
    assert newer["r"] == []
    d.stop()


def test_app_cameras_share_one_detector(cfg):
    from app.server import create_app
    cfg["cameras"] = [dict(cfg["cameras"][0], id=f"c{i}", source="rtsp://127.0.0.1:9/none") for i in range(3)]
    api = create_app(cfg, start_cameras=True)
    core = api.state.core
    ws = list(core.workers.values())
    assert len(ws) == 3 and all(w.detector is core.detector for w in ws) and all(w._model is None for w in ws)
    for w in ws:
        w.stop()
    core.detector.stop()


# ---------- 1.1.8: a slow PC never throws its work away ----------
def test_slow_detector_still_delivers_and_shrinks_its_batch(cfg):
    """On Ali's PC one batch of 8 took ~16 s, longer than a camera waited (3 s), so every result was thrown
    away. A picture already being analysed is now always waited for, and the batch shrinks to stay ~1 s."""
    import threading
    from app.detector import SharedDetector

    class Slow:
        def predict(self, frames, **kw):
            time.sleep(0.6 * len(frames))               # 0.6 s a picture: a batch of 8 would take 4.8 s
            return [type("R", (), {"boxes": None})() for _ in frames]

    d = SharedDetector(cfg, model=Slow())
    d.max_batch = 8
    d.batch = 8
    got = {}

    def ask(i):
        f = np.zeros((10, 10, 3), np.uint8)
        got[i] = d.detect(f"c{i}", f, 1.0, None, timeout=0.5)

    ts = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
    with d._cond:
        for t in ts:
            t.start()
        while len(d._pending) < 8:
            d._cond.wait(0.01)
    for t in ts:
        t.join(20)
    assert all(got.get(i) == [] for i in range(8)), got     # nothing thrown away although it took 4.8 s
    assert d.batch == 1 and d.batch_s > 4                   # 0.6 s a picture: one at a time keeps it ~1 s
    d.stop()


def test_quiet_camera_still_gets_its_turn(cfg):
    """Busy cameras climb faster, but a camera that waited long enough goes before a busy one just analysed."""
    import threading
    from app.detector import SharedDetector

    order = []

    class Model:
        def predict(self, frames, **kw):
            order.extend(int(f[0, 0, 0]) for f in frames)
            return [type("R", (), {"boxes": None})() for _ in frames]

    d = SharedDetector(cfg, model=Model(), batch=1)
    now = time.time()

    def ask(cam, weight, since):
        f = np.zeros((10, 10, 3), np.uint8)
        f[0, 0, 0] = cam
        d.detect(f"c{cam}", f, weight, since)

    ts = [threading.Thread(target=ask, args=a) for a in ((1, 8.0, now - 0.5), (2, 1.0, now - 30))]
    with d._cond:
        for t in ts:
            t.start()
        while len(d._pending) < 2:
            d._cond.wait(0.01)
    for t in ts:
        t.join(5)
    assert order == [2, 1]          # quiet for 30 s (1 x 30) beats busy but just seen (8 x 0.5)
    d.stop()


def test_many_cameras_decode_light_and_only_make_pictures_when_needed(cfg):
    """32 cameras decoding at full speed left no processor for the detector: each stream decodes on one
    thread, and a frame becomes a picture only when the analysis wants one or someone is watching."""
    import threading
    from app.db import Database
    from app.pipeline import CameraWorker

    cfg["cameras"] = [dict(cfg["cameras"][0], id=f"c{i}") for i in range(32)]
    cam = dict(cfg["cameras"][0], source="rtsp://test/stream")
    w = CameraWorker(cam, cfg, Database(cfg["database"], cfg["timezone"]))
    assert w._decode_threads() == 1
    cfg["cameras"] = cfg["cameras"][:2]
    assert w._decode_threads() == 0

    class Cam:
        def __init__(self):
            self.grabs = self.made = 0
        def grab(self):
            time.sleep(0.01)
            self.grabs += 1
            return self.grabs <= 300
        def retrieve(self):
            self.made += 1
            return True, np.zeros((120, 160, 3), np.uint8)

    class Slow:
        def track(self, frame, **kw):
            time.sleep(0.25)
            from tests.test_integration import _Res
            return [_Res([], [])]

    w._model = Slow()
    c = Cam()
    t = threading.Thread(target=w._loop, args=(c,))
    t.start()
    t.join(30)
    assert c.grabs > 290 and 3 <= c.made < c.grabs / 4, (c.grabs, c.made)   # most frames never became pictures
    assert w.analysed_at


# ---------- 1.1.9: faster ----------
def test_very_sure_moving_person_counts_at_first_sight(cfg):
    """With 32 cameras the next look at a camera can be seconds away: someone the detector is very sure of
    and who moves counts at once; a sure box that does not move still has to be seen twice."""
    cfg["alerts"]["after_hours_presence"] = True
    cfg["working_hours"]["days"] = []
    person = [300, 100, 380, 400]
    w = _raw_worker(cfg, [[]] + [[(person, 0.8)]] * 2)
    t = time.time()
    w._process(np.full((480, 640, 3), 30, np.uint8), t)                 # the empty room
    frame = np.full((480, 640, 3), 30, np.uint8)
    frame[100:400, 300:380] = 200                                        # someone walks in
    w._process(frame, t + 3)
    assert w.people_now == 1
    assert [a["kind"] for a in w.db.alerts(w.db.today())] == ["after_hours"]

    still = _raw_worker(cfg, [[]] + [[(person, 0.8)]] * 2)
    still._process(np.full((480, 640, 3), 30, np.uint8), t)
    still._process(np.full((480, 640, 3), 30, np.uint8), t + 3)           # a sure box, nothing moved: wait
    assert still.people_now == 0
    still._process(np.full((480, 640, 3), 30, np.uint8), t + 3.5)
    assert still.people_now == 1


def test_fast_engine_kept_in_a_plain_folder_and_never_required(cfg, monkeypatch, tmp_path):
    """OpenVINO cannot open files under an Arabic folder name (the app is installed under D:\\برامج), so the
    converted model goes to a plain folder; and if it cannot be made, the normal engine is used."""
    from app import detector as D
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    dirs = list(D._plain_dirs(Path(tmp_path / "برامج")))
    assert dirs and dirs[0] == (tmp_path / "local" / "CCTVAnalytics").absolute() and all(str(d).isascii() for d in dirs)

    assert not D.SharedDetector._wants_openvino({"engine": "pytorch"})
    assert not D.SharedDetector._wants_openvino({"device": "cuda:0"})

    def broken(model, det):
        raise RuntimeError("no openvino here")
    monkeypatch.setattr(D.SharedDetector, "_wants_openvino", staticmethod(lambda det: True))
    monkeypatch.setattr(D.SharedDetector, "_openvino_model", staticmethod(broken))
    cfg["detector"]["model"] = "models/yolo11n.pt"
    d = D.SharedDetector(cfg)
    for _ in range(300):
        if d.status in ("ready",) or d.status.startswith("model error"):
            break
        time.sleep(0.1)
    assert d.status == "ready" and d.engine == "pytorch" and "no openvino here" in d.last_error
    assert d.detect("c1", np.zeros((480, 640, 3), np.uint8), timeout=30) == []
    d.stop()


# ---------- 1.2.0: running day and night ----------
def test_service_restarts_the_server_and_stops_only_when_asked(tmp_path, monkeypatch):
    """The program now runs in the background: a crash or an online update starts the server again,
    and only a stop request (icon next to the clock, settings page, uninstaller) ends it."""
    import subprocess
    import sys as _sys
    import run as runner
    from app.autostart import STOP_EXIT_CODE
    from app.updater import RESTART_EXIT_CODE

    codes = [1, RESTART_EXIT_CODE, STOP_EXIT_CODE]          # crash, update, stop
    started = []
    svc = runner.Service("config.yaml", 1, "http://localhost:1")
    svc.log_path = tmp_path / "app.log"

    def start_server():
        started.append(time.time())
        code = codes[min(len(started) - 1, len(codes) - 1)]
        svc.server = subprocess.Popen([_sys.executable, "-c", f"import sys, time; time.sleep(0.3); sys.exit({code})"])
        svc.started_at = time.time()
    monkeypatch.setattr(svc, "start_server", start_server)
    monkeypatch.setattr(svc, "_keep_awake", lambda: None)
    assert svc.loop() == 0
    assert len(started) == 3 and svc.restarts == 1
    log = svc.log_path.read_text()
    assert "stopped unexpectedly" in log and "online update" in log and "stopped on request" in log


def test_service_starts_a_server_that_stopped_answering(tmp_path, monkeypatch):
    import subprocess
    import sys as _sys
    import run as runner
    svc = runner.Service("config.yaml", 1, "http://localhost:1")
    svc.log_path = tmp_path / "app.log"
    svc.HEALTH_EVERY, svc.HEALTH_GRACE, svc.HEALTH_FAILS = 0, 0, 2
    started = []

    def start_server():
        started.append(1)
        svc.server = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)"])
        svc.started_at = time.time() - 1
        if len(started) == 2:
            threading.Timer(1.5, svc.stopping.set).start()
    import threading
    monkeypatch.setattr(svc, "start_server", start_server)
    monkeypatch.setattr(svc, "_keep_awake", lambda: None)
    monkeypatch.setattr(runner, "_alive", lambda port: False)    # a frozen server
    svc.loop()
    assert len(started) == 2 and "stopped answering" in svc.log_path.read_text()
    assert svc.server.poll() is not None                          # nothing left running


def test_quit_only_from_this_pc_and_startup_settings(cfg, monkeypatch):
    from fastapi.testclient import TestClient
    from app import server as srv
    from app.server import create_app
    api = create_app(cfg, start_cameras=False)
    fired = []
    monkeypatch.setattr(srv.threading, "Timer", lambda t, f: type("T", (), {"start": lambda self: fired.append(t)})())
    assert TestClient(api, client=("192.168.1.20", 5000)).post("/api/window/quit").status_code == 403
    assert not fired
    assert TestClient(api, client=("127.0.0.1", 5000)).post("/api/window/quit").json() == {"ok": True} and fired
    c = TestClient(api)
    assert c.get("/api/settings").json()["startup"] == {"with_windows": True, "keep_awake": True}
    assert c.post("/api/settings", json={"startup": {"with_windows": False}}).json()["startup"]["with_windows"] is False
