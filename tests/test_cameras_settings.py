"""Adding cameras from the dashboard and the in-program settings: no real camera needed."""
import hashlib
import re
import socket
import threading

import pytest
import yaml
from pathlib import Path

from app import config, discovery


class FakeCamera:
    """A tiny RTSP server with digest login that knows one stream path."""

    def __init__(self, path="/Streaming/Channels/101", user="admin", password="p@ss:word/1"):
        self.path, self.user, self.password = path, user, password
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.auth_failures = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        with conn:
            buf = b""
            while True:
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\r\n\r\n" in buf:
                    req, buf = buf.split(b"\r\n\r\n", 1)
                    conn.sendall(self._answer(req.decode()).encode())

    def _answer(self, req: str) -> str:
        method, uri = req.split()[:2]
        cseq = re.search(r"CSeq: (\d+)", req).group(1)
        if method == "OPTIONS":
            return f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nPublic: DESCRIBE\r\n\r\n"
        auth = re.search(r'Authorization: Digest username="([^"]*)".*response="([^"]*)"', req)
        if not auth:
            return (f"RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\n"
                    'WWW-Authenticate: Digest realm="cam", nonce="abc123"\r\n\r\n')
        md5 = lambda x: hashlib.md5(x.encode()).hexdigest()  # noqa: E731
        good = md5(f"{md5(f'{self.user}:cam:{self.password}')}:abc123:{md5(f'DESCRIBE:{uri}')}")
        if auth.group(1) != self.user or auth.group(2) != good:
            self.auth_failures += 1
            return f"RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\nWWW-Authenticate: Digest realm=\"cam\", nonce=\"abc123\"\r\n\r\n"
        if not uri.endswith(self.path):
            return f"RTSP/1.0 404 Not Found\r\nCSeq: {cseq}\r\n\r\n"
        return f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Length: 0\r\n\r\n"


def test_find_stream_with_right_login():
    cam = FakeCamera(path="/cam/realmonitor?channel=1&subtype=0")
    r = discovery.find_stream("127.0.0.1", cam.port, "admin", "p@ss:word/1")
    assert r["status"] == "ok" and r["vendor"].startswith("Dahua")
    # the password is URL-encoded so that @ : / inside it do not break the address
    assert r["url"] == f"rtsp://admin:p%40ss%3Aword%2F1@127.0.0.1:{cam.port}/cam/realmonitor?channel=1&subtype=0"


def test_find_stream_wrong_password_stops_after_one_attempt():
    cam = FakeCamera()
    assert discovery.find_stream("127.0.0.1", cam.port, "admin", "wrong") == {"status": "auth"}
    assert cam.auth_failures == 1   # cameras lock the account after several failures


def test_find_stream_channel_and_unreachable():
    cam = FakeCamera(path="/Streaming/Channels/301")
    assert discovery.find_stream("127.0.0.1", cam.port, "admin", "p@ss:word/1", channel=3)["status"] == "ok"
    assert discovery.find_stream("127.0.0.1", cam.port, "admin", "p@ss:word/1", channel=9)["status"] == "not_found"
    free = socket.socket(); free.bind(("127.0.0.1", 0)); port = free.getsockname()[1]; free.close()
    assert discovery.find_stream("127.0.0.1", port, "admin", "x")["status"] == "unreachable"


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"site_name": "X", "cameras": [
        {"id": "demo", "source": "samples/people.mp4", "lines": []},
        {"id": "door", "source": "rtsp://10.0.0.5/1"}]}), encoding="utf-8")
    return p


def test_demo_camera_is_hidden_and_added_cameras_persist(cfg_path):
    cfg = config.load_config(cfg_path)
    assert [c["id"] for c in cfg["cameras"]] == ["door"]
    cam = {"id": "cam_10_0_0_9", "name": "مدخل", "source": "rtsp://a:b@10.0.0.9:554/",
           "lines": [{"id": "door1", "points": [[0.1, 0.5], [0.9, 0.5]]}], "zones": []}
    config.add_local_camera(cfg["_cameras_file"], cam)
    again = config.load_config(cfg_path)
    assert [c["id"] for c in again["cameras"]] == ["door", "cam_10_0_0_9"]
    assert again["cameras"][1]["lines"][0]["invert"] is False
    assert config.local_camera_ids(cfg["_cameras_file"]) == {"cam_10_0_0_9"}
    assert config.set_camera_lines(cfg["_cameras_file"], "cam_10_0_0_9",
                                   [{"id": "door1", "name": "خط", "points": [[0, 0], [1, 1]], "invert": True}])
    assert config.load_config(cfg_path)["cameras"][1]["lines"][0]["invert"] is True
    assert config.remove_local_camera(cfg["_cameras_file"], "cam_10_0_0_9")
    assert not config.remove_local_camera(cfg["_cameras_file"], "door")
    assert "samples/people.mp4" in cfg_path.read_text(encoding="utf-8")   # config.yaml itself is never rewritten


def test_settings_are_validated_applied_and_kept(cfg_path):
    cfg = config.load_config(cfg_path)
    config.apply_settings(cfg, {"site_name": " شركتي ", "working_hours": {"start": "9:5", "end": "18:00", "days": [6, 0, 6]},
                                "privacy": {"retention_days": 60}, "agent": {"api_key": "sk-test"},
                                "server": {"host": "evil"}, "database": "x.db"})
    assert cfg["site_name"] == "شركتي" and cfg["working_hours"] == {"start": "09:05", "end": "18:00", "days": [0, 6]}
    assert cfg["server"]["host"] == "0.0.0.0" and cfg["database"] == "data/cctv.db"   # not editable from the page
    again = config.load_config(cfg_path)
    assert again["privacy"]["retention_days"] == 60 and again["agent"]["api_key"] == "sk-test"
    pub = config.public_settings(again)
    assert pub["agent"]["has_api_key"] is True and "api_key" not in pub["agent"] and "sk-test" not in str(pub)
    for bad in ({"working_hours": {"start": "25:00"}}, {"privacy": {"retention_days": 0}}, {"working_hours": {"days": []}}):
        with pytest.raises(ValueError):
            config.apply_settings(cfg, bad)
    assert config.load_config(cfg_path)["privacy"]["retention_days"] == 60


def test_rename_any_camera_and_keep_the_name(cfg_path):
    from fastapi.testclient import TestClient
    from app.server import create_app
    cfg = config.load_config(cfg_path)
    cfg["database"] = str(Path(cfg["_cameras_file"]).parent / "t.db")
    cam = cfg["cameras"][0]
    c = TestClient(create_app(cfg, start_cameras=False))
    r = c.put(f"/api/cameras/{cam['id']}/name", json={"name": "  مخزن\nالطابق الأول  "})
    assert r.status_code == 200 and r.json()["name"] == "مخزن الطابق الأول"
    assert cam["name"] == "مخزن الطابق الأول"
    assert [x["name"] for x in c.get("/api/cameras").json() if x["id"] == cam["id"]] == ["مخزن الطابق الأول"]
    assert c.put(f"/api/cameras/{cam['id']}/name", json={"name": "   "}).status_code == 400
    assert c.put("/api/cameras/nope/name", json={"name": "x"}).status_code == 404
    assert config.read_camera_names(cfg["_names_file"]) == {cam["id"]: "مخزن الطابق الأول"}
    again = config.load_config(cfg_path)                       # after a restart: still the new name
    assert [x["name"] for x in again["cameras"] if x["id"] == cam["id"]] == ["مخزن الطابق الأول"]


def test_dahua_camera_abilities_are_read_and_a_refused_login_stops_at_once():
    from app import camera_api

    class R:
        def __init__(self, code, text=""):
            self.status_code, self.text = code, text

    answers = {
        "/cgi-bin/magicBox.cgi?action=getDeviceType": R(200, "type=IPC-HFW3449T1-AS-PV\r\n"),
        "/cgi-bin/magicBox.cgi?action=getSoftwareVersion": R(200, "version=2.840.0000000.13.R,build:2024-01-01\r\n"),
        "/cgi-bin/devVideoAnalyse.cgi?action=getcaps&channel=1": R(200, "caps.SupportedRules=CrossLineDetection,NumberStat\r\n"),
        "/cgi-bin/configManager.cgi?action=getConfig&name=VideoAnalyseRule": R(200, "\r\n".join([
            "table.VideoAnalyseRule[0][0].Type=CrossLineDetection", "table.VideoAnalyseRule[0][0].Enable=true",
            "table.VideoAnalyseRule[0][1].Type=NumberStat", "table.VideoAnalyseRule[0][1].Enable=false"])),
        "/cgi-bin/configManager.cgi?action=getConfig&name=SmartMotionDetect": R(200, "table.SmartMotionDetect[0].Enable=true\r\n"),
    }
    asked = []
    a = camera_api.probe("10.0.0.5", "admin", "pw", get=lambda p: (asked.append(p), answers[p])[1])
    assert a["status"] == "ok" and a["model"] == "IPC-HFW3449T1-AS-PV" and a["firmware"].startswith("2.840")
    assert a["can"] == ["CrossLineDetection", "NumberStat"] and a["on"] == ["CrossLineDetection"] and a["human_motion"] is True
    asked.clear()
    a = camera_api.probe("10.0.0.5", "admin", "bad", get=lambda p: (asked.append(p), R(401))[1])
    assert a["status"].startswith("login refused") and len(asked) == 1
    assert camera_api.login_of("rtsp://ad%40m:p%3Aw@10.0.0.5:554/cam/realmonitor?channel=1&subtype=0") == ("10.0.0.5", "ad@m", "p:w")
    assert camera_api.login_of("rtsp://10.0.0.6/Streaming/Channels/101") is None
