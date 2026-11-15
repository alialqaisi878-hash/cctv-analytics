import io
import json
import zipfile

import pytest

from app import updater as up


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_parse_version():
    assert up.parse_version("v1.2.10") > up.parse_version("1.2.9")
    assert up.parse_version("1.0") == (1, 0, 0)


def test_apply_update_keeps_user_files(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "old_module.py").write_text("old")
    (tmp_path / "config.yaml").write_text("my cameras")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cctv.db").write_text("stats")
    (tmp_path / "requirements.txt").write_text("same")
    (tmp_path / "launch.bat").write_text("running launcher")
    data = _zip({"run.py": "new run", "app/__init__.py": "", "app/version.py": '__version__ = "1.1.0"',
                 "config.yaml": "SHOULD NOT OVERWRITE", "requirements.txt": "same",
                 "launch.bat": "new launcher"})
    up.apply_update_zip(data, tmp_path)
    assert (tmp_path / "run.py").read_text() == "new run"
    assert not (tmp_path / "app" / "old_module.py").exists()
    assert "1.1.0" in (tmp_path / "app" / "version.py").read_text()
    assert (tmp_path / "config.yaml").read_text() == "my cameras"
    assert (tmp_path / "data" / "cctv.db").read_text() == "stats"
    assert (tmp_path / "launch.bat").read_text() == "running launcher"


def test_rejects_bad_packages(tmp_path):
    with pytest.raises(RuntimeError):
        up.apply_update_zip(_zip({"hello.txt": "x"}), tmp_path)
    with pytest.raises(RuntimeError):
        up.apply_update_zip(_zip({"run.py": "", "app/x.py": "", "../evil.py": ""}), tmp_path)


def test_check_detects_newer_release(monkeypatch):
    rel = {"tag_name": "v9.0.0", "body": "notes",
           "assets": [{"name": up.ASSET_NAME, "url": "https://api.github.com/assets/1"}]}

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["auth"] = req.headers.get("Authorization")
        return R(json.dumps(rel).encode())

    monkeypatch.setattr(up.urllib.request, "urlopen", fake_urlopen)
    u = up.Updater({"updates": {"enabled": True, "repo": "o/r", "token": "tok"}})
    st = u.check()
    assert st["available"] and st["latest"] == "v9.0.0" and st["error"] is None
    assert seen["auth"] == "Bearer tok"
