"""Find IP cameras on the local network.

Two methods, both read-only:
  1. ONVIF WS-Discovery: a multicast "who's there?" probe that most IP cameras answer.
  2. RTSP port scan of the local /24 subnet (ports 554 and 8554), confirmed with an RTSP OPTIONS request.

Run standalone:  python -m app.discovery
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor

WS_DISCOVERY_ADDR = ("239.255.255.250", 3702)
PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header><w:MessageID>uuid:{mid}</w:MessageID>
 <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
 <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""

# Common RTSP paths by vendor; the user picks one and adds credentials.
RTSP_TEMPLATES = {
    "Hikvision": "rtsp://USER:PASS@{ip}:554/Streaming/Channels/101",
    "Dahua / Imou": "rtsp://USER:PASS@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "Uniview": "rtsp://USER:PASS@{ip}:554/unicast/c1/s0/live",
    "TP-Link Tapo/VIGI": "rtsp://USER:PASS@{ip}:554/stream1",
    "Reolink": "rtsp://USER:PASS@{ip}:554/h264Preview_01_main",
    "Generic": "rtsp://USER:PASS@{ip}:554/",
}


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the LAN interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def onvif_probe(timeout: float = 3.0) -> list[dict]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)
    found: dict[str, dict] = {}
    try:
        sock.sendto(PROBE.format(mid=uuid.uuid4()).encode(), WS_DISCOVERY_ADDR)
        while True:
            try:
                data, (ip, _) = sock.recvfrom(65535)
            except socket.timeout:
                break
            text = data.decode(errors="ignore")
            xaddrs = re.findall(r"<[^>]*XAddrs>([^<]+)<", text)
            scopes = " ".join(re.findall(r"<[^>]*Scopes>([^<]+)<", text))
            name = re.search(r"onvif://www\.onvif\.org/name/([^\s]+)", scopes)
            hw = re.search(r"onvif://www\.onvif\.org/hardware/([^\s]+)", scopes)
            found[ip] = {
                "ip": ip, "method": "onvif",
                "onvif_url": xaddrs[0].split()[0] if xaddrs else None,
                "name": _unq(name.group(1)) if name else None,
                "model": _unq(hw.group(1)) if hw else None,
            }
    except OSError as e:
        return [{"error": f"ONVIF probe failed: {e}"}]
    finally:
        sock.close()
    return list(found.values())


def _unq(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s)


def _rtsp_check(ip: str, port: int, timeout: float) -> dict | None:
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(f"OPTIONS rtsp://{ip}:{port}/ RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: cctv-analytics\r\n\r\n".encode())
            reply = s.recv(1024).decode(errors="ignore")
            server = re.search(r"Server:\s*(.+)", reply)
            return {"ip": ip, "port": port, "method": "rtsp",
                    "rtsp_ok": reply.startswith("RTSP/1.0"),
                    "server": server.group(1).strip() if server else None}
    except OSError:
        return None


def rtsp_scan(subnet: str | None = None, ports=(554, 8554), timeout: float = 0.6, workers: int = 128) -> list[dict]:
    if subnet is None:
        subnet = str(ipaddress.ip_network(local_ip() + "/24", strict=False))
    net = ipaddress.ip_network(subnet, strict=False)
    if net.num_addresses > 1024:
        raise ValueError("subnet too large; scan a /22 or smaller")
    targets = [(str(ip), p) for ip in net.hosts() for p in ports]
    with ThreadPoolExecutor(workers) as ex:
        results = ex.map(lambda t: _rtsp_check(t[0], t[1], timeout), targets)
    return [r for r in results if r]


# Stream paths tried when a camera is added from the dashboard ({ch} = channel number, for recorders).
STREAM_PATHS = [
    ("Hikvision", "/Streaming/Channels/{ch}01"),
    ("Dahua / Imou", "/cam/realmonitor?channel={ch}&subtype=0"),
    ("Uniview", "/unicast/c{ch}/s0/live"),
    ("TP-Link Tapo/VIGI", "/stream1"),
    ("Reolink", "/h264Preview_{ch:02d}_main"),
    ("Generic", "/live"),
    ("Generic", "/h264"),
    ("Generic", "/11"),
    ("Generic", "/"),
]


def _rtsp_request(sock: socket.socket, uri: str, cseq: int, auth: str | None) -> tuple[int, str]:
    req = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: {cseq}\r\nUser-Agent: cctv-analytics\r\nAccept: application/sdp\r\n"
    if auth:
        req += f"Authorization: {auth}\r\n"
    sock.sendall((req + "\r\n").encode())
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 16384:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    head = data.decode(errors="ignore")
    m = re.match(r"RTSP/1\.\d\s+(\d{3})", head)
    return (int(m.group(1)) if m else 0), head


def _auth_header(head: str, uri: str, user: str, password: str) -> str | None:
    challenges = re.findall(r"WWW-Authenticate:\s*(.+)", head, re.I)
    digest = next((c for c in challenges if c.strip().lower().startswith("digest")), None)
    if digest:
        realm = re.search(r'realm="([^"]*)"', digest)
        nonce = re.search(r'nonce="([^"]*)"', digest)
        if not (realm and nonce):
            return None
        md5 = lambda x: hashlib.md5(x.encode()).hexdigest()  # noqa: E731  (RTSP digest auth is MD5 by spec)
        ha1 = md5(f"{user}:{realm.group(1)}:{password}")
        ha2 = md5(f"DESCRIBE:{uri}")
        resp = md5(f"{ha1}:{nonce.group(1)}:{ha2}")
        return (f'Digest username="{user}", realm="{realm.group(1)}", nonce="{nonce.group(1)}", '
                f'uri="{uri}", response="{resp}"')
    if any(c.strip().lower().startswith("basic") for c in challenges):
        import base64
        return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    return None


def rtsp_describe(ip: str, port: int, path: str, user: str, password: str, timeout: float = 4.0) -> int:
    """Ask the camera for a stream description. Returns the RTSP status (200 = stream exists and the
    login is right, 401/403 = wrong login, 404 = no such stream) or 0 when the camera did not answer."""
    uri = f"rtsp://{ip}:{port}{path}"
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.settimeout(timeout)
            code, head = _rtsp_request(s, uri, 1, None)
            if code != 401:
                return code
            auth = _auth_header(head, uri, user, password)
            if not auth:
                return 401
            try:
                code, _ = _rtsp_request(s, uri, 2, auth)
                if code:
                    return code
            except OSError:
                pass
        with socket.create_connection((ip, port), timeout=timeout) as s:   # camera closed the connection
            s.settimeout(timeout)
            return _rtsp_request(s, uri, 2, auth)[0]
    except OSError:
        return 0


def find_stream(ip: str, port: int, user: str, password: str, channel: int = 1) -> dict:
    """Find the working video address of a camera. Stops at the first wrong-login answer, so a
    mistyped password costs one attempt (cameras lock the account after several)."""
    from urllib.parse import quote
    answered = False
    for vendor, tpl in STREAM_PATHS:
        path = tpl.format(ch=channel)
        code = rtsp_describe(ip, port, path, user, password)
        if code == 200:
            cred = f"{quote(user, safe='')}:{quote(password, safe='')}@" if user else ""
            return {"status": "ok", "vendor": vendor, "url": f"rtsp://{cred}{ip}:{port}{path}"}
        if code in (401, 403):
            return {"status": "auth"}
        if code == 0 and not answered:
            if _rtsp_check(ip, port, 2.0) is None:
                return {"status": "unreachable"}
        answered = answered or code != 0
    return {"status": "not_found"}


def discover(subnet: str | None = None) -> dict:
    """Run both methods and merge results by IP."""
    onvif = onvif_probe()
    errors = [d["error"] for d in onvif if "error" in d]
    onvif = [d for d in onvif if "error" not in d]
    try:
        rtsp = rtsp_scan(subnet)
    except ValueError as e:
        rtsp, errors = [], errors + [str(e)]
    merged: dict[str, dict] = {}
    for d in onvif:
        merged[d["ip"]] = {**d, "rtsp_ports": []}
    for d in rtsp:
        m = merged.setdefault(d["ip"], {"ip": d["ip"], "method": "rtsp", "rtsp_ports": []})
        m["rtsp_ports"].append(d["port"])
        m.setdefault("server", d.get("server"))
        if m["method"] == "onvif":
            m["method"] = "onvif+rtsp"
    for m in merged.values():
        m["rtsp_url_examples"] = {k: v.format(ip=m["ip"]) for k, v in RTSP_TEMPLATES.items()}
    return {"local_ip": local_ip(), "subnet": subnet or "auto /24", "cameras": sorted(merged.values(),
            key=lambda d: tuple(int(x) for x in d["ip"].split("."))), "errors": errors}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(discover(sys.argv[1] if len(sys.argv) > 1 else None), ensure_ascii=False, indent=2))
