"""Send alerts with their picture to WhatsApp or Telegram.

WhatsApp works through a gateway that links a normal WhatsApp number by scanning a QR code:
  - Green-API   (green-api.com):  instance id + API token
  - UltraMsg    (ultramsg.com):   instance id + token
Telegram works with a bot token (from @BotFather) and the chat id.
Webhook (e.g. an n8n flow): every alert is POSTed as JSON with its picture in base64, so the flow
can forward it anywhere (WhatsApp, ERP, tasks...).
Messages go out from a background thread so a slow network never delays the cameras.
"""
from __future__ import annotations

import base64
import logging
import time
import queue
import re
import threading

import httpx

log = logging.getLogger("cctv.notify")

PROVIDERS = ("", "greenapi", "ultramsg", "telegram", "webhook")


def whatsapp_number(raw: str, country: str = "964") -> str:
    """'0770 123 4567' or '+964 770...' -> '9647701234567'."""
    d = re.sub(r"\D", "", raw or "")
    if d.startswith("00"):
        d = d[2:]
    elif d.startswith("0"):
        d = country + d[1:]
    return d


class Notifier:
    def __init__(self, cfg: dict, client: httpx.Client | None = None):
        self.cfg = cfg
        self._client = client
        self._q: queue.Queue = queue.Queue(maxsize=50)
        self.last_error = ""
        self.sent = 0
        threading.Thread(target=self._worker, daemon=True, name="notify").start()

    @property
    def n(self) -> dict:
        return self.cfg["notify"]

    @property
    def enabled(self) -> bool:
        n = self.n
        if n.get("provider") == "webhook":
            return str(n.get("api_url") or "").startswith(("http://", "https://"))
        return bool(n.get("provider") and n.get("token") and n.get("to") and (n["provider"] == "telegram" or n.get("instance")))

    def wants(self, kind: str) -> bool:
        kinds = self.n.get("kinds") or []
        base = kind.split(":", 1)[0]
        return self.enabled and (not kinds or base in kinds)

    def queue_alert(self, kind: str, text: str, jpeg: bytes | None, meta: dict | None = None) -> None:
        if not self.wants(kind):
            return
        meta = dict(meta or {}, kind=kind.split(":", 1)[0], time=time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            self._q.put_nowait((text, jpeg if self.n.get("send_images", True) else None, meta))
        except queue.Full:
            log.warning("notification queue full, dropping: %s", text)

    def _worker(self) -> None:
        while True:
            text, jpeg, meta = self._q.get()
            try:
                self.send(text, jpeg, meta)
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)
                log.warning("notification failed: %s", e)

    # ---------- sending ----------
    def send(self, text: str, jpeg: bytes | None = None, meta: dict | None = None) -> None:
        """Send now (raises on failure). Used by the queue and by the "send test" button."""
        n = self.n
        site = self.cfg.get("site_name") or ""
        plain = text
        text = f"{site}\n{text}" if site else text
        client = self._client or httpx.Client(timeout=30)
        try:
            p = n.get("provider")
            if p == "greenapi":
                self._greenapi(client, n, text, jpeg)
            elif p == "ultramsg":
                self._ultramsg(client, n, text, jpeg)
            elif p == "telegram":
                self._telegram(client, n, text, jpeg)
            elif p == "webhook":
                self._webhook(client, n, plain, jpeg, dict(meta or {}, site=site))
            else:
                raise ValueError("اختر طريقة الإرسال أولاً.")
        except httpx.HTTPError as e:
            raise RuntimeError(f"تعذر الاتصال بخدمة الإرسال (تأكد من الإنترنت): {e}") from None
        finally:
            if client is not self._client:
                client.close()
        self.sent += 1
        self.last_error = ""

    @staticmethod
    def _check(r: httpx.Response, what: str) -> dict:
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code >= 400 or (isinstance(body, dict) and (body.get("error") or body.get("ok") is False)):
            detail = (body.get("description") or body.get("error") or body.get("message")) if isinstance(body, dict) else ""
            raise RuntimeError(f"{what}: {r.status_code} {detail or r.text[:200]}")
        return body if isinstance(body, dict) else {}

    def _greenapi(self, c: httpx.Client, n: dict, text: str, jpeg: bytes | None) -> None:
        base = (n.get("api_url") or "https://api.green-api.com").rstrip("/")
        inst, tok = str(n["instance"]).strip(), str(n["token"]).strip()
        chat = whatsapp_number(n["to"]) + "@c.us"
        if jpeg:
            r = c.post(f"{base}/waInstance{inst}/sendFileByUpload/{tok}",
                       data={"chatId": chat, "caption": text, "fileName": "alert.jpg"},
                       files={"file": ("alert.jpg", jpeg, "image/jpeg")})
        else:
            r = c.post(f"{base}/waInstance{inst}/sendMessage/{tok}", json={"chatId": chat, "message": text})
        self._check(r, "Green-API")

    def _ultramsg(self, c: httpx.Client, n: dict, text: str, jpeg: bytes | None) -> None:
        base = (n.get("api_url") or "https://api.ultramsg.com").rstrip("/")
        inst, tok = str(n["instance"]).strip(), str(n["token"]).strip()
        to = "+" + whatsapp_number(n["to"])
        if jpeg:
            r = c.post(f"{base}/{inst}/messages/image",
                       data={"token": tok, "to": to, "caption": text, "image": base64.b64encode(jpeg).decode()})
        else:
            r = c.post(f"{base}/{inst}/messages/chat", data={"token": tok, "to": to, "body": text})
        self._check(r, "UltraMsg")

    def _telegram(self, c: httpx.Client, n: dict, text: str, jpeg: bytes | None) -> None:
        base = (n.get("api_url") or "https://api.telegram.org").rstrip("/")
        tok, chat = str(n["token"]).strip(), str(n["to"]).strip()
        if jpeg:
            r = c.post(f"{base}/bot{tok}/sendPhoto", data={"chat_id": chat, "caption": text[:1000]},
                       files={"photo": ("alert.jpg", jpeg, "image/jpeg")})
        else:
            r = c.post(f"{base}/bot{tok}/sendMessage", json={"chat_id": chat, "text": text})
        self._check(r, "Telegram")

    def _webhook(self, c: httpx.Client, n: dict, text: str, jpeg: bytes | None, meta: dict) -> None:
        body = {"source": "cctv-analytics", "site": meta.get("site", ""), "kind": meta.get("kind", "test"),
                "message": text, "time": meta.get("time") or time.strftime("%Y-%m-%d %H:%M:%S"),
                "camera_id": meta.get("camera_id"), "camera": meta.get("camera"), "person": meta.get("person"),
                "to": n.get("to") or None, "has_image": jpeg is not None,
                "image_base64": base64.b64encode(jpeg).decode() if jpeg else None,
                "image_mime": "image/jpeg" if jpeg else None, "image_name": "alert.jpg" if jpeg else None}
        headers = {"X-CCTV-Token": str(n["token"]).strip()} if n.get("token") else {}
        self._check(c.post(str(n["api_url"]).strip(), json=body, headers=headers), "Webhook")
