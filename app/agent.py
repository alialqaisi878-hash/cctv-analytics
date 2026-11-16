"""AI analyst agent: answers questions and writes reports from the aggregate database.

Works with a Claude API key (sk-ant-..., tool runner) or an OpenAI API key (sk-..., function calling);
the kind of key is detected automatically. Without a key it falls back to a built-in report.
The agent sees numbers and, when face recognition is on, the arrival/leaving times of enrolled
people - never video frames.
"""
from __future__ import annotations

import json
import logging
import os

import httpx

from .db import Database

log = logging.getLogger("cctv.agent")

SYSTEM_PROMPT = """You are the analytics assistant for a workplace people-counting system at "{site}".
You read aggregate, anonymous statistics through your tools: counts of people entering and leaving
through doors (virtual lines), how many people were visible per hour, time spent in named zones,
and alerts (after-hours presence, crowding, cameras offline, recognised people). If the owner turned on
face recognition, get_attendance gives the first/last time each enrolled person was recognised.

How to work:
- Always call the tools to get real numbers before answering; never invent figures.
- Write in {lang_name}, clearly and practically, for a business owner/manager. Use short headings and
  bullet points. Give concrete operational suggestions (staffing at peak hours, queue handling,
  opening hours, energy use, safety) that follow from the data.
- Track IDs are temporary anonymous numbers from the tracker; they do not identify anyone.
- Attendance times of enrolled people are facts you may report (arrived / left / not seen). Do not
  judge a person's character or performance from them, and mention that a person can be missed if
  their face was not visible to a camera.
- Be honest about data quality: counts from cameras can be off by a few percent (occlusion, groups
  walking together). Point out anything that looks like a counting problem, e.g. many more "in" than
  "out" at day end, which usually means a line needs adjusting.
- Today is {today} (timezone {tz}). Working hours: {wh}.
"""


def _lang_name(code: str) -> str:
    return {"ar": "Arabic", "en": "English"}.get(code, code)


OPENAI_PREFERRED = ("gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o")


class Analyst:
    def __init__(self, db: Database, cfg: dict, client=None, openai_http: httpx.Client | None = None):
        self.db, self.cfg = db, cfg
        self._client = client  # injectable for tests
        self._openai_http = openai_http
        self.locations = None   # set by the server: () -> where each enrolled person was last seen
        self._openai_model_cache: str | None = None

    @property
    def provider(self) -> str | None:
        """'claude', 'openai' or None (no key)."""
        a = self.cfg["agent"]
        if self._client is not None:
            return "claude"
        key = (a.get("api_key") or "").strip()
        if a.get("provider") in ("claude", "openai") and key:
            return a["provider"]
        if key:
            return "claude" if key.startswith("sk-ant-") else "openai"
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return "claude"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return None

    @property
    def ai_enabled(self) -> bool:
        return self.provider is not None

    # ---------- tools ----------
    def _tools(self):
        from anthropic import beta_tool

        db, cfg = self.db, self.cfg

        @beta_tool
        def get_day_summary(day: str) -> str:
            """Totals for one day: people in/out per door line, busiest hour, peak people in view,
            first and last entry time.

            Args:
                day: Date as YYYY-MM-DD.
            """
            return json.dumps(db.summary(day), ensure_ascii=False)

        @beta_tool
        def get_hourly(day: str) -> str:
            """Per-hour entries, exits, and average/max number of people visible, for one day.

            Args:
                day: Date as YYYY-MM-DD.
            """
            rows = [h for h in db.hourly(day) if h["in"] or h["out"] or h["max_people"]]
            return json.dumps(rows, ensure_ascii=False)

        @beta_tool
        def get_daily_trend(days: int = 7) -> str:
            """Daily entry/exit totals for the last N days (including today).

            Args:
                days: Number of days, 1-90.
            """
            return json.dumps(db.daily_trend(max(1, min(int(days), 90))), ensure_ascii=False)

        @beta_tool
        def get_zone_activity(day: str) -> str:
            """Zone statistics for one day: visits, average/max dwell seconds, total minutes, and per-hour
            breakdown. Zones are areas like 'reception', 'work area', 'waiting area'.

            Args:
                day: Date as YYYY-MM-DD.
            """
            return json.dumps({"zones": db.zone_dwell(day), "by_hour": db.zone_hourly(day)}, ensure_ascii=False)

        @beta_tool
        def get_alerts(day: str) -> str:
            """Alerts for one day (after-hours presence, crowding).

            Args:
                day: Date as YYYY-MM-DD.
            """
            return json.dumps(db.alerts(day), ensure_ascii=False)

        @beta_tool
        def get_attendance(day: str) -> str:
            """Face recognition (if turned on): first and last time each enrolled person was recognised
            on a day, and on which cameras. Empty when face recognition is off or nobody was recognised.

            Args:
                day: Date as YYYY-MM-DD.
            """
            return json.dumps(db.attendance(day), ensure_ascii=False)

        @beta_tool
        def get_locations() -> str:
            """Face recognition (if turned on): where each enrolled employee is now, meaning the camera
            and floor-plan place that last recognised them, how many minutes ago, their job and ERP number.
            "present" is true when they were seen recently. Use it for questions like "where is Ahmed?"."""
            return json.dumps(self.locations() if self.locations else [], ensure_ascii=False)

        @beta_tool
        def get_setup() -> str:
            """Camera, door-line and zone names configured in the system, plus working hours."""
            cams = [{"id": c["id"], "name": c["name"],
                     "lines": [{"id": l["id"], "name": l["name"]} for l in c["lines"]],
                     "zones": [{"id": z["id"], "name": z["name"]} for z in c["zones"]]}
                    for c in cfg["cameras"]]
            return json.dumps({"cameras": cams, "working_hours": cfg["working_hours"]}, ensure_ascii=False)

        return [get_day_summary, get_hourly, get_daily_trend, get_zone_activity, get_alerts, get_attendance, get_locations, get_setup]

    def _openai_tools(self) -> tuple[list[dict], dict]:
        """The same tools in OpenAI's function-calling format, with their Python functions."""
        tools = self._tools()
        specs, funcs = [], {}
        for t in tools:
            p = t.to_dict() if hasattr(t, "to_dict") else {"name": t.name, "description": t.description,
                                                              "input_schema": t.input_schema}
            specs.append({"type": "function", "function": {"name": p["name"], "description": p.get("description", ""),
                                                           "parameters": p["input_schema"]}})
            funcs[p["name"]] = t
        return specs, funcs

    # ---------- public ----------
    def _system(self) -> str:
        wh = self.cfg["working_hours"]
        return SYSTEM_PROMPT.format(site=self.cfg["site_name"], lang_name=_lang_name(self.cfg["agent"]["language"]),
                                    today=self.db.today(), tz=self.cfg["timezone"],
                                    wh=f"{wh['start']}-{wh['end']}, weekdays {wh['days']} (Mon=0)")

    def ask(self, question: str) -> str:
        provider = self.provider
        if provider is None:
            return ("المساعد الذكي غير مفعّل: ضع مفتاح Claude أو OpenAI في صفحة الإعدادات.\n\n"
                    + self.fallback_report(self.db.today()))
        if provider == "openai":
            return self._ask_openai(question)
        import anthropic

        system = self._system()
        key = self.cfg["agent"].get("api_key")
        client = self._client or (anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic())
        try:
            runner = client.beta.messages.tool_runner(
                model=self.cfg["agent"]["model"],
                max_tokens=16000,
                system=system,
                tools=self._tools(),
                messages=[{"role": "user", "content": question}],
                output_config={"effort": "medium"},
                # if the main model declines, let the API retry on a suitable fallback model
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                max_iterations=12,
            )
            final = runner.until_done()
        except anthropic.AuthenticationError:
            return "مفتاح Claude API غير صحيح. صحّحه في صفحة الإعدادات."
        except anthropic.RateLimitError:
            return "تم تجاوز حد الطلبات على Claude API، حاول بعد دقيقة."
        except anthropic.APIConnectionError:
            return "تعذر الاتصال بـ Claude API (تحقق من الإنترنت).\n\n" + self.fallback_report(self.db.today())
        except anthropic.APIStatusError as e:
            log.exception("Claude API error")
            return f"خطأ من Claude API ({e.status_code}): {e.message}"
        if final.stop_reason == "refusal":
            return "لم يتمكن الوكيل من الإجابة على هذا الطلب."
        return "\n".join(b.text for b in final.content if b.type == "text").strip()

    # ---------- OpenAI ----------
    def _openai(self) -> httpx.Client:
        key = (self.cfg["agent"].get("api_key") or os.environ.get("OPENAI_API_KEY") or "").strip()
        return self._openai_http or httpx.Client(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                                 headers={"Authorization": f"Bearer {key}"}, timeout=120)

    def _openai_model(self, http: httpx.Client) -> str:
        chosen = (self.cfg["agent"].get("openai_model") or "").strip()
        if chosen:
            return chosen
        if self._openai_model_cache:
            return self._openai_model_cache
        model = OPENAI_PREFERRED[0]
        try:
            r = http.get("/models")
            if r.status_code == 200:
                have = {m["id"] for m in r.json().get("data", [])}
                model = next((m for m in OPENAI_PREFERRED if m in have), None) or next(
                    (m for m in sorted(have, reverse=True) if m.startswith("gpt-") and "audio" not in m
                     and "realtime" not in m and "image" not in m and "transcribe" not in m and "tts" not in m), model)
        except httpx.HTTPError:
            pass
        self._openai_model_cache = model
        return model

    def _ask_openai(self, question: str) -> str:
        http = self._openai()
        try:
            specs, funcs = self._openai_tools()
            model = self._openai_model(http)
            messages = [{"role": "system", "content": self._system()}, {"role": "user", "content": question}]
            for _ in range(12):
                r = http.post("/chat/completions", json={"model": model, "messages": messages, "tools": specs})
                if r.status_code == 401:
                    return "مفتاح OpenAI غير صحيح. صحّحه في صفحة الإعدادات."
                if r.status_code == 429:
                    return "تجاوزت حد الطلبات أو الرصيد في حساب OpenAI. تأكد من الرصيد وحاول بعد دقيقة."
                if r.status_code >= 400:
                    try:
                        msg = r.json().get("error", {}).get("message", r.text[:300])
                    except ValueError:
                        msg = r.text[:300]
                    log.error("OpenAI error %s: %s", r.status_code, msg)
                    return f"خطأ من OpenAI ({r.status_code}): {msg}"
                choice = r.json()["choices"][0]
                msg = choice["message"]
                calls = msg.get("tool_calls") or []
                if not calls:
                    return (msg.get("content") or "").strip() or "لم يرجع OpenAI أي جواب."
                messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
                for c in calls:
                    fn = funcs.get(c["function"]["name"])
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                        result = fn.call(args) if fn is not None and hasattr(fn, "call") else fn(**args) if fn else "unknown tool"
                    except Exception as e:  # noqa: BLE001
                        result = json.dumps({"error": str(e)})
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False, default=str)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            return "توقف المساعد قبل ما يكمل الجواب. جرّب سؤال أبسط."
        except httpx.HTTPError:
            return "تعذر الاتصال بـ OpenAI (تحقق من الإنترنت).\n\n" + self.fallback_report(self.db.today())
        finally:
            if http is not self._openai_http:
                http.close()

    def daily_report(self, day: str | None = None) -> str:
        day = day or self.db.today()
        if self.ai_enabled:
            text = self.ask(
                f"اكتب التقرير اليومي ليوم {day}: ملخص الدخول والخروج لكل باب، ساعات الذروة، "
                f"نشاط المناطق ومدة البقاء، التنبيهات، الحضور (إذا التعرف على الوجه مفعّل)، "
                f"مقارنة مع الأيام السبعة الماضية، "
                f"وثلاث توصيات عملية. اجعل التقرير مختصراً وواضحاً.")
        else:
            text = self.fallback_report(day)
        self.db.add_report(day, "daily", text)
        return text

    def fallback_report(self, day: str) -> str:
        s = self.db.summary(day)
        zones = self.db.zone_dwell(day)
        alerts = self.db.alerts(day)
        trend = self.db.daily_trend(7)
        avg_in = sum(t["in"] for t in trend[:-1]) / max(len(trend) - 1, 1)
        lines = [f"# تقرير يوم {day} — {self.cfg['site_name']}",
                 f"- عدد الداخلين: **{s['total_in']}**، عدد الخارجين: **{s['total_out']}**",
                 f"- تقدير الموجودين حالياً: {s['net_inside_estimate']}",
                 f"- أول دخول: {s['first_entry'] or '-'}، آخر دخول: {s['last_entry'] or '-'}"]
        if s["busiest_hour"] is not None:
            lines.append(f"- ساعة الذروة: {s['busiest_hour']:02d}:00")
        lines.append(f"- أعلى عدد أشخاص ظاهرين في نفس الوقت: {s['peak_people_in_view']}")
        if avg_in:
            diff = (s["total_in"] - avg_in) / avg_in * 100
            lines.append(f"- مقارنة بمعدل الأيام الستة السابقة ({avg_in:.0f}): {diff:+.0f}%")
        for pl in s["per_line"]:
            lines.append(f"  - {pl['camera_id']}/{pl['line_id']}: دخول {pl['in']}، خروج {pl['out']}")
        if zones:
            lines.append("\n## المناطق")
            for z in zones:
                lines.append(f"- {z['zone_id']}: {z['visits']} زيارة، متوسط البقاء {z['avg_dwell_s']} ثانية")
        att = self.db.attendance(day)
        if att:
            lines.append("\n## الحضور (التعرف على الوجه)")
            lines += [f"- {a['name']}: أول ظهور {a['first']}، آخر ظهور {a['last']}" for a in att]
        if alerts:
            lines.append(f"\n## التنبيهات ({len(alerts)})")
            lines += [f"- {a['time']}: {a['message']}" for a in alerts[:10]]
        if s["total_in"] and abs(s["total_in"] - s["total_out"]) > max(5, 0.15 * s["total_in"]):
            lines.append("\n> ملاحظة: فرق كبير بين الدخول والخروج — قد يحتاج خط العدّ لتعديل مكانه.")
        return "\n".join(lines)
