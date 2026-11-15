"""AI analyst agent: answers questions and writes reports from the aggregate database.

Uses the Claude API (tool runner) when ANTHROPIC_API_KEY is set. Without a key it
falls back to a built-in rule-based report so the app still works offline.
The agent only ever sees aggregate numbers - never video frames or identities.
"""
from __future__ import annotations

import json
import logging
import os

from .db import Database

log = logging.getLogger("cctv.agent")

SYSTEM_PROMPT = """You are the analytics assistant for a workplace people-counting system at "{site}".
You read aggregate, anonymous statistics through your tools: counts of people entering and leaving
through doors (virtual lines), how many people were visible per hour, time spent in named zones,
and alerts (after-hours presence, crowding).

How to work:
- Always call the tools to get real numbers before answering; never invent figures.
- Write in {lang_name}, clearly and practically, for a business owner/manager. Use short headings and
  bullet points. Give concrete operational suggestions (staffing at peak hours, queue handling,
  opening hours, energy use, safety) that follow from the data.
- Track IDs are temporary anonymous numbers from the tracker. They reset often and do not identify
  anyone. Do not try to identify, single out, or profile individual employees, and do not draw
  conclusions about a specific person's performance. If asked to, explain politely that the system
  only measures aggregate activity, and offer the aggregate view instead.
- Be honest about data quality: counts from cameras can be off by a few percent (occlusion, groups
  walking together). Point out anything that looks like a counting problem, e.g. many more "in" than
  "out" at day end, which usually means a line needs adjusting.
- Today is {today} (timezone {tz}). Working hours: {wh}.
"""


def _lang_name(code: str) -> str:
    return {"ar": "Arabic", "en": "English"}.get(code, code)


class Analyst:
    def __init__(self, db: Database, cfg: dict, client=None):
        self.db, self.cfg = db, cfg
        self._client = client  # injectable for tests

    @property
    def ai_enabled(self) -> bool:
        return self._client is not None or bool(os.environ.get("ANTHROPIC_API_KEY")
                                                 or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

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
        def get_setup() -> str:
            """Camera, door-line and zone names configured in the system, plus working hours."""
            cams = [{"id": c["id"], "name": c["name"],
                     "lines": [{"id": l["id"], "name": l["name"]} for l in c["lines"]],
                     "zones": [{"id": z["id"], "name": z["name"]} for z in c["zones"]]}
                    for c in cfg["cameras"]]
            return json.dumps({"cameras": cams, "working_hours": cfg["working_hours"]}, ensure_ascii=False)

        return [get_day_summary, get_hourly, get_daily_trend, get_zone_activity, get_alerts, get_setup]

    # ---------- public ----------
    def ask(self, question: str) -> str:
        if not self.ai_enabled:
            return ("الوكيل الذكي غير مفعّل: ضع مفتاح ANTHROPIC_API_KEY ثم أعد تشغيل البرنامج.\n\n"
                    + self.fallback_report(self.db.today()))
        import anthropic

        wh = self.cfg["working_hours"]
        system = SYSTEM_PROMPT.format(site=self.cfg["site_name"], lang_name=_lang_name(self.cfg["agent"]["language"]),
                                      today=self.db.today(), tz=self.cfg["timezone"],
                                      wh=f"{wh['start']}-{wh['end']}, weekdays {wh['days']} (Mon=0)")
        client = self._client or anthropic.Anthropic()
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
            return "مفتاح Claude API غير صحيح. تأكد من ANTHROPIC_API_KEY."
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

    def daily_report(self, day: str | None = None) -> str:
        day = day or self.db.today()
        if self.ai_enabled:
            text = self.ask(
                f"اكتب التقرير اليومي ليوم {day}: ملخص الدخول والخروج لكل باب، ساعات الذروة، "
                f"نشاط المناطق ومدة البقاء، التنبيهات، مقارنة مع الأيام السبعة الماضية، "
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
        if alerts:
            lines.append(f"\n## التنبيهات ({len(alerts)})")
            lines += [f"- {a['time']}: {a['message']}" for a in alerts[:10]]
        if s["total_in"] and abs(s["total_in"] - s["total_out"]) > max(5, 0.15 * s["total_in"]):
            lines.append("\n> ملاحظة: فرق كبير بين الدخول والخروج — قد يحتاج خط العدّ لتعديل مكانه.")
        return "\n".join(lines)
