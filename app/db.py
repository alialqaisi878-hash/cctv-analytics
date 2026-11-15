"""SQLite storage: counts, timings, alerts (with their picture) and face sightings.

Counts are anonymous: track IDs are short-lived numbers assigned by the tracker.
Names appear only for people the owner enrolled in face recognition (off by default).
"""
from __future__ import annotations

import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS crossings (
    id INTEGER PRIMARY KEY, ts REAL, day TEXT, hour INTEGER,
    camera_id TEXT, line_id TEXT, direction TEXT, track_id INTEGER);
CREATE INDEX IF NOT EXISTS ix_cross_day ON crossings(day);
CREATE TABLE IF NOT EXISTS zone_visits (
    id INTEGER PRIMARY KEY, enter_ts REAL, exit_ts REAL, dwell_s REAL, day TEXT, hour INTEGER,
    camera_id TEXT, zone_id TEXT, track_id INTEGER);
CREATE INDEX IF NOT EXISTS ix_zone_day ON zone_visits(day);
CREATE TABLE IF NOT EXISTS occupancy (
    id INTEGER PRIMARY KEY, ts REAL, day TEXT, hour INTEGER, camera_id TEXT, people INTEGER);
CREATE INDEX IF NOT EXISTS ix_occ_day ON occupancy(day);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY, ts REAL, day TEXT, camera_id TEXT, kind TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY, ts REAL, day TEXT, camera_id TEXT, person_id TEXT, name TEXT, score REAL);
CREATE INDEX IF NOT EXISTS ix_sight_day ON sightings(day);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY, ts REAL, day TEXT, kind TEXT, content TEXT);
"""


class Database:
    def __init__(self, path: str, tz: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tz = ZoneInfo(tz)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(alerts)")}
        for col, typ in (("image", "TEXT"), ("seen", "INTEGER DEFAULT 0"), ("person_id", "TEXT")):
            if col not in cols:   # databases of older versions
                self._conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} {typ}")
        self._conn.commit()
        self.snap_dir = Path(path).parent / "snapshots"

    # ---------- helpers ----------
    def local(self, ts: float) -> datetime:
        return datetime.fromtimestamp(ts, self.tz)

    def today(self) -> str:
        return self.local(time.time()).strftime("%Y-%m-%d")

    def _dh(self, ts: float) -> tuple[str, int]:
        d = self.local(ts)
        return d.strftime("%Y-%m-%d"), d.hour

    def _exec(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    def _query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # ---------- writes ----------
    def add_crossing(self, ts, camera_id, line_id, direction, track_id):
        day, hour = self._dh(ts)
        self._exec("INSERT INTO crossings(ts,day,hour,camera_id,line_id,direction,track_id) VALUES (?,?,?,?,?,?,?)",
                   (ts, day, hour, camera_id, line_id, direction, track_id))

    def add_visit(self, camera_id, zone_id, track_id, enter_ts, exit_ts):
        day, hour = self._dh(enter_ts)
        self._exec("INSERT INTO zone_visits(enter_ts,exit_ts,dwell_s,day,hour,camera_id,zone_id,track_id) "
                   "VALUES (?,?,?,?,?,?,?,?)",
                   (enter_ts, exit_ts, exit_ts - enter_ts, day, hour, camera_id, zone_id, track_id))

    def add_occupancy(self, ts, camera_id, people):
        day, hour = self._dh(ts)
        self._exec("INSERT INTO occupancy(ts,day,hour,camera_id,people) VALUES (?,?,?,?,?)",
                   (ts, day, hour, camera_id, people))

    def add_alert(self, ts, camera_id, kind, message, jpeg: bytes | None = None, person_id: str | None = None) -> int:
        """Store an alert; its picture (if any) goes to data/snapshots/<day>/ and is deleted with the alert."""
        day, _ = self._dh(ts)
        image = None
        if jpeg:
            folder = self.snap_dir / day
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{int(ts * 1000)}_{''.join(ch if ch.isalnum() else '_' for ch in str(camera_id))}.jpg"
            (folder / name).write_bytes(jpeg)
            image = f"{day}/{name}"
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO alerts(ts,day,camera_id,kind,message,image,seen,person_id) VALUES (?,?,?,?,?,?,0,?)",
                (ts, day, camera_id, kind, message, image, person_id))
            self._conn.commit()
            return cur.lastrowid

    def alert_image(self, alert_id: int) -> Path | None:
        rows = self._query("SELECT image FROM alerts WHERE id=?", (alert_id,))
        if not rows or not rows[0]["image"]:
            return None
        p = (self.snap_dir / rows[0]["image"]).resolve()
        return p if p.is_file() and self.snap_dir.resolve() in p.parents else None

    def mark_seen(self, ids: list[int] | None = None) -> None:
        if ids is None:
            self._exec("UPDATE alerts SET seen=1 WHERE seen=0")
        else:
            with self._lock:
                self._conn.executemany("UPDATE alerts SET seen=1 WHERE id=?", [(int(i),) for i in ids])
                self._conn.commit()

    def unseen_count(self) -> int:
        return self._query("SELECT COUNT(*) n FROM alerts WHERE seen=0")[0]["n"]

    def alerts_after(self, after_id: int, limit: int = 20) -> list[dict]:
        rows = self._query("SELECT * FROM alerts WHERE id>? ORDER BY id LIMIT ?", (int(after_id), limit))
        return [self._alert_row(r) for r in rows]

    def last_alert_id(self) -> int:
        return self._query("SELECT COALESCE(MAX(id),0) m FROM alerts")[0]["m"]

    def add_sighting(self, ts, camera_id, person_id, name, score):
        day, _ = self._dh(ts)
        self._exec("INSERT INTO sightings(ts,day,camera_id,person_id,name,score) VALUES (?,?,?,?,?,?)",
                   (ts, day, camera_id, person_id, name, round(float(score), 3)))

    def attendance(self, day: str) -> list[dict]:
        """First and last time each enrolled person was recognised on a day."""
        rows = self._query("SELECT person_id, name, MIN(ts) first, MAX(ts) last, COUNT(*) n, "
                           "GROUP_CONCAT(DISTINCT camera_id) cams FROM sightings WHERE day=? "
                           "GROUP BY person_id ORDER BY first", (day,))
        for r in rows:
            r["first"] = self.local(r["first"]).strftime("%H:%M")
            r["last"] = self.local(r["last"]).strftime("%H:%M")
            r["cameras"] = (r.pop("cams") or "").split(",")
        return rows

    def last_seen(self) -> dict[str, dict]:
        """Where and when each enrolled person was last recognised (the row with the newest time)."""
        rows = self._query("SELECT person_id, MAX(ts) ts, camera_id FROM sightings GROUP BY person_id")
        return {r["person_id"]: {"time": self.local(r["ts"]).strftime("%Y-%m-%d %H:%M"), "ts": r["ts"],
                                 "camera_id": r["camera_id"]} for r in rows}

    def last_sighting(self, person_id: str) -> dict | None:
        rows = self._query("SELECT camera_id, ts FROM sightings WHERE person_id=? ORDER BY ts DESC LIMIT 1", (person_id,))
        return rows[0] if rows else None

    def forget_person(self, person_id: str) -> None:
        self._exec("DELETE FROM sightings WHERE person_id=?", (person_id,))

    def add_report(self, day, kind, content):
        self._exec("INSERT INTO reports(ts,day,kind,content) VALUES (?,?,?,?)", (time.time(), day, kind, content))

    def purge_older_than(self, days: int) -> int:
        cutoff = (self.local(time.time()) - timedelta(days=days)).strftime("%Y-%m-%d")
        n = 0
        with self._lock:
            for t in ("crossings", "zone_visits", "occupancy", "alerts", "reports", "sightings"):
                n += self._conn.execute(f"DELETE FROM {t} WHERE day < ?", (cutoff,)).rowcount
            self._conn.commit()
        if self.snap_dir.is_dir():   # alert pictures are kept for the same number of days
            for folder in self.snap_dir.iterdir():
                if folder.is_dir() and folder.name < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
        return n

    # ---------- reads ----------
    def summary(self, day: str) -> dict:
        rows = self._query("SELECT camera_id, line_id, direction, COUNT(*) n FROM crossings WHERE day=? "
                           "GROUP BY camera_id, line_id, direction", (day,))
        per_line: dict[str, dict] = {}
        total_in = total_out = 0
        for r in rows:
            key = f"{r['camera_id']}/{r['line_id']}"
            per_line.setdefault(key, {"camera_id": r["camera_id"], "line_id": r["line_id"], "in": 0, "out": 0})
            per_line[key][r["direction"]] = r["n"]
            if r["direction"] == "in":
                total_in += r["n"]
            else:
                total_out += r["n"]
        hourly = self.hourly(day)
        busiest = max(hourly, key=lambda h: h["in"] + h["out"], default=None)
        peak_occ = self._query("SELECT MAX(people) m FROM occupancy WHERE day=?", (day,))[0]["m"] or 0
        first_last = self._query("SELECT MIN(ts) first, MAX(ts) last FROM crossings WHERE day=? AND direction='in'",
                                 (day,))[0]
        fmt = lambda ts: self.local(ts).strftime("%H:%M") if ts else None
        return {
            "day": day,
            "total_in": total_in,
            "total_out": total_out,
            "net_inside_estimate": max(total_in - total_out, 0),
            "per_line": list(per_line.values()),
            "busiest_hour": busiest["hour"] if busiest and (busiest["in"] + busiest["out"]) else None,
            "peak_people_in_view": peak_occ,
            "first_entry": fmt(first_last["first"]),
            "last_entry": fmt(first_last["last"]),
        }

    def hourly(self, day: str) -> list[dict]:
        rows = self._query("SELECT hour, direction, COUNT(*) n FROM crossings WHERE day=? GROUP BY hour, direction",
                           (day,))
        occ = {r["hour"]: r for r in self._query(
            "SELECT hour, ROUND(AVG(people),1) avg_people, MAX(people) max_people FROM occupancy WHERE day=? "
            "GROUP BY hour", (day,))}
        out = [{"hour": h, "in": 0, "out": 0, "avg_people": occ.get(h, {}).get("avg_people", 0) or 0,
                "max_people": occ.get(h, {}).get("max_people", 0) or 0} for h in range(24)]
        for r in rows:
            out[r["hour"]][r["direction"]] = r["n"]
        return out

    def daily_trend(self, days: int = 7) -> list[dict]:
        end = self.local(time.time()).date()
        res = []
        for i in range(days - 1, -1, -1):
            d = (end - timedelta(days=i)).strftime("%Y-%m-%d")
            r = self._query("SELECT SUM(direction='in') i, SUM(direction='out') o FROM crossings WHERE day=?", (d,))[0]
            res.append({"day": d, "in": r["i"] or 0, "out": r["o"] or 0})
        return res

    def zone_dwell(self, day: str) -> list[dict]:
        return self._query(
            "SELECT camera_id, zone_id, COUNT(*) visits, ROUND(AVG(dwell_s),1) avg_dwell_s, "
            "ROUND(MAX(dwell_s),1) max_dwell_s, ROUND(SUM(dwell_s)/60.0,1) total_minutes "
            "FROM zone_visits WHERE day=? GROUP BY camera_id, zone_id ORDER BY camera_id, zone_id", (day,))

    def zone_hourly(self, day: str) -> list[dict]:
        return self._query(
            "SELECT zone_id, hour, COUNT(*) visits, ROUND(SUM(dwell_s)/60.0,1) total_minutes "
            "FROM zone_visits WHERE day=? GROUP BY zone_id, hour ORDER BY zone_id, hour", (day,))

    def _alert_row(self, r: dict) -> dict:
        r["time"] = self.local(r["ts"]).strftime("%Y-%m-%d %H:%M")
        r["has_image"] = bool(r.pop("image", None))
        r["seen"] = bool(r.get("seen"))
        return r

    def alerts(self, day: str | None = None, limit: int = 50, kind: str | None = None) -> list[dict]:
        where, args = [], []
        if day:
            where.append("day=?"); args.append(day)
        if kind:
            where.append("kind LIKE ?"); args.append(kind + "%")
        sql = "SELECT * FROM alerts" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY ts DESC LIMIT ?"
        return [self._alert_row(r) for r in self._query(sql, (*args, limit))]

    def reports(self, limit: int = 20) -> list[dict]:
        rows = self._query("SELECT * FROM reports ORDER BY ts DESC LIMIT ?", (limit,))
        for r in rows:
            r["time"] = self.local(r["ts"]).strftime("%Y-%m-%d %H:%M")
        return rows

    def crossings_csv_rows(self, day: str) -> list[dict]:
        rows = self._query("SELECT ts, camera_id, line_id, direction FROM crossings WHERE day=? ORDER BY ts", (day,))
        for r in rows:
            r["time"] = self.local(r.pop("ts")).strftime("%Y-%m-%d %H:%M:%S")
        return rows
