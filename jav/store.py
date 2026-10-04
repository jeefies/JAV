"""SQLite (WAL) persistence: jobs, batches, assets, outputs, runtime_events."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, batch_id TEXT, client_ref TEXT,
  provider TEXT NOT NULL, workflow TEXT NOT NULL, runtime_profile TEXT NOT NULL,
  status TEXT NOT NULL, priority INTEGER DEFAULT 0,
  payload TEXT NOT NULL, assets TEXT NOT NULL DEFAULT '[]',
  cache_key TEXT,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  error TEXT, error_type TEXT, retry_count INTEGER DEFAULT 0,
  external_ref TEXT, cancel_requested INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_sched ON jobs(status, runtime_profile, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_cache ON jobs(cache_key, status);
CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);
CREATE TABLE IF NOT EXISTS batches(
  id TEXT PRIMARY KEY, client_ref TEXT, created_at TEXT NOT NULL, meta TEXT
);
CREATE TABLE IF NOT EXISTS assets(
  id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, kind TEXT NOT NULL,
  path TEXT NOT NULL, size INTEGER NOT NULL, mime TEXT, meta TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outputs(
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
  kind TEXT NOT NULL, asset_id TEXT, path TEXT, role TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
  profile_from TEXT, profile_to TEXT, reason TEXT,
  duration_ms INTEGER, ok INTEGER, detail TEXT
);
"""

TERMINAL = ("completed", "failed", "cancelled")
SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def cache_hash(data: dict) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class Store:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = str(db_path or config.DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # tools/*_smoke.py write to the live DB from a second process; without
        # a busy timeout a colliding WAL write raises SQLITE_BUSY instantly and
        # can strand a job mid-transition (default is 0ms).
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        """PRAGMA user_version migration ladder (idempotent, additive-only)."""
        ver = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if ver > SCHEMA_VERSION:
            raise RuntimeError(
                f"jav.db schema version {ver} is newer than supported "
                f"{SCHEMA_VERSION}; upgrade the JAV service before starting")
        if ver < 1:
            try:
                self._conn.execute(
                    "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # fresh DB already has the column via SCHEMA
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self):
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _rows(self, sql: str, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params=()):
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    # ---------- jobs ----------
    def create_job(self, *, provider: str, workflow: str, runtime_profile: str,
                   payload: dict, assets: list[str], batch_id: str | None = None,
                   client_ref: str | None = None, priority: int = 0,
                   cache_key: str | None = None, job_id: str | None = None,
                   status: str = "queued") -> dict:
        jid = job_id or new_id("job")
        self._exec(
            """INSERT INTO jobs(id,batch_id,client_ref,provider,workflow,runtime_profile,
               status,priority,payload,assets,cache_key,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (jid, batch_id, client_ref, provider, workflow, runtime_profile,
             status, priority, json.dumps(payload), json.dumps(assets),
             cache_key, now()),
        )
        return self.get_job(jid)

    def get_job(self, job_id: str) -> dict | None:
        row = self._one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if row:
            row["payload"] = json.loads(row["payload"])
            row["assets"] = json.loads(row["assets"])
        return row

    def update_job(self, job_id: str, **fields):
        if not fields:
            return
        cols = ",".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def set_status(self, job_id: str, status: str, **extra):
        fields = {"status": status, **extra}
        if status in ("starting_runtime", "running") and "started_at" not in fields:
            fields.setdefault("started_at", now())
        if status in TERMINAL:
            fields.setdefault("finished_at", now())
        self.update_job(job_id, **fields)

    def claim_next(self, profile: str) -> dict | None:
        """Atomically take highest-priority queued job for a profile."""
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM jobs WHERE status='queued' AND runtime_profile=?
                   ORDER BY priority DESC, created_at ASC LIMIT 1""",
                (profile,),
            ).fetchone()
            if not row:
                return None
            self._conn.execute(
                "UPDATE jobs SET status='starting_runtime', started_at=? WHERE id=?",
                (now(), row["id"]),
            )
            self._conn.commit()
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        d["assets"] = json.loads(d["assets"])
        d["status"] = "starting_runtime"
        return d

    def oldest_queued(self) -> list[dict]:
        rows = self._rows(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at ASC")
        for r in rows:
            r["payload"] = json.loads(r["payload"])
            r["assets"] = json.loads(r["assets"])
        return rows

    def queued_profiles(self) -> list[tuple]:
        return [
            (r["runtime_profile"], r["n"])
            for r in self._rows(
                "SELECT runtime_profile, COUNT(*) n FROM jobs WHERE status='queued' "
                "GROUP BY runtime_profile")
        ]

    def queued_count(self) -> int:
        r = self._one("SELECT COUNT(*) n FROM jobs WHERE status='queued'")
        return r["n"] if r else 0

    def queue_position(self, job_id: str) -> int | None:
        job = self.get_job(job_id)
        if not job or job["status"] != "queued":
            return None
        r = self._one(
            """SELECT COUNT(*) n FROM jobs WHERE status='queued'
               AND (priority > ? OR (priority = ? AND created_at < ?))""",
            (job["priority"], job["priority"], job["created_at"]),
        )
        return (r["n"] if r else 0) + 1

    def list_jobs(self, status=None, runtime_profile=None, batch_id=None,
                  provider=None, limit=100, offset=0) -> dict:
        where, params = [], []
        if status:
            vals = status.split(",")
            clause = "status=?" if len(vals) == 1 else \
                "status IN (%s)" % ",".join("?" * len(vals))
            where.append(clause)
            params.extend(vals)
        for col, val in (("runtime_profile", runtime_profile),
                         ("batch_id", batch_id), ("provider", provider)):
            if val:
                where.append(f"{col}=?")
                params.append(val)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = self._one(f"SELECT COUNT(*) n FROM jobs {clause}", tuple(params))["n"]
        rows = self._rows(
            f"SELECT * FROM jobs {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        for r in rows:
            r["payload"] = json.loads(r["payload"])
            r["assets"] = json.loads(r["assets"])
        return {"total": total, "jobs": rows}

    def find_cache_hit(self, cache_key: str) -> dict | None:
        # deterministic providers are zit (images) and cosyvoice (audio);
        # video kinds deliberately never hit (expensive to reuse, cheap-ish
        # to re-roll semantics differ)
        return self._one(
            """SELECT j.*, o.path AS output_path FROM jobs j
               JOIN outputs o ON o.job_id=j.id
               WHERE j.cache_key=? AND j.status='completed'
                 AND o.kind IN ('image','audio')
               ORDER BY j.created_at DESC LIMIT 1""",
            (cache_key,),
        )

    def recover_interrupted(self) -> list[str]:
        """Crash recovery: in-flight -> queued (JAV-DESIGN 4.1)."""
        rows = self._rows(
            "SELECT id FROM jobs WHERE status IN ('starting_runtime','running')")
        ids = [r["id"] for r in rows]
        if ids:
            q = ",".join("?" * len(ids))
            self._exec(
                f"UPDATE jobs SET status='queued', started_at=NULL WHERE id IN ({q})",
                tuple(ids))
        return ids

    def cancel_queued(self, job_id: str) -> bool:
        r = self._exec(
            "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
            (now(), job_id))
        return r.rowcount > 0

    def request_cancel(self, job_id: str) -> bool:
        """Persist a soft cancel for in-flight jobs; the scheduler observes
        the flag at its next checkpoint and finalizes to 'cancelled'."""
        r = self._exec(
            "UPDATE jobs SET cancel_requested=1 WHERE id=? "
            "AND status IN ('starting_runtime','running') AND cancel_requested=0",
            (job_id,))
        return r.rowcount > 0

    def cancel_requested(self, job_id: str) -> bool:
        r = self._one("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,))
        return bool(r and r["cancel_requested"])

    # ---------- batches ----------
    def create_batch(self, client_ref: str | None, meta: dict | None) -> dict:
        bid = new_id("bat")
        self._exec("INSERT INTO batches(id,client_ref,created_at,meta) VALUES(?,?,?,?)",
                   (bid, client_ref, now(), json.dumps(meta or {})))
        return {"id": bid, "client_ref": client_ref, "created_at": now()}

    def get_batch(self, batch_id: str) -> dict | None:
        b = self._one("SELECT * FROM batches WHERE id=?", (batch_id,))
        if not b:
            return None
        counts = {}
        for r in self._rows("SELECT status, COUNT(*) n FROM jobs WHERE batch_id=? GROUP BY status",
                            (batch_id,)):
            counts[r["status"]] = r["n"]
        b["counts"] = counts
        b["jobs"] = [j["id"] for j in self._rows(
            "SELECT id FROM jobs WHERE batch_id=? ORDER BY created_at", (batch_id,))]
        return b

    # ---------- assets / outputs ----------
    def put_asset(self, *, sha256: str, kind: str, path: str, size: int,
                  mime: str | None = None, meta: dict | None = None,
                  asset_id: str | None = None) -> dict:
        existing = self._one("SELECT * FROM assets WHERE sha256=?", (sha256,))
        if existing:
            return existing
        aid = asset_id or f"asset_{sha256[:16]}"
        self._exec(
            """INSERT OR IGNORE INTO assets(id,sha256,kind,path,size,mime,meta,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (aid, sha256, kind, path, size, mime, json.dumps(meta or {}), now()))
        return self._one("SELECT * FROM assets WHERE id=?", (aid,))

    def get_asset(self, asset_id: str) -> dict | None:
        return self._one("SELECT * FROM assets WHERE id=?", (asset_id,))

    def add_output(self, job_id: str, kind: str, asset_id: str | None,
                   path: str, role: str = "main") -> dict:
        oid = new_id("out")
        self._exec(
            "INSERT INTO outputs(id,job_id,kind,asset_id,path,role,created_at) VALUES(?,?,?,?,?,?,?)",
            (oid, job_id, kind, asset_id, path, role, now()))
        return {"id": oid, "job_id": job_id, "kind": kind, "asset_id": asset_id,
                "path": path, "role": role}

    @staticmethod
    def _with_asset_meta(rows: list[dict]) -> list[dict]:
        for r in rows:
            meta = r.pop("asset_meta", None)
            try:
                r["asset_meta"] = json.loads(meta) if meta else {}
            except Exception:
                r["asset_meta"] = {}
        return rows

    def get_outputs(self, job_id: str) -> list[dict]:
        return self._with_asset_meta(self._rows(
            """SELECT o.*, a.meta AS asset_meta FROM outputs o
               LEFT JOIN assets a ON a.id=o.asset_id
               WHERE o.job_id=? ORDER BY o.created_at""", (job_id,)))

    def outputs_for(self, job_ids: list[str]) -> dict[str, list[dict]]:
        """Batch variant of get_outputs (avoids N+1 on list endpoints)."""
        if not job_ids:
            return {}
        out: dict[str, list[dict]] = {jid: [] for jid in job_ids}
        q = ",".join("?" * len(job_ids))
        rows = self._with_asset_meta(self._rows(
            f"""SELECT o.*, a.meta AS asset_meta FROM outputs o
                LEFT JOIN assets a ON a.id=o.asset_id
                WHERE o.job_id IN ({q}) ORDER BY o.created_at""",
            tuple(job_ids)))
        for r in rows:
            out.setdefault(r["job_id"], []).append(r)
        return out

    def queue_positions_for(self, job_ids: list[str]) -> dict[str, int]:
        """Batch queue positions: one scan of the queued set."""
        targets = set(job_ids)
        rows = self._rows(
            "SELECT id, priority, created_at FROM jobs WHERE status='queued'")
        by_id = {r["id"]: r for r in rows}
        out: dict[str, int] = {}
        for jid in job_ids:
            j = by_id.get(jid)
            if not j:
                continue
            ahead = sum(1 for r in rows
                        if r["priority"] > j["priority"]
                        or (r["priority"] == j["priority"] and r["created_at"] < j["created_at"]))
            out[jid] = ahead + 1
        return out

    def asset_references(self, asset_id: str) -> dict:
        """Active-job + output references that block asset deletion."""
        o = self._one("SELECT COUNT(*) n FROM outputs WHERE asset_id=?", (asset_id,))
        pattern = f'%"{asset_id}"%'
        q = ",".join("?" * len(TERMINAL))
        j = self._one(
            f"SELECT COUNT(*) n FROM jobs WHERE status NOT IN ({q}) "
            "AND (assets LIKE ? OR payload LIKE ?)",
            (*TERMINAL, pattern, pattern))
        return {"outputs": o["n"] if o else 0, "active_jobs": j["n"] if j else 0}

    # ---------- events ----------
    def log_event(self, profile_from: str | None, profile_to: str | None,
                  reason: str, duration_ms: int | None = None,
                  ok: bool = True, detail: str | None = None):
        self._exec(
            """INSERT INTO runtime_events(ts,profile_from,profile_to,reason,duration_ms,ok,detail)
               VALUES(?,?,?,?,?,?,?)""",
            (now(), profile_from, profile_to, reason, duration_ms, int(ok), detail))

    def recent_events(self, limit=20) -> list[dict]:
        return self._rows(
            "SELECT * FROM runtime_events ORDER BY id DESC LIMIT ?", (limit,))

    # ---------- runtime state persistence (crash sweep) ----------
    def save_runtime_state(self, pid: int, profile: str, start_time: int | None = None):
        (config.DATA_DIR / "runtime.state").write_text(json.dumps(
            {"pid": pid, "profile": profile, "start_time": start_time, "ts": now()}))

    def load_runtime_state(self) -> dict | None:
        p = config.DATA_DIR / "runtime.state"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def clear_runtime_state(self):
        try:
            (config.DATA_DIR / "runtime.state").unlink()
        except FileNotFoundError:
            pass
