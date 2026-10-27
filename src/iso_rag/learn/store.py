"""自主學習的狀態：data/learn/learn.sqlite（單檔，跟索引一樣好備份、好交接）。

  obs       每頁每個 OCR 策略的觀測（品質估計、秒數、特徵、文字）—— 學習的原始資料，只增不改
  profiles  版型（成員頁指紋、頁數、出現過的文件）
  policy    版型 × 策略的累計獎勵（bandit 統計，線上更新）
  gold      標準答案頁：文字層（自動）或人工確認/校正（learn accept）
  selfsup   自監督樣本：文字層頁 → 模擬掃描劣化 → 各策略 OCR 的輸出與正確率
  models    品質模型版本（candidate / active / retired）+ 評測指標與閘門結果
  review    待人工確認的頁（品質低、新版型）
  cond      自監督樣本那張（劣化後）影像的掃描狀況特徵（learn.context，情境模型的訓練資料）
  meta      雜項設定（例：紅章樣本版本）

原則：學到的模型先是 candidate，通過評測閘門才 promote；原始 OCR 快取不改寫，人工校正另外存（review.py）。
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import zlib
from pathlib import Path

from ..config import DATA

LEARN_DIR = DATA / "learn"
DB_PATH = LEARN_DIR / "learn.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS obs (
  sha TEXT, page INTEGER, variant TEXT, origin TEXT, profile TEXT, quality REAL, seconds REAL,
  chars INTEGER, text_sha TEXT, feats TEXT, text BLOB, chosen INTEGER, ts TEXT,
  PRIMARY KEY (sha, page, variant, origin));
CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY, exemplars BLOB, n INTEGER, docs TEXT, created TEXT, updated TEXT, origin TEXT);
CREATE TABLE IF NOT EXISTS policy (
  profile TEXT, variant TEXT, n INTEGER, reward REAL, PRIMARY KEY (profile, variant));
CREATE TABLE IF NOT EXISTS gold (
  sha TEXT, page INTEGER, origin TEXT, doc TEXT, text TEXT, ts TEXT, PRIMARY KEY (sha, page, origin));
CREATE TABLE IF NOT EXISTS selfsup (
  sha TEXT, page INTEGER, aug TEXT, variant TEXT, acc REAL, seconds REAL, feats TEXT, text BLOB, ts TEXT,
  PRIMARY KEY (sha, page, aug, variant));
CREATE TABLE IF NOT EXISTS models (
  version TEXT PRIMARY KEY, status TEXT, created TEXT, params BLOB, metrics TEXT, gate TEXT);
CREATE TABLE IF NOT EXISTS cond (
  sha TEXT, page INTEGER, aug TEXT, feats TEXT, PRIMARY KEY (sha, page, aug));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS review (
  sha TEXT, page INTEGER, doc TEXT, pdf TEXT, reason TEXT, quality REAL, status TEXT, ts TEXT,
  PRIMARY KEY (sha, page));
"""


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def pack(text: str) -> bytes:
    return zlib.compress((text or "").encode("utf-8"))


def unpack(blob: bytes | None) -> str:
    return zlib.decompress(blob).decode("utf-8") if blob else ""


class Store:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=30)      # build 與 learn 指令可能同時寫
        self.db.execute("PRAGMA journal_mode=WAL")       # 讀寫不互卡；寫入後立刻 commit，不要在 OCR 期間握鎖
        self.db.executescript(SCHEMA)

    def commit(self):
        self.db.commit()

    # ------------------------------------------------------------ 觀測
    def obs_put(self, *, sha, page, variant, origin, profile, quality, seconds, chars, text_sha, feats, text,
                chosen):
        self.db.execute("INSERT OR REPLACE INTO obs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            sha, page, variant, origin, profile, quality, seconds, chars, text_sha,
            json.dumps(feats), pack(text), int(chosen), now()))

    def obs_rows(self, where: str = "1=1", args: tuple = ()) -> list[dict]:
        cur = self.db.execute(f"SELECT sha, page, variant, origin, profile, quality, seconds, chars, feats, text, "
                              f"chosen FROM obs WHERE {where}", args)
        return [{"sha": r[0], "page": r[1], "variant": r[2], "origin": r[3], "profile": r[4], "quality": r[5],
                 "seconds": r[6], "chars": r[7], "feats": json.loads(r[8]), "text": unpack(r[9]),
                 "chosen": bool(r[10])} for r in cur]

    # ------------------------------------------------------------ 版型
    def profiles_rows(self) -> list[tuple]:
        return self.db.execute("SELECT id, exemplars, n, docs, created, updated, origin FROM profiles "
                               "ORDER BY id").fetchall()

    def profile_put(self, pid, exemplars: bytes, n: int, docs: list[str], created: str, origin: str):
        self.db.execute("INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?,?,?)",
                        (pid, exemplars, n, json.dumps(docs, ensure_ascii=False), created, now(), origin))

    # ------------------------------------------------------------ 策略統計
    def policy_rows(self) -> dict[tuple[str, str], tuple[int, float]]:
        return {(p, v): (n, r) for p, v, n, r in self.db.execute("SELECT profile, variant, n, reward FROM policy")}

    def policy_add(self, profile: str, variant: str, reward: float):
        self.db.execute("INSERT INTO policy VALUES (?,?,1,?) ON CONFLICT(profile, variant) DO UPDATE SET "
                        "n = n + 1, reward = reward + excluded.reward", (profile, variant, reward))

    # ------------------------------------------------------------ 標準答案
    def gold_put(self, sha, page, origin, doc, text):
        self.db.execute("INSERT OR REPLACE INTO gold VALUES (?,?,?,?,?,?)", (sha, page, origin, doc, text, now()))

    def gold_rows(self, origin: str | None = None) -> list[dict]:
        q = "SELECT sha, page, origin, doc, text FROM gold" + (" WHERE origin = ?" if origin else "")
        return [{"sha": r[0], "page": r[1], "origin": r[2], "doc": r[3], "text": r[4]}
                for r in self.db.execute(q, (origin,) if origin else ())]

    # ------------------------------------------------------------ 自監督樣本
    def selfsup_has(self, sha, page, aug, variant) -> bool:
        return self.db.execute("SELECT 1 FROM selfsup WHERE sha=? AND page=? AND aug=? AND variant=?",
                               (sha, page, aug, variant)).fetchone() is not None

    def selfsup_put(self, sha, page, aug, variant, acc, seconds, feats, text):
        self.db.execute("INSERT OR REPLACE INTO selfsup VALUES (?,?,?,?,?,?,?,?,?)",
                        (sha, page, aug, variant, acc, seconds, json.dumps(feats), pack(text), now()))

    def selfsup_rows(self) -> list[dict]:
        return [{"sha": r[0], "page": r[1], "aug": r[2], "variant": r[3], "acc": r[4], "seconds": r[5],
                 "feats": json.loads(r[6]), "text": unpack(r[7])}
                for r in self.db.execute("SELECT sha, page, aug, variant, acc, seconds, feats, text FROM selfsup")]

    def cond_put(self, sha, page, aug, feats: dict):
        self.db.execute("INSERT OR REPLACE INTO cond VALUES (?,?,?,?)", (sha, page, aug, json.dumps(feats)))

    def cond_rows(self) -> dict[tuple, dict]:
        return {(r[0], r[1], r[2]): json.loads(r[3]) for r in self.db.execute("SELECT sha, page, aug, feats FROM cond")}

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r[0] if r else default

    def meta_put(self, key: str, value: str):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    # ------------------------------------------------------------ 模型版本
    def model_put(self, version, status, params: bytes, metrics: dict, gate: dict):
        self.db.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?)", (
            version, status, now(), params, json.dumps(metrics, ensure_ascii=False),
            json.dumps(gate, ensure_ascii=False)))

    def model_rows(self) -> list[dict]:
        return [{"version": r[0], "status": r[1], "created": r[2], "metrics": json.loads(r[3] or "{}"),
                 "gate": json.loads(r[4] or "{}")}
                for r in self.db.execute("SELECT version, status, created, metrics, gate FROM models "
                                         "ORDER BY version")]

    def model_params(self, version: str) -> bytes | None:
        r = self.db.execute("SELECT params FROM models WHERE version = ?", (version,)).fetchone()
        return r[0] if r else None

    def model_active(self) -> str | None:
        r = self.db.execute("SELECT version FROM models WHERE status = 'active'").fetchone()
        return r[0] if r else None

    def model_status(self, version: str, status: str):
        self.db.execute("UPDATE models SET status = ? WHERE version = ?", (status, version))

    # ------------------------------------------------------------ 待人工確認
    def review_put(self, sha, page, doc, pdf, reason, quality):
        """同一頁已經處理過（confirmed/corrected）就不再排入。"""
        row = self.db.execute("SELECT status, reason FROM review WHERE sha=? AND page=?", (sha, page)).fetchone()
        if row and row[0] != "open":
            return
        if row and reason not in row[1].split(","):
            reason = f"{row[1]},{reason}"
        self.db.execute("INSERT OR REPLACE INTO review VALUES (?,?,?,?,?,?,?,?)",
                        (sha, page, doc, pdf, reason, quality, "open", now()))

    def review_rows(self, status: str | None = "open") -> list[dict]:
        q = "SELECT sha, page, doc, pdf, reason, quality, status FROM review" + \
            (" WHERE status = ?" if status else "") + " ORDER BY quality"
        return [{"sha": r[0], "page": r[1], "doc": r[2], "pdf": r[3], "reason": r[4], "quality": r[5],
                 "status": r[6]} for r in self.db.execute(q, (status,) if status else ())]

    def review_set(self, sha, page, status):
        self.db.execute("UPDATE review SET status = ?, ts = ? WHERE sha = ? AND page = ?", (status, now(), sha, page))
