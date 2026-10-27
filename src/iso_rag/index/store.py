"""索引儲存：單一 SQLite 檔（docs、chunks、FTS5、向量 blob）。

- 每次 build 寫一個新檔 data/index/<build_id>.sqlite，寫完才改 CURRENT 指標
  -> 建置失敗不會弄壞正在使用的索引；刪掉的文件自然消失。
- 向量以 float32 blob 存，載入時組成 numpy 矩陣，暴力內積（<50 份約 1~2 千段，<5ms）。
- FTS5 欄位放 textnorm.fts_text() 斷好的 token（bigram + 英數 + 代碼），bm25() 有 IDF。
- 只有這個模組知道儲存格式；要換 pgvector 只改這裡。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import INDEX_DIR
from . import textnorm

SCHEMA = """
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE docs (
  doc_no TEXT PRIMARY KEY, iso_no TEXT, iso_key TEXT, title TEXT, status TEXT,
  iso_system TEXT, doc_type TEXT, dept TEXT, effective_date TEXT, sys_version TEXT, doc_version TEXT,
  pdf_path TEXT, pdf_sha TEXT, pages INTEGER, meta_json TEXT);
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY, doc_no TEXT, kind TEXT, page_start INTEGER, page_end INTEGER,
  section TEXT, clause TEXT, text TEXT, embed_text TEXT, vec BLOB,
  pdf_sha TEXT, pdf_path TEXT);          -- 段落出自哪個 PDF（一份文件可能掛好幾個 PDF：本文 + 附件表單）
CREATE INDEX chunks_doc ON chunks(doc_no);
CREATE VIRTUAL TABLE chunks_fts USING fts5(tokens, content='', tokenize='unicode61 remove_diacritics 0');
"""


def current_path() -> Path | None:
    ptr = INDEX_DIR / "CURRENT"
    if not ptr.exists():
        return None
    p = INDEX_DIR / ptr.read_text(encoding="utf-8").strip()
    return p if p.exists() else None


class Writer:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def manifest(self, **kv):
        self.db.executemany("INSERT OR REPLACE INTO manifest VALUES (?,?)",
                            [(k, json.dumps(v, ensure_ascii=False)) for k, v in kv.items()])

    def add_doc(self, meta: dict, *, pdf_path: str | None, pdf_sha: str | None, pages: int):
        self.db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            meta["doc_no"], meta.get("iso_no"), textnorm.code_key(meta.get("iso_no") or ""), meta.get("title"),
            meta.get("status"), meta.get("iso_system"), meta.get("doc_type"), meta.get("dept"),
            meta.get("effective_date"), meta.get("sys_version"), meta.get("doc_version"),
            pdf_path, pdf_sha, pages, json.dumps(meta, ensure_ascii=False)))

    def add_chunk(self, doc_no: str, kind: str, page_start: int, page_end: int, section: str, clause: str,
                  text: str, embed_text: str, vec: np.ndarray, pdf_sha: str | None = None,
                  pdf_path: str | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO chunks (doc_no, kind, page_start, page_end, section, clause, text, embed_text, vec,"
            " pdf_sha, pdf_path) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (doc_no, kind, page_start, page_end, section, clause, text, embed_text,
             vec.astype(np.float32).tobytes(), pdf_sha, pdf_path))
        cid = cur.lastrowid
        self.db.execute("INSERT INTO chunks_fts (rowid, tokens) VALUES (?, ?)", (cid, textnorm.fts_text(embed_text)))
        return cid

    def commit(self):
        """寫完、關檔，但還不切換（build 成功才 activate）。"""
        self.db.commit()
        self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        self.db.commit()
        self.db.close()

    def discard(self):
        """build 中途失敗：刪掉寫一半的檔，不留下空索引。"""
        try:
            self.db.close()
        finally:
            self.path.unlink(missing_ok=True)


def activate(path: Path) -> None:
    """原子切換 CURRENT：下一次查詢就換用這個索引。"""
    (INDEX_DIR / "CURRENT.tmp").write_text(path.name, encoding="utf-8")
    (INDEX_DIR / "CURRENT.tmp").replace(INDEX_DIR / "CURRENT")


@dataclass
class ChunkRow:
    id: int
    doc_no: str
    kind: str
    page_start: int
    page_end: int
    section: str
    clause: str
    text: str
    pdf_sha: str | None = None
    pdf_path: str | None = None


class Reader:
    """唯讀；整份載入記憶體（向量矩陣 + 段落 + 文件屬性）。"""

    def __init__(self, path: Path | None = None):
        self.path = path or current_path()
        if not self.path:
            raise FileNotFoundError("沒有可用的索引（data/index/CURRENT），請先執行 build")
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self.manifest = {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM manifest")}
        cols = [c[1] for c in self.db.execute("PRAGMA table_info(docs)")]
        self.docs = {}
        for row in self.db.execute("SELECT * FROM docs"):
            d = dict(zip(cols, row))
            d["meta"] = json.loads(d.pop("meta_json"))
            self.docs[d["doc_no"]] = d
        rows = self.db.execute("SELECT id, doc_no, kind, page_start, page_end, section, clause, text, vec,"
                               " pdf_sha, pdf_path FROM chunks ORDER BY id").fetchall()
        self.chunks = {r[0]: ChunkRow(*r[:8], *r[9:]) for r in rows}
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.mat = np.vstack([np.frombuffer(r[8], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1024), np.float32)

    def close(self):
        self.db.close()

    def dense(self, qvec: np.ndarray, k: int, allow: set[int] | None = None) -> list[tuple[int, float]]:
        sims = self.mat @ qvec
        order = np.argsort(-sims)
        out = []
        for i in order:
            cid = int(self.ids[i])
            if allow is None or cid in allow:
                out.append((cid, float(sims[i])))
                if len(out) >= k:
                    break
        return out

    def bm25(self, query: str, k: int, allow: set[int] | None = None) -> list[tuple[int, float]]:
        toks = sorted({t for t in textnorm.fts_text(query).split() if t})
        if not toks:
            return []
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in toks)
        # 有 allow 時先取夠多再篩：只取全庫前 k*4 名，範圍小的文件段落會被截掉
        limit = k * 4 if allow is None else max(k * 4, 3000)
        rows = self.db.execute("SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? "
                               "ORDER BY bm25(chunks_fts) LIMIT ?", (match, limit)).fetchall()
        out = [(int(r), -float(s)) for r, s in rows if allow is None or int(r) in allow]
        return out[:k]
