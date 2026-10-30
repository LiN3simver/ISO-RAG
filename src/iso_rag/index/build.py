"""PDF 資料夾 -> 檢索索引（單一 SQLite 檔），寫完才原子切換 CURRENT。

- 一個 PDF 一份文件：屬性預設從檔名解析（開頭的文件編號、名稱）；data/pdf/metadata.jsonl 有對應紀錄就以它補齊
  （生效日期、ISO 體系、文件階層、狀態…）。內容相同（sha256 相同）的 PDF 只索引一次。
- 先擷取全部頁（快取），從整批文件學頁首頁尾樣板（ingest.template），再切段。
- 每份文件另有一段「屬性卡」，讓「生效日期／版本／體系」這類問題也能從檢索找到。
- embedding 依 (模型 revision, max_seq, 文字) 雜湊快取，重建只算新段落。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, INDEX_DIR, PDF_DIR
from ..ingest import chunker, template
from ..ingest.extract import OCR_TAG, extract_pdf, sha256
from . import embedder
from .store import Writer, activate

META_FILE = PDF_DIR / "metadata.jsonl"
# 檔名開頭的文件編號：「SOP-QA-001 樣品保存作業規範.pdf」「QP0101_文件管制程序.pdf」
_NO_RE = re.compile(r"^([A-Za-z]{1,6}(?:[-_]?[A-Za-z0-9]{1,6}){1,4})(?=[\s_]|$)")
CLAUSE_MIN = 8        # 條文級向量：短於這麼多字的條文（只有標題的空條）不算


class EmbedCache:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE_DIR / "embed.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v BLOB)")
        self.tag = f"{embedder.MODEL}@{embedder.REVISION}/{embedder.MAX_SEQ}"

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.tag}\n{text}".encode()).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [self.key(t) for t in texts]
        got = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = f"SELECT k, v FROM e WHERE k IN ({','.join('?' * len(part))})"
            got.update({k: np.frombuffer(v, dtype=np.float32) for k, v in self.db.execute(q, part)})
        todo = [i for i, k in enumerate(keys) if k not in got]
        if todo:
            vecs = embedder.get().encode([texts[i] for i in todo])
            self.db.executemany("INSERT OR REPLACE INTO e VALUES (?,?)",
                                [(keys[i], v.tobytes()) for i, v in zip(todo, vecs)])
            self.db.commit()
            got.update({keys[i]: v for i, v in zip(todo, vecs)})
        return np.vstack([got[k] for k in keys]), len(todo)


def doc_from_file(path: Path, extra: dict | None = None) -> dict:
    """檔名 -> 文件屬性；extra（metadata.jsonl 的同檔紀錄）覆寫預設值。"""
    m = _NO_RE.match(path.stem)
    no = m.group(1) if m and any(ch.isdigit() for ch in m.group(1)) else None
    return {"doc_no": path.stem, "iso_no": no, "title": path.stem, "files": [{"filename": path.name}], **(extra or {})}


def load_docs() -> list[tuple[dict, Path]]:
    """data/pdf/*.pdf -> [(屬性, 路徑)]。metadata.jsonl 每行一筆，以 "file" 欄位對應檔名。"""
    extra: dict[str, dict] = {}
    if META_FILE.exists():
        for line in META_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                extra[r.pop("file")] = r
    return [(doc_from_file(p, extra.get(p.name)), p) for p in sorted(PDF_DIR.glob("*.pdf"))]


def _code_version() -> str:
    """src/iso_rag 全部 .py 的雜湊：辨識是哪一版程式建的索引（不依賴 git）。"""
    h = hashlib.sha256()
    src = Path(__file__).resolve().parents[1]
    for p in sorted(src.rglob("*.py")):
        h.update(p.relative_to(src).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:12]


def _ocr_model() -> str | None:
    try:
        from ..learn.store import Store
        return Store().model_active()
    except sqlite3.Error:
        return None


def build(log=print) -> str:
    """寫新索引 -> 成功才切換 CURRENT；中途失敗刪掉寫一半的檔，服務繼續用原本的索引。"""
    t0 = time.monotonic()
    build_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    w = Writer(INDEX_DIR / f"{build_id}.sqlite")
    try:
        stats = _write(w, build_id, t0, log)
    except BaseException:
        w.discard()
        raise
    activate(w.path)
    log(f"build {build_id}: {stats}")
    return build_id


def _write(w: Writer, build_id: str, t0: float, log) -> dict:
    docs, seen = [], set()
    for d, p in load_docs():
        sha = sha256(p)
        if sha in seen:
            log(f"  略過重複內容的 PDF：{p.name}")
            continue
        seen.add(sha)
        docs.append((d, p, sha))

    # 第一輪：擷取全部頁（快取），從整批文件學頁首頁尾樣板
    pages_of = {sha: extract_pdf(p)[1] for _, p, sha in docs}
    tmpl = template.Templates.learn(template.by_sha(pages_of), chunker.is_heading_line)
    stats = {"docs": len(docs), "chunks": 0, "embedded_new": 0, "pages": 0, "templates": tmpl.stats()}
    log(f"樣板：{stats['templates']}；例：{tmpl.examples(12)}")

    cache = EmbedCache()
    for d, p, sha in docs:
        card = chunker.meta_card(d)
        rows = [("meta", 0, 0, "文件屬性", "", card, f"{chunker.doc_header(d, '文件屬性')}\n{card}")]
        for c in chunker.chunk_pages(pages_of[sha], d.get("category") or "", tmpl, sha):
            rows.append((c.kind, c.page_start, c.page_end, c.section, c.clause, c.text,
                         f"{chunker.doc_header(d, c.section)}\n{c.text}"))
        vecs, new = cache.encode([r[6] for r in rows])
        stats["embedded_new"] += new
        stats["pages"] += len(pages_of[sha])
        w.add_doc(d, pdf_path=str(p), pdf_sha=sha, pages=len(pages_of[sha]))
        # 條文級向量：本文段落依條文標題切開，每條另算一個向量（同一個快取），檢索時取 MaxSim（store.Reader.dense）
        blocks = [[(num, t) for num, _, t in chunker.clause_blocks(r[5], r[4]) if len(t.strip()) >= CLAUSE_MIN]
                  if r[0] == "body" else [] for r in rows]
        flat = [t for bs in blocks for _, t in bs]
        bvecs, bnew = cache.encode(flat) if flat else (np.zeros((0, 1)), 0)
        stats["embedded_new"] += bnew
        bi = 0
        for r, v, bs in zip(rows, vecs, blocks):
            cid = w.add_chunk(d["doc_no"], *r, vec=v,
                              pdf_sha=None if r[0] == "meta" else sha, pdf_path=None if r[0] == "meta" else str(p))
            stats["chunks"] += 1
            for num, _ in bs:
                w.add_clause(cid, num, bvecs[bi])
                bi += 1
            stats["clauses"] = stats.get("clauses", 0) + len(bs)
        log(f"  {d['doc_no']} chunks={len(rows):3} pages={len(pages_of[sha]):3} new_vec={new}")
    stats["seconds"] = round(time.monotonic() - t0, 1)
    w.manifest(build_id=build_id, built_at=dt.datetime.now().isoformat(timespec="seconds"),
               embed_model=embedder.MODEL, embed_revision=embedder.REVISION, embed_max_seq=embedder.MAX_SEQ,
               ocr=OCR_TAG, stats=stats, code=_code_version(), ocr_model=_ocr_model())
    w.commit()
    return stats
