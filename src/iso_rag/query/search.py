"""混合檢索：BM25（FTS5）+ 向量（bge-m3）+ 文件代碼命中 + 原文片段，加權 RRF 融合。

原文片段通道：問題跟段落有 ≥ PHRASE_MIN 個字連續相同（正規化後）就依相同長度排名。
使用者貼表單名稱、條文原句時，常是 BM25 排第 1 但向量沒找到，RRF 融合後會被「兩邊都中等」的段落擠掉，
這個通道把逐字命中的段落拉回前面。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..index import embedder, textnorm
from ..index.store import ChunkRow, Reader
from .router import Analysis, filter_docs

RRF_K = 60
W_CODE, W_BM25, W_DENSE, W_PHRASE = 2.0, 1.0, 1.0, 2.0
W_SCOPE = 0.75            # 問題提到的體系／階層：加權偏好（BM25、向量各一個通道），不硬性排除
PHRASE_MIN = 8
PHRASE_LOCK = 12          # 問題跟焦點以外的文件有這麼長的原文相同 -> 不鎖定焦點文件
PER_DOC = 4
CONTEXT_CHARS = 6000


@dataclass
class Hit:
    chunk: ChunkRow
    score: float
    why: list[str]


def _rrf(channels: dict[str, tuple[float, list[int]]]) -> dict[int, tuple[float, list[str]]]:
    """加權 RRF：score = Σ w / (k + rank)。"""
    out: dict[int, tuple[float, list[str]]] = {}
    for name, (w, ids) in channels.items():
        for rank, cid in enumerate(ids):
            s, why = out.get(cid, (0.0, []))
            out[cid] = (s + w / (RRF_K + rank + 1), why + [f"{name}#{rank + 1}"])
    return out


FOCUS_TOP = 10      # 焦點文件要出現在一般檢索前幾名，才鎖定（第二道保險，防標題詞誤判）


def search(a: Analysis, idx: Reader, *, top_k: int = 6, cand: int = 40) -> list[Hit]:
    """問題明確指向某份文件（router 的 focus_docs）且該文件本來就排在前面時，
    只回傳那份文件的段落 —— 避免 LLM 把其他文件的相似規定混進答案
    （例：問 ORT 留樣保存期，卻帶出另一份文件的一般樣品保存期限）。"""
    hits = _search(a, idx, top_k=max(top_k, FOCUS_TOP), cand=cand)
    a.focused = False
    if a.focus_docs:
        focus = set(a.focus_docs)
        top_docs = [h.chunk.doc_no for h in hits[:FOCUS_TOP]]
        if _quotes_elsewhere(a.query, idx, hits[:FOCUS_TOP], focus):
            # 問題引用的原文在別份文件：標題詞（「不合格」「教育訓練」）只是主題相同，不鎖定
            return _quoted_only(a, idx, hits[:top_k])
        if any(d in focus for d in top_docs) and any(d not in focus for d in top_docs):
            allow = {cid for cid, c in idx.chunks.items() if c.doc_no in focus}
            hits = _search(a, idx, top_k=top_k, cand=cand, allow=allow, per_doc=top_k)
            a.focused = True
        elif top_docs and all(d in focus for d in top_docs):
            a.focused = True
    return _quoted_only(a, idx, hits[:top_k])


def _quoted_only(a: Analysis, idx: Reader, hits: list[Hit]) -> list[Hit]:
    """問題引用了一段原文（≥ PHRASE_LOCK 字：文件名稱、表單名稱、條文原句）：只留含這段原文的文件。
    其他文件只是主題相近、RRF 補滿 top_k 的低分段落，留著只會讓 LLM 看到沒問到的文件。
    問題點名的文件代碼照留。"""
    q = _flat(a.query)
    if len(q) < PHRASE_LOCK:
        return hits
    key = lambda h: idx.docs[h.chunk.doc_no]["iso_key"] or h.chunk.doc_no  # noqa: E731
    quoted = {key(h) for h in hits if _longest(q, idx, h.chunk.id) >= PHRASE_LOCK}
    if not quoted:
        return hits
    named = set(a.doc_nos)
    return [h for h in hits if key(h) in quoted or h.chunk.doc_no in named]


def quoted_span(query: str, idx: Reader, hits: list[Hit], min_len: int = PHRASE_LOCK) -> str:
    """問題逐字引用、而且原文真的出現在檢索結果裡的最長一段（_flat 後、≥ min_len 字）；沒有 -> ""。
    由資料判斷（問題跟原文比對），不看問法。"""
    q = _flat(query)
    if len(q) < min_len:
        return ""
    best = ""
    for h in hits:
        t = _flat_of(idx, h.chunk.id)
        m = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
        if m.size >= min_len and m.size > len(best):
            best = q[m.a:m.a + m.size]
    return best


def _quotes_elsewhere(query: str, idx: Reader, hits: list[Hit], focus: set[str]) -> bool:
    q = _flat(query)
    if len(q) < PHRASE_LOCK:
        return False
    best_in = best_out = 0
    for h in hits:
        t = _flat(h.chunk.text)
        n = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t)).size
        if h.chunk.doc_no in focus:
            best_in = max(best_in, n)
        else:
            best_out = max(best_out, n)
    return best_out >= PHRASE_LOCK and best_out > best_in


def _flat(s: str) -> str:
    return re.sub(r"[\s|]+", "", textnorm.norm(s))


def _phrase_rank(query: str, idx: Reader, cands: list[int]) -> list[int]:
    """候選段落中，跟問題有 ≥ PHRASE_MIN 字連續相同的，依相同長度排序。"""
    q = _flat(query)
    if len(q) < PHRASE_MIN:
        return []
    scored = []
    for cid in cands:
        n = _longest(q, idx, cid)
        if n >= PHRASE_MIN:
            scored.append((n, cid))
    return [cid for _, cid in sorted(scored, key=lambda x: -x[0])]


def _flat_of(idx: Reader, cid: int) -> str:
    cache = idx.__dict__.setdefault("_flat_cache", {})
    t = cache.get(cid)
    if t is None:
        t = cache[cid] = _flat(idx.chunks[cid].text)
    return t


def _longest(q: str, idx: Reader, cid: int) -> int:
    """q（已 _flat）跟段落最長的連續相同字數。"""
    t = _flat_of(idx, cid)
    return SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t)).size


def _search(a: Analysis, idx: Reader, *, top_k: int, cand: int, allow: set[int] | None = None,
            per_doc: int = PER_DOC) -> list[Hit]:
    scope = None
    if allow is None and (a.iso_system or a.doc_type) and not a.doc_nos:
        # 只加權、不硬性篩選：體系／階層詞也可能只是內文用語，硬篩會整份排除
        docs = {d["doc_no"] for d in filter_docs(a, idx)}
        scope = {cid for cid, c in idx.chunks.items() if c.doc_no in docs}
    channels: dict[str, tuple[float, list[int]]] = {}
    if a.doc_nos:
        # 代碼命中的文件：屬性卡排最前，再依 BM25 排該文件的段落
        hits = set(a.doc_nos)
        own = {cid for cid, c in idx.chunks.items() if c.doc_no in hits}
        ranked = [cid for cid, _ in idx.bm25(a.query, cand, own)]
        metas = [cid for cid in own if idx.chunks[cid].kind == "meta"]
        channels["code"] = (W_CODE, metas + [c for c in ranked if c not in metas])
    channels["bm25"] = (W_BM25, [cid for cid, _ in idx.bm25(a.query, cand, allow)])
    qv = embedder.get().encode([a.query])[0]
    channels["dense"] = (W_DENSE, [cid for cid, _ in idx.dense(qv, cand, allow)])
    if scope:
        channels["scope_bm25"] = (W_SCOPE, [cid for cid, _ in idx.bm25(a.query, cand, scope)])
        channels["scope_dense"] = (W_SCOPE, [cid for cid, _ in idx.dense(qv, cand, scope)])
    phrase = _phrase_rank(a.query, idx, list(dict.fromkeys(channels["bm25"][1] + channels["dense"][1])))
    if phrase:
        channels["phrase"] = (W_PHRASE, phrase)
    fused = sorted(_rrf(channels).items(), key=lambda x: -x[1][0])
    per_doc_cap = per_doc
    out, per_doc, used, seen_meta = [], {}, 0, set()
    for cid, (score, why) in fused:
        c = idx.chunks[cid]
        key = idx.docs[c.doc_no]["iso_key"] or c.doc_no      # 同 ISO 編號的多份文件算同一份
        if per_doc.get(key, 0) >= per_doc_cap:
            continue
        if c.kind == "meta":
            if key in seen_meta:
                continue
            seen_meta.add(key)
        if used + len(c.text) > CONTEXT_CHARS and out:
            continue
        out.append(Hit(c, score, why))
        per_doc[key] = per_doc.get(key, 0) + 1
        used += len(c.text)
        if len(out) >= top_k:
            break
    return out
