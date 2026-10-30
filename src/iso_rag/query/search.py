"""混合檢索：BM25（FTS5）+ 向量（bge-m3，段落與條文 MaxSim）+ 文件代碼命中 + 原文片段，加權 RRF 融合。

原文片段通道：問題跟段落有 ≥ PHRASE_MIN 個字連續相同（正規化後）就依相同長度排名。
使用者貼表單名稱、條文原句時，常是 BM25 排第 1 但向量沒找到，RRF 融合後會被「兩邊都中等」的段落擠掉，
這個通道把逐字命中的段落拉回前面。

融合後再依文件結構補齊（_finish）：檢索只取前 top_k 段，但很多問題的答案跨好幾段——
整個章節、整張表、被切在段落邊界的表格、同名文件的對應條文。補什麼都由文件資料決定，不看問法。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..index import embedder, textnorm
from ..index.store import ChunkRow, Reader
from ..ingest import chunker
from .router import Analysis, filter_docs

RRF_K = 60
W_CODE, W_BM25, W_DENSE, W_PHRASE = 2.0, 1.0, 1.0, 2.0
W_SCOPE = 0.75            # 問題提到的體系／階層：加權偏好（BM25、向量各一個通道），不硬性排除
CLAUSE_MAXSIM = True      # 向量分數也看段落裡最像的一條條文（store.Reader.dense）
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
    hits = _search(a, idx, top_k=max(top_k, FOCUS_TOP), cand=cand, final_k=top_k)
    a.focused = False
    if a.focus_docs:
        focus = set(a.focus_docs)
        top_docs = [h.chunk.doc_no for h in hits[:FOCUS_TOP]]
        if _quotes_elsewhere(a.query, idx, hits[:FOCUS_TOP], focus) or _named_elsewhere(a, idx, hits, focus):
            # 問題引用的原文、或鎖定用的標題詞在別份文件：標題詞只是主題相同，不鎖定
            return _finish(a, idx, hits[:top_k])
        if any(d in focus for d in top_docs) and any(d not in focus for d in top_docs):
            allow = {cid for cid, c in idx.chunks.items() if c.doc_no in focus}
            hits = _search(a, idx, top_k=top_k, cand=cand, allow=allow, per_doc=top_k)
            a.focused = True
        elif top_docs and all(d in focus for d in top_docs):
            a.focused = True
    return _finish(a, idx, hits[:top_k])


def _finish(a: Analysis, idx: Reader, hits: list[Hit]) -> list[Hit]:
    hits = _whole_tables(a, idx, _whole_sections(a, idx, _quoted_only(a, idx, hits)))
    return _sisters(a, idx, _split_rows(a, idx, _column_tables(a, idx, hits)))


# ---------------------------------------------------------------- 表格欄位
COLUMN_MAX = 2        # 欄名命中的表格最多補幾段（依頁序）


def _column_tables(a: Analysis, idx: Reader, hits: list[Hit]) -> list[Hit]:
    """問句含指名文件裡某張表的一格（≥3 欄的列裡、不含數字的格，例如表頭「版次」）：那張表的段落一定給 LLM，
    接在屬性卡後面。問句跟表格常只重疊欄名兩個字，一般檢索排不上來，LLM 只好拿別處的值回答。"""
    if not a.doc_nos:
        return hits
    q = _flat(a.query)
    have = {h.chunk.id for h in hits}
    add = []
    for cid, c in sorted(idx.chunks.items()):
        if c.doc_no not in a.doc_nos or c.kind == "meta" or cid in have:
            continue
        for ln in c.text.splitlines():
            cells = [x for x in re.split(r"\s*[|｜]\s*", ln) if x.strip()]
            if len(cells) >= 3 and any(len(f := _flat(x)) >= 2 and not re.search(r"\d", f) and f in q for x in cells):
                add.append(Hit(c, 0.0, ["column"]))
                break
        if len(add) >= COLUMN_MAX:
            break
    if not add:
        return hits
    k = sum(1 for h in hits if h.chunk.kind == "meta")
    return hits[:k] + add + hits[k:]


# ---------------------------------------------------------------- 整個章節
@dataclass
class Section:
    """文件的第一層章節（「3.0 定義」）。ids：內容落在這一節的 body 段落，依文件順序（跨節的段落兩節都算）；
    items：節內下一層條文 (條號, 條文名稱)，例 ("3.3", "嚴重缺點(CR)")；text：節的內文（不含章節標題那一行）。"""
    doc_no: str
    num: str
    title: str
    ids: list[int]
    items: list[tuple[str, str]]
    text: str = ""

    def label(self, idx: Reader) -> str:
        return f"{idx.docs[self.doc_no]['iso_no'] or self.doc_no} {self.num}. {self.title}"


def sections_of(idx: Reader, doc_no: str, pdf_sha: str | None) -> list[Section]:
    """依文件自己的條文標題（跟切段同一套 HeadingSeq）把 body 段落分到第一層章節。"""
    cache = idx.__dict__.setdefault("_section_cache", {})
    if (doc_no, pdf_sha) in cache:
        return cache[(doc_no, pdf_sha)]
    seq = chunker.HeadingSeq()
    out: list[Section] = []
    cur = None
    for cid in sorted(cid for cid, c in idx.chunks.items()
                      if c.doc_no == doc_no and c.pdf_sha == pdf_sha and c.kind == "body"):
        for ln in idx.chunks[cid].text.splitlines():
            h = seq.accept(chunker._is_heading(ln))
            title = re.sub(r"[\s:：.．、]+$", "", h[1]).strip() if h else ""
            # 相鄰段重疊一行：同一個標題不重開；同號不同名照開
            if h and "." not in h[0] and not (cur and (cur.num, cur.title) == (h[0], title)):
                cur = Section(doc_no, h[0], title, [], [])
                out.append(cur)
            if cur is None:
                continue
            if cid not in cur.ids:
                cur.ids.append(cid)
            if not (h and h[0] == cur.num):
                cur.text += ln + "\n"
            if h and h[0].count(".") == 1 and h[0].split(".")[0] == cur.num and h[0] not in dict(cur.items):
                cur.items.append((h[0], re.split(r"[:：]", h[1], maxsplit=1)[0].strip()[:20]))
    cache[(doc_no, pdf_sha)] = out
    return out


def _word(s: str) -> str:
    return re.sub(r"\W", "", _flat(s))


def common_len(a: str, b: str) -> int:
    """a、b 最長的連續相同字數。"""
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def _rest_of_query(a: Analysis, idx: Reader, hits: list[Hit]) -> dict[str, str]:
    """每份命中文件：問題（_flat）去掉文件代碼、跟文件名稱相同的片段（≥ 3 字）之後剩下的部分，片段處換成 \\0。
    文件名稱裡的「作業」「文件」「清單」不是在問章節或表格。"""
    q = _flat(a.query)
    rest: dict[str, str] = {}
    for doc_no in {h.chunk.doc_no for h in hits}:
        d = idx.docs[doc_no]
        r = q.replace(_flat(d["iso_no"] or "\0"), "\0")
        t = _flat(d["title"] or "")
        while len(r) >= 3 and (m := SequenceMatcher(None, r, t, autojunk=False)
                                .find_longest_match(0, len(r), 0, len(t))).size >= 3:
            r = r[:m.a] + "\0" + r[m.a + m.size:]
        rest[doc_no] = r
    return rest


def _place(out: list[Hit], idx: Reader, ids: list[int]) -> list[Hit]:
    """把 ids 的段落依文件順序放到 out 裡第一個屬於 ids 的位置（已在 out 的段落搬過去，沒有的補上）。"""
    pos = min(j for j, h in enumerate(out) if h.chunk.id in ids)
    have = {h.chunk.id: h for h in out}
    block = [have.get(i) or Hit(idx.chunks[i], out[pos].score, ["section"]) for i in ids]
    return out[:pos] + block + [h for h in out[pos:] if h.chunk.id not in ids]


def _trim(out: list[Hit], keep: set[int]) -> list[Hit]:
    """總長超過 CONTEXT_CHARS：從排名最後、不在 keep 的段落開始拿掉。"""
    while sum(len(h.chunk.text) for h in out) > CONTEXT_CHARS:
        j = next((j for j in range(len(out) - 1, -1, -1) if out[j].chunk.id not in keep), None)
        if j is None:
            break
        out.pop(j)
    return out


def _whole_sections(a: Analysis, idx: Reader, hits: list[Hit]) -> list[Hit]:
    """問題問的是指定文件的某一整章（「抽樣計畫作業指導書的定義是?」-> 3.0 定義）：把這一節的段落全補進來，
    記在 a.sections（答題時要求逐條列出、審查員據此判斷、程式核對有沒有漏條）。
    一節被切成好幾段時，檢索常漏掉中間那段，答案就少了幾條。
    補的條件全看文件結構：問題指向特定文件，且問題（去掉文件名稱、代碼後）含有某個被檢索到的章節標題。
    去掉章節標題後剩下的字詞還出現在這一節內文裡 -> 問的是節裡的某一條（「嚴重缺點的定義」），不補。
    整節超過 CONTEXT_CHARS 就不補。"""
    a.sections = []
    if not (a.focused or a.doc_nos):
        return hits
    rest = _rest_of_query(a, idx, hits)
    asked: list[Section] = []
    for h in hits:
        c = h.chunk
        if c.kind != "body":
            continue
        for sec in sections_of(idx, c.doc_no, c.pdf_sha):
            t = _word(sec.title)
            if c.id not in sec.ids or len(t) < 2 or t not in rest[c.doc_no] or sec in asked:
                continue
            body = _word(sec.text)
            if any(common_len(p, body) >= 2 for p in map(_word, rest[c.doc_no].replace(t, "\0").split("\0"))):
                continue
            asked.append(sec)
    out = list(hits)
    for sec in asked:
        if sum(len(idx.chunks[i].text) for i in sec.ids) > CONTEXT_CHARS:
            continue
        out = _place(out, idx, sec.ids)
        a.sections.append(sec)
    return _trim(out, {i for s in a.sections for i in s.ids})


# ---------------------------------------------------------------- 整張表
_CAPTION_RE = re.compile(r"^\s*(表\s*(\d{1,2}))\s*[:：.、]?\s*(\S.{1,40})$")


@dataclass
class Table:
    """文件裡一張有標題的表（「表 5 管制物質清單」）。ids：表格所在的段落，依文件順序；head：表頭那一行。"""
    doc_no: str
    num: str
    title: str
    ids: list[int]
    head: str = ""

    def label(self, idx: Reader) -> str:
        return f"{idx.docs[self.doc_no]['iso_no'] or self.doc_no} 表{self.num} {self.title}"


def tables_of(idx: Reader, doc_no: str, pdf_sha: str | None) -> list[Table]:
    """依表格標題（「表 N 名稱」）找出文件裡的表。表的範圍由資料決定：表頭（標題下一行）在後面每一頁重複出現，
    表格延伸到表頭最後出現的那一頁；遇到下一個表格標題就停。"""
    cache = idx.__dict__.setdefault("_table_cache", {})
    if (doc_no, pdf_sha) in cache:
        return cache[(doc_no, pdf_sha)]
    ids = sorted(cid for cid, c in idx.chunks.items()
                 if c.doc_no == doc_no and c.pdf_sha == pdf_sha and c.kind in ("body", "table"))
    lines = [(cid, ln) for cid in ids for ln in idx.chunks[cid].text.splitlines() if ln.strip()]
    caps = [(i, m) for i, (_, ln) in enumerate(lines) if (m := _CAPTION_RE.match(ln))]
    out: list[Table] = []
    for k, (i, m) in enumerate(caps):
        end = caps[k + 1][0] if k + 1 < len(caps) else len(lines)
        start_cid = lines[i][0]
        head = _word(lines[i + 1][1]) if i + 1 < end else ""
        last_page = idx.chunks[start_cid].page_start
        if len(head) >= 4:
            for cid, ln in lines[i + 2:end]:
                if common_len(head, _word(ln)) >= 0.7 * len(head):     # 跨頁段落：表頭可能在它的最後一頁
                    last_page = max(last_page, idx.chunks[cid].page_end)
        stop = lines[end][0] if end < len(lines) else None
        span = []
        for cid in ids[ids.index(start_cid):]:
            if (cid == stop and cid != start_cid) or idx.chunks[cid].page_start > last_page:
                break
            span.append(cid)
        out.append(Table(doc_no, m.group(2), m.group(3).strip(), span,
                         " ".join(lines[i + 1][1].replace("|", " ").split()) if len(head) >= 4 else ""))
    cache[(doc_no, pdf_sha)] = out
    return out


def _names_table(rest: str, t: Table) -> bool:
    """問題（去掉文件名稱後）提到這張表：寫了「表N」，或表名（含／去掉括號）大半出現在問題裡。"""
    r = _word(rest)
    if re.search(rf"表{t.num}(?!\d)", r):
        return True
    for v in {_word(t.title), _word(re.sub(r"[(（][^)）]*[)）]", "", t.title))}:
        if len(v) >= 3 and common_len(v, r) >= min(len(v), max(4, -(-len(v) * 3 // 5))):
            return True
    return False


def _whole_tables(a: Analysis, idx: Reader, hits: list[Hit]) -> list[Hit]:
    """問題問的是指定文件的某一整張表（「…的管制物質清單裡等級一的有哪些」）：要篩選或列出整張表。
    檢索只取前 top_k 段，幾百列的長表大部分送不到 LLM。表不長（≤ CONTEXT_CHARS）就把整張表放進摘錄；
    長表記在 a.tables，答題時分段逐段篩選再合併（answer._generate_table）。"""
    a.tables = []
    if not (a.focused or a.doc_nos):
        return hits
    rest = _rest_of_query(a, idx, hits)
    out = list(hits)
    seen: set[tuple[str, str]] = set()
    for h in hits:
        c = h.chunk
        if c.kind not in ("body", "table"):
            continue
        for t in tables_of(idx, c.doc_no, c.pdf_sha):
            if c.id not in t.ids or (t.doc_no, t.num) in seen or not _names_table(rest[c.doc_no], t):
                continue
            seen.add((t.doc_no, t.num))
            if sum(len(idx.chunks[i].text) for i in t.ids) > CONTEXT_CHARS:
                a.tables.append(t)
            else:
                out = _place(out, idx, t.ids)
    keep = {i for t in a.tables for i in t.ids} | {i for s in a.sections for i in s.ids}
    return _trim(out, keep) if out != hits else hits


def _row_cells(line: str) -> int:
    """表格列（「a | b | c」）有幾欄；不是表格列 -> 0。"""
    return line.count(" | ") + 1 if " | " in line else 0


def _edge_rows(c) -> tuple[int, int]:
    """段落第一行、最後一行各是幾欄的表格列（不是表格列 -> 0）。"""
    lines = [ln for ln in c.text.splitlines() if ln.strip()]
    return (_row_cells(lines[0]), _row_cells(lines[-1])) if lines else (0, 0)


def _split_rows(a: Analysis, idx: Reader, hits: list[Hit]) -> list[Hit]:
    """表格被切在段落邊界：檢索到的段落以表格列結尾（或開頭），相鄰段落（同文件、同 PDF、頁碼相接）接著是
    同樣欄數的表格列 -> 是同一張表被切開，把接續的段落補進來（依文件順序放在一起）。
    例：修訂履歷被切成 3 段、只檢索到前兩段，問「有哪些版次」就漏掉最新版。整串超過 CONTEXT_CHARS 就不補。"""
    out = list(hits)
    keep: set[int] = set()
    for h in hits:
        c = h.chunk
        if c.kind not in ("body", "table") or c.id in keep:
            continue
        run = [c.id]

        def joined(a, b) -> bool:              # a、b 相鄰（a 在前）且 a 的最後一列接著 b 的第一列
            return (b.doc_no == a.doc_no and b.pdf_sha == a.pdf_sha and b.kind in ("body", "table")
                    and a.page_end <= b.page_start <= a.page_end + 1
                    and _edge_rows(a)[1] and _edge_rows(a)[1] == _edge_rows(b)[0])
        while (p := idx.chunks.get(run[0] - 1)) and joined(p, idx.chunks[run[0]]):
            run.insert(0, p.id)
        while (n := idx.chunks.get(run[-1] + 1)) and joined(idx.chunks[run[-1]], n):
            run.append(n.id)
        if len(run) == 1 or sum(len(idx.chunks[i].text) for i in run) > CONTEXT_CHARS:
            continue
        out = _place(out, idx, run)
        keep |= set(run)
    return _trim(out, keep | {i for t in a.tables for i in t.ids} | {i for s in a.sections for i in s.ids}) if keep else hits


# ---------------------------------------------------------------- 原文引用與焦點
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


SISTER_MIN = 0.7      # 同名文件的對應段落：內容相似度門檻（對應條文通常 > 0.9，不對應的 < 0.65）


def _title_core(d: dict) -> str:
    """文件名稱去掉代碼、版次、數字與符號：「SOP-X-001 樣品制作作業辦法 A5」=「WI1001-A0-樣品制作作業辦法」。"""
    t = textnorm.CODE_RE.sub(" ", textnorm.norm(d["title"] or ""))
    return re.sub(r"[-_]?[a-z]\d{1,2}\b|v\d+(?:\.\d+)?|[\d\W_]+", "", t)


def _sister_map(idx: Reader) -> dict[str, list[str]]:
    """文件 -> 名稱相同、但是另一份（不同單位各自制定）的文件。"""
    cache = idx.__dict__.setdefault("_sister_cache", {})
    if not cache:
        groups: dict[str, list[dict]] = {}
        for d in idx.docs.values():
            if d.get("pdf_path") and len(core := _title_core(d)) >= 3:
                groups.setdefault(core, []).append(d)
        for ds in groups.values():
            for d in ds:
                cache[d["doc_no"]] = [o["doc_no"] for o in ds if o["doc_no"] != d["doc_no"]]
        cache.setdefault("", [])
    return cache


def _sisters(a: Analysis, idx: Reader, hits: list[Hit]) -> list[Hit]:
    """問題沒指向特定文件時，同名文件（不同單位各有一份）要並列：檢索到其中一份的段落，
    就把另一份裡內容最像的那段（≥ SISTER_MIN）接在後面，讓 LLM 分文件列出各自的規定，
    而不是只答出排名較前的那一份。"""
    if a.doc_nos or a.focused or a.sections or a.tables:
        return hits
    sis = _sister_map(idx)
    have = {h.chunk.id for h in hits}
    out: list[Hit] = []
    for h in hits:
        out.append(h)
        if h.chunk.kind != "body":
            continue
        fa = _flat_of(idx, h.chunk.id)
        for y in sis.get(h.chunk.doc_no, []):
            cands = [cid for cid, c in idx.chunks.items() if c.doc_no == y and c.kind == "body"]
            best = max(((SequenceMatcher(None, fa, _flat_of(idx, cid), autojunk=False).ratio(), cid) for cid in cands),
                       default=(0.0, None))
            if best[0] >= SISTER_MIN and best[1] not in have:
                have.add(best[1])
                out.append(Hit(idx.chunks[best[1]], h.score, h.why + ["sister"]))
    return out


def _named_elsewhere(a: Analysis, idx: Reader, hits: list[Hit], focus: set[str]) -> bool:
    """用標題詞鎖定的焦點文件（不是明講文件代碼），那個詞卻原封不動出現在排第一、而且不是焦點文件的段落：
    那份文件跟這個名稱一樣相關，不鎖定。例：問「ORT 留樣登記表的表單編號」，「ORT 留樣登記表」是一份表單文件的
    標題，但答案（表單編號）寫在引用它的程序書條文裡。"""
    if a.focus_term == "文件代碼" or not a.focus_term or not hits or hits[0].chunk.doc_no in focus:
        return False
    return _flat(a.focus_term) in _flat_of(idx, hits[0].chunk.id)


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


# ---------------------------------------------------------------- 多通道融合
def _search(a: Analysis, idx: Reader, *, top_k: int, cand: int, allow: set[int] | None = None,
            per_doc: int = PER_DOC, final_k: int = 0) -> list[Hit]:
    """final_k：呼叫端最後只取前幾筆（各路第 1 名要排進這個範圍，見 _champions）；0 = top_k。"""
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
    channels["dense"] = (W_DENSE, [cid for cid, _ in idx.dense(qv, cand, allow, clauses=CLAUSE_MAXSIM)])
    if scope:
        channels["scope_bm25"] = (W_SCOPE, [cid for cid, _ in idx.bm25(a.query, cand, scope)])
        channels["scope_dense"] = (W_SCOPE, [cid for cid, _ in idx.dense(qv, cand, scope, clauses=CLAUSE_MAXSIM)])
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
    return _champions(channels, dict(fused), idx, out, final_k or top_k)


def _champions(channels: dict, fused: dict, idx: Reader, out: list[Hit], final_k: int) -> list[Hit]:
    """每一路的第 1 名一定排進前 final_k（換掉前 final_k 裡融合分數最低、又不是哪一路第 1 名的段落，被換掉的往後排）。
    RRF 只看名次：只有一路排第 1 的段落（例如向量最像、但關鍵字完全對不上的換句話問法），
    會輸給兩路都排中段的段落；某一路最有把握的答案不該被融合擠掉。"""
    tops = list(dict.fromkeys(lst[0] for _, lst in channels.values() if lst))
    head, tail = out[:final_k], out[final_k:]
    have = {h.chunk.id for h in head}
    for cid in sorted((c for c in tops if c not in have and c in fused), key=lambda c: -fused[c][0]):
        drop = next((i for i in range(len(head) - 1, -1, -1) if head[i].chunk.id not in tops), None)
        if drop is None:
            break
        tail.insert(0, head.pop(drop))
        head.append(Hit(idx.chunks[cid], *fused[cid]))
    ids = {h.chunk.id for h in head}
    return head + [h for h in tail if h.chunk.id not in ids]
