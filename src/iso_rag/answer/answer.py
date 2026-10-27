"""產生答案。

- 列表題、純屬性題：直接由屬性表組答案，不呼叫 LLM（LLM 單題要十幾秒到數分鐘）。
- 內容題：編號摘錄 [1]..[n] 送 LLM，要求只用 [n] 引用；程式再把 [n] 換成
  【SOP-QA-001 p.4 5.7】（小模型自己寫文件/頁碼容易照抄模板）。
- 全文核對：問題逐字引用原文或提到少見的代碼時，比對全部文件，結果當成一段證據附給 LLM。
- 逐行核對（_verify）：每一行只拿它引用的摘錄核對數字、代碼、英文字與搬錯段落的中文詞；
  不通過就附上原因重寫，到上限仍不通過就在那一行標出核對不到的詞。
"""
from __future__ import annotations

import re
import unicodedata
import time
from dataclasses import dataclass, field

from .. import llm
from ..index import textnorm
from ..ingest import chunker
from ..index.store import Reader
from ..query.router import Analysis, filter_docs
from ..query.search import Hit, quoted_span, search

SYSTEM = """你是 ISO 體系文件問答助理。只根據使用者提供的編號摘錄回答。
規則：
1. 使用繁體中文純文字，不要用 Markdown（不要 **、#、表格）。
2. 每個重點一行，以「- 」開頭；行尾用方括號標出依據的摘錄編號，例如 [1] 或 [2][3]。
3. 數字、單位、頻率、文件與表單編號照摘錄原文寫，不要換算、不要推測。
4. 只回答問到的那一件事，通常 1～3 行就夠；沒問到的不要提供（不要附帶摘錄裡其他規定、文件屬性、版本、日期）。
5. 摘錄可能來自好幾份文件；只採用跟問題主題同一份文件的摘錄，不要把不同文件的規定混在一起。
6. 摘錄中有寫到（即使用詞不同，例如「需求數量」對應「可以申請幾個」）就要回答；完全沒寫才回答「文件未規定」，不要用常識補。
7. 摘錄是 OCR 文字，可能夾雜簡體字或錯字，回答時用正確的繁體字。
8. ISO 文件的固定章節是：目的、範圍、權責、定義、作業內容。問「適用哪些人／適用範圍／適用於哪裡」時，
   第一行要先寫「範圍」章節的內容，再補充作業內容裡的細項。
9. 簽名、簽核欄只回報「已簽」或「未簽」，不要寫簽名的人名（手寫字 OCR 不可靠）。
10. 標題為「全文核對」的摘錄是系統逐字比對全部文件的結果（不限前幾段）。問題問「哪些文件提到／引用／出現在哪裡」時，
   依它逐一列出文件與頁碼，引用它的編號；問題問的是內容時，不要提它。"""


@dataclass
class Answer:
    text: str
    mode: str                     # list / attribute / llm / no_llm / not_found
    hits: list[Hit] = field(default_factory=list)
    docs: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    llm_seconds: float = 0.0
    grounded: bool | None = None
    note: str = ""
    # 答案實際引用的 (hits 索引 0 起, 頁碼)，依引用先後；跨頁段落只留答案字詞真的出現的頁
    cited: list[tuple[int, list[int]]] = field(default_factory=list)
    # 全文核對（_fulltext）的結果：送 LLM 時是最後一段摘錄 [len(hits)+1]，被引用時 cited 記成索引 len(hits)
    fulltext_text: str = ""
    fulltext: list[tuple[Hit, list[int]]] = field(default_factory=list)
    verify_log: list[str] = field(default_factory=list)   # 每一輪驗證的結果（評測、除錯用）


def page_label(pages: list[int]) -> str:
    """[4] -> p.4；[1, 2, 3] -> p.1-3；[1, 3] -> p.1、3"""
    ps = sorted(set(pages))
    if len(ps) > 1 and ps == list(range(ps[0], ps[-1] + 1)):
        return f"p.{ps[0]}-{ps[-1]}"
    return "p." + "、".join(map(str, ps))


def cite_label(idx: Reader, c, pages: list[int] | None = None) -> str:
    """pages：只標這幾頁（答案的證據頁，見 _evidence_pages）；None = 段落的整個頁碼範圍。"""
    d = idx.docs[c.doc_no]
    name = d["iso_no"] or d["doc_no"]
    if c.kind == "meta":
        return f"{name} 文件屬性"
    page = page_label(pages or list(range(c.page_start, c.page_end + 1)))
    if c.kind == "table":                 # 表格類文件沒有條文編號，只標頁碼
        return f"{name} {page} 表格"
    nums = chunker.headings_in(c.text, c.clause)
    first, last = (nums[0], nums[-1]) if nums else (c.clause, c.clause)
    clause = f" {first}" if first and first == last else (f" {first}–{last}" if first else "")
    kind = {"issue": " 會審表", "revision": " 修訂履歷"}.get(c.kind, "")
    return f"{name} {page}{clause}{kind}"


def _meta_value(d: dict, f: str):
    m = d["meta"]
    v = m.get(f)
    if f == "files":
        v = "、".join(x["filename"] for x in m.get("files") or [])
    if f == "related_iso_docs":
        v = "、".join(v or [])
    return v or "（未填）"


def _answer_list(a: Analysis, idx: Reader) -> Answer:
    docs = filter_docs(a, idx)
    cond = "、".join(x for x in [f"ISO {a.iso_system}" if a.iso_system else "", a.doc_type or "", a.status or ""] if x) or "全部"
    lines = [f"ISO 文件中，符合「{cond}」的共 {len(docs)} 份："]
    for d in docs:
        t = (d["title"] or "").strip()
        name = t if d["iso_no"] and t.startswith(d["iso_no"]) else f"{d['iso_no'] or '（無ISO編號）'} {t}"
        lines.append(f"- {name}（{d['doc_no']}，{d['iso_system'] or '-'}，"
                     f"{d['doc_type'] or '-'}，{d['status'] or '-'}，生效 {(d['effective_date'] or '-')[:10]}）")
    return Answer("\n".join(lines), "list", docs=docs)


def _answer_attr(a: Analysis, idx: Reader) -> Answer:
    lines = []
    docs = [idx.docs[n] for n in a.doc_nos]
    for d in docs:
        t = d["title"] or ""
        head = f"{t if d['iso_no'] and t.startswith(d['iso_no']) else (d['iso_no'] or '') + ' ' + t}（{d['doc_no']}）".strip()
        lines.append(head)
        for f, label in a.attrs:
            lines.append(f"- {label}：{_meta_value(d, f)}")
    if len(docs) > 1:
        lines.append(f"註：{len(docs)} 份文件對應到同一個編號，已全部列出。")
    return Answer("\n".join(lines), "attribute", docs=docs)


def _context(hits: list[Hit], idx: Reader) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        d = idx.docs[c.doc_no]
        parts.append(f"[{i}] {d['iso_no'] or d['doc_no']} {d['title']}｜{cite_label(idx, c)}｜{c.section}\n"
                     f"{c.text}")
    return "\n\n".join(parts)


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_CITE_RE = re.compile(r"\[(\d+)\]")
_LATIN_RE = re.compile(r"[a-z][a-z\-]+")
_CJK_RUN = re.compile(f"[{textnorm.CJK}]+")
MIN_CJK_MISS = 3          # 連續 3 字以上沒根據、且原封不動出現在「別段」摘錄 -> 從別段搬來的
# 職稱後面接人名（norm 之後是簡體）：「審核：Alex」「核准 王小明」
_ROLE_PAIR_RE = re.compile(rf"(核准|审核|提出人|制定/修改|制定|制订)[\s:：]*([{textnorm.CJK}]{{2,4}}|[a-z][a-z.]+)"
                           rf"(?=[\s、，,；;。(（\[]|$)")


def _missing(body: str, src: str, every: str = "", question: str = "") -> list[str]:
    """body 裡出現、src（該行引用的摘錄）裡卻找不到的東西：
    - 數字、文件代碼、英文字：一律要出現在 src。
    - 中文：改寫很常見，只抓「src 沒有、但原封不動出現在別段摘錄」的 3 字以上詞——
      把 A 段的人名／規定掛到 B 段引用是最常見的錯。
    - 職稱配人名：「審核 Alex」在 src 裡必須是相鄰的一對（避免把某一欄的人名配到別的職稱）。
    使用者問題裡的詞是答案在複述問題，不算。"""
    src_n = textnorm.norm(src)
    flat = src_n.replace(" ", "")
    every_flat = textnorm.norm(every).replace(" ", "")
    q_flat = textnorm.norm(question).replace(" ", "")
    keys = textnorm.code_key(src)
    body_n = textnorm.norm(body)
    bad: list[str] = []

    def add(x):
        if x not in bad:
            bad.append(x)
    for code in textnorm.find_codes(body):
        if code not in keys:
            add(code)
    for n in _NUM_RE.findall(body_n):
        if n not in flat:
            add(n)
    for w in _LATIN_RE.findall(body_n):
        if w not in src_n and textnorm.code_key(w) not in keys:
            add(w)
    # 中文：一個字只要跟左或右鄰字組成的雙字組出現在摘錄裡就算有根據；
    # 連續 MIN_CJK_MISS 個以上都沒根據的字才算「摘錄沒有的詞」。
    # 例：「主管包含製造部」只有「包含」沒根據（2 字的連接詞，不算）；換掉的人名 3 字都沒根據。
    grams = {flat[i:i + 2] for i in range(len(flat) - 1)}
    for rm in _CJK_RUN.finditer(body_n):
        run = rm.group(0)
        ok = [run[k - 1:k + 1] in grams if k else False for k in range(len(run))]
        ok = [ok[k] or (k + 1 < len(run) and run[k:k + 2] in grams) for k in range(len(run))]
        if len(run) == 1:
            ok = [run in flat]
        k = 0
        while k < len(run):
            if ok[k]:
                k += 1
                continue
            j = k
            while j < len(run) and not ok[j]:
                j += 1
            span = run[k:j]
            left = body_n[:rm.start() + k].replace(" ", "")[-2:]
            right = body_n[rm.start() + j:].replace(" ", "")[:2]
            if len(span) >= MIN_CJK_MISS and _ocr_fixed(left, span, right, flat):
                k = j                       # LLM 把 OCR 錯字改正了（rule 7），不是搬錯內容
                continue
            for size in range(len(span), MIN_CJK_MISS - 1, -1):     # 最長、且在別段原封不動出現的那一截
                hit = next((span[a:a + size] for a in range(len(span) - size + 1)
                            if span[a:a + size] in every_flat and span[a:a + size] not in q_flat), None)
                if hit:
                    add(textnorm.to_traditional(hit))
                    break
            k = j
    for role, name in _ROLE_PAIR_RE.findall(body_n):
        if "签" in name:                       # 「核准欄已簽」是狀態不是人名
            continue
        if not re.search(re.escape(role) + r"[\s:：]*" + re.escape(name), src_n):
            m = re.search(re.escape(name), body, re.I)          # 英文名保留原本大小寫
            add(textnorm.to_traditional(f"{role} {m.group(0) if m else name}"))
    return bad


OCR_MARGIN = 3.0      # 對齊位置的原文比 LLM 寫的字串，每字 log2 機率低這麼多 -> 原文是 OCR 錯字
_lm_cache: dict = {}


def _lm():
    """學習層上線中的字元語言模型（learn/quality.py 的 CharLM）；載不到 -> None。"""
    if "lm" not in _lm_cache:
        try:
            from ..learn.quality import QualityModel
            from ..learn.store import Store
            _lm_cache["lm"] = QualityModel.load_active(Store()).lm
        except Exception:  # noqa: BLE001 -- 沒有學習層就照舊嚴格核對
            _lm_cache["lm"] = None
    return _lm_cache["lm"]


def _ocr_fixed(left: str, span: str, right: str, src_flat: str) -> bool:
    """答案裡核對不到的 span，在引用原文的同一位置（前後文 left/right 對得上、長度差 ≤ 1）有一段字，
    而那段字依字元語言模型明顯不像正常文字（OCR 錯字）-> LLM 是在改正 OCR（SYSTEM rule 7），不算捏造。
    人名換成別人的名字不會過：原文那段本身就是正常文字。"""
    lm = _lm()
    if lm is None or len(left) + len(right) < 2:
        return False
    mine = lm.score(left + span + right)[0]
    n = len(span)
    for size in range(max(1, n - 1), n + 2):
        for m in re.finditer(f"(?={re.escape(left)}(.{{{size}}}){re.escape(right)})", src_flat):
            w = m.group(1)
            if w != span and lm.score(left + w + right)[0] < mine - OCR_MARGIN:
                return True
    return False


def _verify(ans: str, texts: list[str], question: str = "",
            kin: list[set[int]] | None = None) -> dict[int, list[str]]:
    """逐行核對：行號 -> 核對不到的詞。每行只拿它自己引用的摘錄比對（沒標引用就比對全部摘錄）。
    kin[i]：跟摘錄 i 同文件、頁碼重疊的摘錄（0 起）。引用標籤只顯示到文件＋頁，
    內容在 A 段卻標同頁的 B 段，顯示出來的出處一樣，不算錯。"""
    every = "\n".join(texts)
    out: dict[int, list[str]] = {}
    for i, ln in enumerate(ans.splitlines()):
        nums = [int(n) for n in _CITE_RE.findall(ln) if 1 <= int(n) <= len(texts)]
        if kin:
            nums = list(dict.fromkeys(m + 1 for n in nums for m in sorted({n - 1} | kin[n - 1])))
        src = "\n".join(texts[n - 1] for n in nums) if nums else every
        bad = _missing(_CITE_RE.sub("", ln), src, every, question)
        if bad:
            out[i] = bad
    return out


_DIGITS_RE = re.compile(r"\d+")


def _evidence_pages(line: str, c) -> list[int]:
    """段落跨好幾頁時，答案這一行的字詞真的出現在哪幾頁（引用只標那幾頁）。
    依逐頁快取文字計分：數字、英文代碼命中數優先，再比中文雙字組命中數，取最高分的頁；
    都沒命中（或沒有頁面快取）就整段的頁都留。"""
    pages = list(range(c.page_start, c.page_end + 1)) if c.page_start else []
    if len(pages) < 2 or not c.pdf_sha:
        return pages
    from ..ingest.extract import load_page
    body = textnorm.norm(_CITE_RE.sub("", line))
    hard = set(_DIGITS_RE.findall(body)) | set(_LATIN_RE.findall(body))
    flat = body.replace(" ", "")
    grams = {flat[i:i + 2] for i in range(len(flat) - 1) if _CJK_RUN.fullmatch(flat[i:i + 2])}
    scored = []
    for p in pages:
        rec = load_page(c.pdf_sha, p)
        t = textnorm.norm(rec.get("text") or "") if rec else ""
        tf = t.replace(" ", "")
        scored.append(((sum(1 for x in hard if re.search(rf"(?<![0-9a-z]){re.escape(x)}(?![0-9a-z])", t)),
                        sum(1 for g in grams if g in tf)), p))
    best = max(s for s, _ in scored)
    return [p for s, p in scored if s == best] if best > (0, 0) else pages


def _replace_cites(ans: str, hits: list[Hit], idx: Reader,
                   fulltext: bool = False) -> tuple[str, list[tuple[int, list[int]]]]:
    """[n] -> 【SOP-QA-001 p.4 5.7】，頁碼用這一行的證據頁（_evidence_pages）。
    fulltext：最後一段 [len(hits)+1] 是全文核對 -> 【全文核對】，cited 記成索引 len(hits)。
    回傳 (換好的答案, 引用的 (hits 索引, 頁碼))。"""
    cited: dict[int, list[int]] = {}
    out = []
    for ln in ans.splitlines():
        def rep(m, ln=ln):
            labels, seen = [], set()     # seen：(文件, 頁)——兩段縮到同一頁時只標一次
            for n in dict.fromkeys(int(x) for x in re.findall(r"\d+", m.group(0))):
                if fulltext and n == len(hits) + 1:
                    cited.setdefault(len(hits), [])
                    if "全文核對" not in labels:
                        labels.append("全文核對")
                elif 1 <= n <= len(hits):
                    c = hits[n - 1].chunk
                    pages = _evidence_pages(ln, c)
                    got = cited.setdefault(n - 1, [])
                    got += [p for p in pages if p not in got]
                    lab = cite_label(idx, c, pages)
                    key = (c.doc_no, c.pdf_sha, tuple(pages)) if pages else lab
                    if lab not in labels and key not in seen:
                        labels.append(lab)
                        seen.add(key)
            return f"【{'；'.join(labels)}】" if labels else ""
        out.append(re.sub(r"(?:\[\d+\])+", rep, ln))
    return "\n".join(out), list(cited.items())


def _fulltext(needle: str, a: Analysis, idx: Reader,
              label: str = "") -> tuple[str, list[tuple[Hit, list[int]]], int] | None:
    """全文核對：問題逐字引用的原文 needle（_flat 後，見 search.quoted_span）在全部文件（每一段，含屬性卡）
    出現在哪些文件、哪幾頁。結果當成一段「系統核對過的證據」交給 LLM，要不要用由 LLM 依問題判斷。
    LLM 只看得到檢索的前幾段，問「哪些文件有提到 X」時會答不完整，所以由資料觸發補上全文比對。
    回傳 (證據文字, [(段落, 頁碼)], 文件總數)；全文都沒有 -> None。"""
    from ..ingest.extract import load_page
    from ..query.search import _flat
    if len(needle) < 2:
        return None
    # 英文、數字開頭／結尾要有字界：ORT 不能算到 report 裡。字界要在保留空白的原文上判斷
    # （去掉空白後「SOP-QA-001 ORT」變成「001ort」），中間允許 OCR 插進來的空白。
    edge = needle[0].isascii() and needle[0].isalnum() or needle[-1].isascii() and needle[-1].isalnum()
    spaced = re.compile(("(?<![0-9a-z])" if needle[0].isascii() and needle[0].isalnum() else "")
                        + r"[\s|]*".join(map(re.escape, needle))
                        + ("(?![0-9a-z])" if needle[-1].isascii() and needle[-1].isalnum() else ""))

    def has(text: str, flat: str | None = None) -> bool:
        if needle not in (_flat(text) if flat is None else flat):
            return False
        return not edge or bool(spaced.search(textnorm.norm(text)))
    allow = {d["doc_no"] for d in filter_docs(a, idx)} if a.iso_system else None
    cache = idx.__dict__.setdefault("_flat_cache", {})
    # 一個 PDF 一組（同 ISO 編號的不同版本各自的頁碼不能混在一起）
    found: dict[tuple[str, str], dict] = {}     # (文件編號, pdf_sha) -> {"d", "c", "hits": [(Hit, pages)], "pages", "fields"}
    for cid, c in idx.chunks.items():
        if allow is not None and c.doc_no not in allow:
            continue
        t = cache.get(cid)
        if t is None:
            t = cache[cid] = _flat(c.text)
        if not has(c.text, t):
            continue
        d = idx.docs[c.doc_no]
        meta = c.kind == "meta" or not c.page_start
        g = found.setdefault((c.doc_no, (d["pdf_sha"] if meta else c.pdf_sha) or ""),
                             {"d": d, "c": c, "hits": [], "pages": [], "fields": []})
        if meta:                                # 只在屬性卡裡：標出是哪個欄位
            for ln in c.text.splitlines():
                k, _, v = ln.partition("：")
                if v and has(v) and k.strip() not in g["fields"]:
                    g["fields"].append(k.strip())
            g["hits"].append((Hit(c, 1.0, ["mention"]), []))
            continue
        g["c"] = c
        pages = list(range(c.page_start, c.page_end + 1))
        if len(pages) > 1 and c.pdf_sha:        # 跨頁段落：只留 X 真的出現的頁
            on = [p for p in pages if (r := load_page(c.pdf_sha, p)) and has(r.get("text") or "")]
            pages = on or pages
        new = [p for p in pages if p not in g["pages"]]
        if new:
            g["pages"] += new
            g["hits"].append((Hit(c, 1.0, ["mention"]), new))
    if not found:
        return None
    # 同 ISO 編號的另一份已有內文命中：只在屬性命中的那份不另列
    in_body = {g["d"]["iso_key"] for g in found.values() if g["pages"] and g["d"]["iso_key"]}
    groups = [g for g in found.values() if g["pages"] or g["d"]["iso_key"] not in in_body]
    groups.sort(key=lambda g: (g["d"]["iso_no"] or "~", g["d"]["doc_no"], g["c"].kind == "meta"))
    lines = [f"全文核對（系統逐字比對全部文件的結果，不是摘錄）：「{label or textnorm.to_traditional(needle)}」"
             f"出現在 {len({g['d']['doc_no'] for g in groups})} 份文件："]
    entries: list[tuple[Hit, list[int]]] = []
    if len(groups) > FULLTEXT_LINES:
        lines[0] += f"（只列前 {FULLTEXT_LINES} 份）"
    for g in groups[:FULLTEXT_LINES]:
        d = g["d"]
        t = (d["title"] or "").strip()
        name = " ".join((t if d["iso_no"] and t.startswith(d["iso_no"]) else f"{d['iso_no'] or d['doc_no']} {t}").split())
        where = "、".join(([page_label(g["pages"])] if g["pages"] else [])
                         + ([f"屬性「{'、'.join(g['fields'])}」"] if g["fields"] and not g["pages"] else []))
        own = "（文件本身）" if has(t) else ""
        lines.append(f"- {name}：{where or '屬性'}{own}")
        entries += g["hits"]
    return "\n".join(lines), entries, len({g["d"]["doc_no"] for g in groups})


_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")


def _pick_fulltext(q: str, a: Analysis, idx: Reader, hits: list[Hit]):
    """要不要附全文核對、核對哪一段，由資料決定（不看問法）：
    1. 問題跟檢索結果有 ≥ QUOTE_MIN 字逐字相同的一段（文件名稱、條文原句）-> 優先核對那一段。
    2. 問題裡的英文詞／代碼（ORT、MRB、SIP-QA101）：全庫含它的文件 ≤ FULLTEXT_LINES 份（列得完整）的，
       取最少見的那個。ISO、SOP 這種幾乎每份都有的詞自然不會選上。
    「只出現在問題點名的那份文件」的候選排最後（「SOP-QA-001 的 ORT…」核對 SOP-QA-001 只得到它自己，ORT 才有資訊）；
    沒有別的候選時還是用它（「哪些文件引用 WI1001」要的正是「只有它自己」這個事實）。"""
    from ..query.search import _flat
    named = set(a.doc_nos)
    cands = []                      # (只在點名文件, 不是長原文, 文件數, 核對結果)

    def add(ft, is_term: bool):
        if ft and (not is_term or ft[2] <= FULLTEXT_LINES):
            cands.append(({h.chunk.doc_no for h, _ in ft[1]} <= named, is_term, ft[2], ft))

    terms = list(dict.fromkeys(_TERM_RE.findall(unicodedata.normalize("NFKC", q))))
    quote = quoted_span(q, idx, hits, QUOTE_MIN)
    if quote:
        add(_fulltext(quote, a, idx, label=next((r for r in terms if _flat(r) == quote), "")), False)
    for raw in terms:
        term = _flat(raw)
        if len(term) >= 2:
            add(_fulltext(term, a, idx, label=raw), True)
    return min(cands, key=lambda c: c[:3])[3] if cands else None


def prepare(q: str, idx: Reader, a: Analysis) -> Answer:
    """不呼叫 LLM 的部分：列表題/屬性題直接得出答案；內容題做檢索（mode=pending）。"""
    t0 = time.monotonic()
    if a.intent == "list":
        res = _answer_list(a, idx)
    elif a.intent == "attribute" and a.attr_only:
        res = _answer_attr(a, idx)
        res.hits = search(a, idx)[:3]
    else:
        hits = search(a, idx)
        res = Answer("查無相關文件。", "not_found") if not hits else Answer("", "pending", hits)
        if hits:
            ft = _pick_fulltext(q, a, idx, hits)
            if ft:
                res.fulltext_text, res.fulltext, _ = ft
    res.seconds = round(time.monotonic() - t0, 2)
    return res


def generate(q: str, idx: Reader, res: Answer, *, timeout: float = 240, on_wait=None) -> Answer:
    """內容題：送 LLM、grounding 檢查、換引用。on_wait(秒數) 在等待期間被呼叫（給串流送心跳）。"""
    if res.mode != "pending":
        return res
    t0 = time.monotonic()
    hits = res.hits
    ft = bool(res.fulltext_text)
    ctx = _context(hits, idx) + (f"\n\n[{len(hits) + 1}] {res.fulltext_text}" if ft else "")
    # 核對用的原文要包含 LLM 看得到的摘錄標頭（文件編號、名稱）：只比內文時，
    # 正確寫出的文件編號也會被判核對不到，白白重寫到驗證上限
    texts = [f"{idx.docs[h.chunk.doc_no]['iso_no'] or h.chunk.doc_no} {idx.docs[h.chunk.doc_no]['title']}\n"
             f"{h.chunk.text}" for h in hits] + ([res.fulltext_text] if ft else [])
    kin = [{j for j, o in enumerate(hits) if j != i and o.chunk.doc_no == h.chunk.doc_no
            and o.chunk.pdf_sha == h.chunk.pdf_sha and h.chunk.page_start and o.chunk.page_start
            and o.chunk.page_start <= h.chunk.page_end and h.chunk.page_start <= o.chunk.page_end}
           for i, h in enumerate(hits)] + ([set()] if ft else [])
    user = f"摘錄：\n{ctx}\n\n問題：{q}"

    def left() -> float:
        return timeout - (time.monotonic() - t0)

    r = llm.complete(SYSTEM, user, timeout=timeout, on_wait=on_wait)
    text, llm_s = r.text, r.seconds
    # 自主驗證，最多 MAX_VERIFY 輪。每一輪：
    #   1. 逐行核對原文（_verify，程式比對，不花 LLM）：數字、代碼、人名要出現在該行引用的摘錄
    #   2. 原文都對得上，再請 LLM 自我檢查每一行是不是在回答問題（_off_topic）
    # 部分行答非所問 -> 直接刪掉（不必重寫）；有核對不到的字詞、或整個回答答非所問 -> 附上原因重寫，再驗下一輪。
    # 第 MAX_VERIFY 輪還不過：核對不到的行標出來，整個答非所問就改說找不到。
    log: list[str] = []
    first = None
    bad: dict[int, list[str]] = {}
    for rnd in range(1, MAX_VERIFY + 1):
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        text = "\n".join(lines)
        bad = _verify(text, texts, q, kin)
        first = bad if first is None else first
        off: list[int] = []
        asked = ""
        if not bad and not _unspecified(lines) and left() > MIN_CALL_S:
            try:
                off, s, asked = _off_topic(q, lines, timeout=left(), on_wait=on_wait, log=log)
                llm_s += s
            except Exception as e:  # noqa: BLE001 -- 檢查失敗不擋回答
                log.append(f"第 {rnd} 輪：答題檢查失敗（{type(e).__name__}）")
        if off and len(off) < len(lines):
            log.append(f"第 {rnd} 輪：刪掉答非所問的行：{'；'.join(lines[i] for i in off)}")
            text = "\n".join(ln for i, ln in enumerate(lines) if i not in off)
            off = []
        if not bad and not off:
            log.append(f"第 {rnd} 輪：通過")
            break
        log.append(f"第 {rnd} 輪：" + ("核對不到原文 " + str(bad) if bad else "整個回答答非所問"))

        def settle() -> str:
            """不再重寫時用目前這版收尾：核對不到的標出來；整個答非所問就說找不到。"""
            if bad:
                return "\n".join(ln + (f"（核對不到原文：{'、'.join(bad[i])}）" if i in bad else "")
                                 for i, ln in enumerate(lines))
            return "文件未規定。"
        if rnd == MAX_VERIFY or left() < MIN_CALL_S:
            text = settle()
            break
        if bad:
            detail = "\n".join(f"「{lines[i].strip()}」核對不到：{'、'.join(b)}" for i, b in bad.items())
            fix = (f"{user}\n\n你上一版的回答：\n{text}\n\n以下各行寫了它所引用的摘錄裡沒有的字詞：\n{detail}\n"
                   "請重寫：每一行只寫該行引用的摘錄原文裡有的內容，人名、單位、數字照原文；"
                   "摘錄沒寫的就不要寫，也不要把某一欄的人名配到別的職稱。")
        else:
            fix = (f"{user}\n\n你上一版的回答：\n{text}\n\n審查結果：這個回答沒有回答問題，或夾帶了問題沒要的資訊。\n"
                   + (f"審查員認為問題要的是：{asked}\n" if asked else "")
                   + "請重寫：只回答問題要的那一件事；摘錄完全沒寫才回答「文件未規定」。")
        try:
            r2 = llm.complete(SYSTEM, fix, timeout=left(), on_wait=on_wait)
        except Exception as e:  # noqa: BLE001 -- 重寫失敗（gateway 中途斷線、逾時）：不丟掉已經有的回答
            log.append(f"第 {rnd} 輪：重寫失敗（{type(e).__name__}），用目前這版收尾")
            text = settle()
            break
        llm_s += r2.seconds
        text = r2.text
    body = [ln for ln in text.strip().splitlines() if ln.strip()]
    if _unspecified(body):
        text = "文件未規定。（已查閱的相關文件中沒有寫到這件事）"
    text = textnorm.to_traditional(text)
    text, cited = _replace_cites(text, hits, idx, fulltext=ft)
    return Answer(text, "llm", hits, llm_seconds=llm_s, grounded=not bad,
                  note=f"第一版核對不到：{first}" if first else "",
                  seconds=round(res.seconds + time.monotonic() - t0, 2), cited=cited,
                  fulltext_text=res.fulltext_text, fulltext=res.fulltext, verify_log=log)


MAX_VERIFY = 3        # 驗證上限（回答 1 次 + 重寫最多 2 次）
QUOTE_MIN = 6         # 問題跟文件有這麼長的逐字相同（_flat 後）就做全文核對；6 個字以上逐字相同幾乎不會是巧合
FULLTEXT_LINES = 30   # 全文核對最多列幾份文件給 LLM（其餘只報份數）
MIN_CALL_S = 15       # 總期限剩不到這麼多秒，就不再多問一次 LLM（直接用目前的回答收尾）

# 審查員先寫出「問題要的是什麼」再逐行判斷：比只輸出行號抓得到更多答非所問的行，
# 配合「兩次都標出才刪」幾乎不會誤刪（以評測集校準）
JUDGE ="""你是審查員。先用一句話寫出「問題要的是什麼」，再逐行判斷回答有沒有在提供這個。
答非所問、或附帶問題沒要的資訊（其他規定、流程、文件屬性、版本、日期、表單清單…）的行不合格。
同一件事在不同文件（例如不同單位）各有規定、分行列出，都算有回答。
某一行說明問題要的東西文件沒寫（文件未規定）時，其他行提供的別的資訊就是沒問到的。
同一行裡夾帶了問題沒要的資訊，該行也不合格。
輸出格式（兩行）：
問題要的是：…
不合格：行號（逗號分隔；全部合格寫「無」）"""


def _unspecified(lines: list[str]) -> bool:
    return len(lines) == 1 and "文件未規定" in lines[0]


def _off_topic(q: str, lines: list[str], *, timeout: float, on_wait=None,
               log: list[str] | None = None) -> tuple[list[int], float, str]:
    """LLM 自我檢查：回傳 (答非所問的行（0 起）, 花的秒數, 審查員寫的「問題要的是…」)。
    只看問題和回答，不看摘錄（原文核對已由 _verify 做過）。
    審查員本身也會誤判（把正確答案判成答非所問）：第一次有標出行時再獨立問一次，只採兩次都標出的行。"""
    numbered = "\n".join(f"{i}. {re.sub(r'^-\s*', '', _CITE_RE.sub('', ln).strip())}" for i, ln in enumerate(lines, 1))
    t0 = time.monotonic()
    votes = []
    asked = ""
    for _ in range(2):
        r = llm.complete(JUDGE, f"問題：{q}\n\n回答：\n{numbered}", timeout=max(1.0, timeout - (time.monotonic() - t0)),
                         on_wait=on_wait)
        m = re.search(r"不合格[:：](.*)", r.text)
        got = {int(n) - 1 for n in re.findall(r"\d+", m.group(1) if m else r.text) if 1 <= int(n) <= len(lines)}
        if not asked and (w := re.search(r"問題要的是[:：]\s*(.+)", r.text)):
            asked = w.group(1).strip()
        if log is not None:
            log.append("審查：" + " ".join(r.text.split())[:80])
        votes.append(got)
        if not got:
            break
    out = sorted(set.intersection(*votes))
    return out, time.monotonic() - t0, asked


def excerpts_only(idx: Reader, res: Answer) -> str:
    return "（LLM 未啟用，以下為最相關的原文摘錄）\n" + "\n".join(
        f"- {cite_label(idx, h.chunk)}：{h.chunk.text[:120]}" for h in res.hits[:5])


def answer(q: str, idx: Reader, a: Analysis, *, use_llm: bool = True, timeout: float = 240) -> Answer:
    res = prepare(q, idx, a)
    if res.mode != "pending":
        return res
    if not use_llm:
        res.text, res.mode = excerpts_only(idx, res), "no_llm"
        return res
    return generate(q, idx, res, timeout=timeout)
