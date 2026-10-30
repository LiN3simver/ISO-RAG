"""產生答案。

- 列表題、純屬性題：直接由屬性表組答案，不呼叫 LLM（LLM 單題要十幾秒到數分鐘）。
- 內容題：編號摘錄 [1]..[n] 送 LLM，要求只用 [n] 引用；程式再把 [n] 換成
  【SOP-QA-001 p.4 5.7】，頁碼與條號是這一行實際用到的（小模型自己寫文件/頁碼容易照抄模板）。
- 全文核對：問題逐字引用原文或提到少見的代碼時，比對全部文件，結果當成一段證據附給 LLM。
- 自主驗證（generate）：程式逐行核對原文 → LLM 審查員多數決判斷切不切題 → 整節題核對有沒有漏條 →
  請 LLM 補漏、補上的行再驗一輪；不通過就附上原因重寫，最多 MAX_VERIFY 輪。
- 長表（幾百列）：分段平行篩選、各段各自驗證，再依表格順序合併（_generate_table）。
"""
from __future__ import annotations

import re
import unicodedata
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

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
5. 摘錄可能來自好幾份文件，標題括號裡是發行狀態；主題不同的文件不要採用，不同文件的規定不要混在同一行。
   名稱相同的文件（不同單位各有一份）對同一件事各有規定時，每份文件分開寫一行，行首寫文件編號
   （例如「- SOP-X-001：…」）。標「未發行」的文件可以採用（系統會自動標註，不用寫）。
6. 摘錄中有寫到（即使用詞不同，例如「需求數量」對應「可以申請幾個」）就要回答；完全沒寫才回答「文件未規定」，不要用常識補。
7. 摘錄是 OCR 文字，可能夾雜簡體字或錯字，回答時用正確的繁體字。
8. ISO 文件的固定章節是：目的、範圍、權責、定義、作業內容。問「適用哪些人／適用範圍／適用於哪裡」時，
   第一行要先寫「範圍」章節的內容，再補充作業內容裡的細項。
9. 簽名、簽核欄只回報「已簽」或「未簽」，不要寫簽名的人名（手寫字 OCR 不可靠）。
10. 標題為「全文核對」的摘錄是系統逐字比對全部文件的結果（不限前幾段）。問題問「哪些文件提到／引用／出現在哪裡」時，
   依它逐一列出文件與頁碼，引用它的編號；問題問的是內容時，不要提它。
11. 摘錄裡的選項記號是從影像判讀的：◉、■ 是已勾選，○、□ 是未勾選（例如「◉新增 ○修改」= 勾選「新增」）。
12. 標題有「文件屬性」的摘錄是系統登錄的屬性資料，其他摘錄是 PDF 原文。同一件事兩邊寫的不一樣時以 PDF 為準；
   PDF 沒寫、要用屬性資料時，寫明是「文件屬性」的哪一欄。"""


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
    sections: list = field(default_factory=list)          # 問題問的整個章節（search.Section，由 search 判定）
    tables: list = field(default_factory=list)            # 問題問的整張長表（search.Table）：分段篩選（_generate_table）
    cited_clauses: dict[int, str] = field(default_factory=dict)   # hits 索引 -> 答案實際引用的第一條條號


def page_label(pages: list[int]) -> str:
    """[4] -> p.4；[1, 2, 3] -> p.1-3；[1, 3] -> p.1、3"""
    ps = sorted(set(pages))
    if len(ps) > 1 and ps == list(range(ps[0], ps[-1] + 1)):
        return f"p.{ps[0]}-{ps[-1]}"
    return "p." + "、".join(map(str, ps))


def cite_label(idx: Reader, c, pages: list[int] | None = None, clauses: list[str] | None = None) -> str:
    """pages：只標這幾頁（答案的證據頁，見 _evidence_pages）；None = 段落的整個頁碼範圍。
    clauses：只標這幾條（答案實際用到的條，見 _evidence_clauses）；None = 段落涵蓋的條號範圍。"""
    d = idx.docs[c.doc_no]
    name = d["iso_no"] or d["doc_no"]
    if c.kind == "meta":
        return f"{name} 文件屬性"
    page = page_label(pages or list(range(c.page_start, c.page_end + 1)))
    if c.kind == "table":                 # 表格類文件沒有條文編號，只標頁碼
        return f"{name} {page} 表格"
    nums = clauses or chunker.headings_in(c.text, c.clause)
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


def released(d: dict) -> bool:
    """沒有狀態資料的文件視為已發行（只有明確標成草稿、審核中的才算未發行）。"""
    return d.get("status") in (None, "", "已發行")


def doc_head(d: dict) -> str:
    """摘錄標頭：文件編號 名稱（未發行）。LLM 要知道哪份還沒發行，答案裡才能標明。"""
    return f"{d['iso_no'] or d['doc_no']} {d['title']}" + ("" if released(d) else "（未發行）")


def _context(hits: list[Hit], idx: Reader) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        parts.append(f"[{i}] {doc_head(idx.docs[c.doc_no])}｜{cite_label(idx, c)}｜{c.section}\n{c.text}")
    return "\n\n".join(parts)


def _mark_unreleased(ans: str, hits: list[Hit], idx: Reader) -> str:
    """一行引用的文件全都未發行、LLM 卻沒寫明 -> 程式補「（未發行）」。
    整份回答好幾行都只引用同一份未發行文件（例如問整節）：最前面標一次，不要每行都標。"""
    cited = [{hits[int(x) - 1].chunk.doc_no for x in _CITE_RE.findall(ln) if 1 <= int(x) <= len(hits)}
             for ln in ans.splitlines()]
    docs = set().union(*cited) if cited else set()
    if (len(docs) == 1 and sum(1 for c in cited if c) > 1 and "未發行" not in ans
            and not released(d := idx.docs[next(iter(docs))])):
        return f"{d['iso_no'] or d['doc_no']}（未發行）：\n{ans}"
    out = []
    for ln in ans.splitlines():
        ns = [int(x) for x in _CITE_RE.findall(ln) if 1 <= int(x) <= len(hits)]
        if ns and "未發行" not in ln and not any(released(idx.docs[hits[n - 1].chunk.doc_no]) for n in ns):
            m = re.search(r"\s*(?:\[\d+\])+\s*$", ln)
            ln = ln[:m.start()] + "（未發行）" + ln[m.start():] if m else ln + "（未發行）"
        out.append(ln)
    return "\n".join(out)


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_CITE_RE = re.compile(r"\[(\d+)\]")
_LATIN_RE = re.compile(r"[a-z][a-z\-]+")
# 字母數字混合的詞（版次 A0、R2018）：LLM 會把兩欄拼成一個（「版本 A」+「版次 00」->「A0」），
# 拆開看每個字元都核對得到，整串卻是捏造的
_ALNUM_RE = re.compile(r"(?<![a-z0-9])(?=[a-z]*\d)(?=\d*[a-z])[a-z0-9]+(?![a-z0-9])")
_CJK_RUN = re.compile(f"[{textnorm.CJK}]+")
MIN_CJK_MISS = 3          # 連續 3 字以上沒根據、且原封不動出現在「別段」摘錄 -> 從別段搬來的
# 職稱後面接人名（norm 之後是簡體）：「審核：Alex」「核准 王小明」
_ROLE_PAIR_RE = re.compile(rf"(核准|审核|提出人|制定/修改|制定|制订)[\s:：]*([{textnorm.CJK}]{{2,4}}|[a-z][a-z.]+)"
                           rf"(?=[\s、，,；;。(（\[]|$)")


def _missing(body: str, src: str, every: str = "", question: str = "") -> list[str]:
    """body 裡出現、src（該行引用的摘錄）裡卻找不到的東西：
    - 數字、文件代碼、英文字：一律要出現在 src；字母數字混合的詞要整個出現（不能是拼出來的）。
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
    # 字母數字混合：原文要有一模一樣的，或原文某一個代碼去掉符號後相同（WI1001 = WI-1001）；
    # 不能比對整段的 code_key（「版本:a\n版次:00」併成 a00，A0 就成了它的一截）
    toks = None
    for w in _ALNUM_RE.findall(body_n):
        if w in flat:
            continue
        toks = toks if toks is not None else {textnorm.code_key(t) for t in re.findall(r"[a-z0-9][a-z0-9\-_./]*", src_n)}
        if not any(w in t for t in toks):
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


def _line_terms(line: str) -> tuple[set[str], set[str]]:
    """答案一行的 (數字與英文詞, 中文雙字組)，給 _evidence_pages / _evidence_clauses 計分。"""
    body = textnorm.norm(_CITE_RE.sub("", line))
    flat = body.replace(" ", "")
    return (set(_DIGITS_RE.findall(body)) | set(_LATIN_RE.findall(body)),
            {flat[i:i + 2] for i in range(len(flat) - 1) if _CJK_RUN.fullmatch(flat[i:i + 2])})


def _term_found(terms: tuple[set[str], set[str]], text: str) -> tuple[set[str], set[str]]:
    hard, grams = terms
    t = textnorm.norm(text)
    tf = t.replace(" ", "")
    return ({x for x in hard if re.search(rf"(?<![0-9a-z]){re.escape(x)}(?![0-9a-z])", t)},
            {g for g in grams if g in tf})


def _term_hits(terms: tuple[set[str], set[str]], text: str) -> tuple[int, int]:
    hard, grams = _term_found(terms, text)
    return len(hard), len(grams)


def _evidence_pages(line: str, c) -> list[int]:
    """段落跨好幾頁時，答案這一行的字詞真的出現在哪幾頁（引用只標那幾頁）。
    依逐頁快取文字計分：數字、英文代碼命中數優先，再比中文雙字組命中數，取最高分的頁；
    都沒命中（或沒有頁面快取）就整段的頁都留。"""
    pages = list(range(c.page_start, c.page_end + 1)) if c.page_start else []
    if len(pages) < 2 or not c.pdf_sha:
        return pages
    from ..ingest.extract import load_page
    terms = _line_terms(line)
    scored = []
    for p in pages:
        rec = load_page(c.pdf_sha, p)
        scored.append((_term_hits(terms, (rec.get("text") or "") if rec else ""), p))
    best = max(s for s, _ in scored)
    return [p for s, p in scored if s == best] if best > (0, 0) else pages


def _evidence_clauses(line: str, c) -> list[str]:
    """答案這一行實際用到段落裡的哪幾條（引用的條號、_crowded 用），不是整段的條號範圍。
    逐條比對這一行的字詞（數字英文詞一個抵十個雙字組），貪婪覆蓋：先取解釋最多字詞的條，
    剩下沒解釋到的字詞還有一定份量才再取下一條（相鄰兩條大半同字時，不會標成一個範圍）。
    段落只有一條、或都沒命中 -> []（用整段範圍）。"""
    if c.kind != "body":
        return []
    blocks = [(n, t) for n, _, t in chunker.clause_blocks(c.text, c.clause) if n]
    if len(blocks) < 2:
        return []
    terms = _line_terms(line)
    found = [(n, *_term_found(terms, t)) for n, t in blocks]
    left_h, left_g = set(terms[0]), set(terms[1])
    picked: list[str] = []
    first = 0
    while True:
        gain, n = max(((10 * len(h & left_h) + len(g & left_g), n) for n, h, g in found if n not in picked),
                      default=(0, ""))
        if gain <= 0 or (picked and gain < max(3, first * 0.3)):
            break
        first = first or gain
        picked.append(n)
        h, g = next((h, g) for m, h, g in found if m == n)
        left_h -= h
        left_g -= g
    order = [n for n, _ in blocks]
    return sorted(dict.fromkeys(picked), key=order.index)


def _replace_cites(ans: str, hits: list[Hit], idx: Reader,
                   fulltext: bool = False) -> tuple[str, list[tuple[int, list[int]]], dict[int, str]]:
    """[n] -> 【SOP-QA-001 p.4 5.7】：頁碼用這一行的證據頁（_evidence_pages），條號用這一行實際用到的條
    （_evidence_clauses）。fulltext：最後一段 [len(hits)+1] 是全文核對 -> 【全文核對】，cited 記成索引 len(hits)。
    回傳 (換好的答案, 引用的 (hits 索引, 頁碼), hits 索引 -> 第一個引用的條號)。"""
    cited: dict[int, list[int]] = {}
    clauses: dict[int, str] = {}
    out = []
    for ln in ans.splitlines():
        def rep(m, ln=ln):
            labels, seen = [], set()     # seen：(文件, 頁, 條)——兩段縮到同一處時只標一次
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
                    cl = _evidence_clauses(ln, c)
                    if cl:
                        clauses.setdefault(n - 1, cl[0])
                    lab = cite_label(idx, c, pages, cl or None)
                    key = (c.doc_no, c.pdf_sha, tuple(pages), tuple(cl)) if pages else lab
                    if lab not in labels and key not in seen:
                        labels.append(lab)
                        seen.add(key)
            return f"【{'；'.join(labels)}】" if labels else ""
        out.append(re.sub(r"(?:\[\d+\])+", rep, ln))
    return "\n".join(out), list(cited.items()), clauses


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
        res = (Answer("查無相關文件。", "not_found") if not hits
               else Answer("", "pending", hits, sections=a.sections, tables=a.tables))
        if hits:
            ft = _pick_fulltext(q, a, idx, hits)
            if ft:
                res.fulltext_text, res.fulltext, _ = ft
    res.seconds = round(time.monotonic() - t0, 2)
    return res


def generate(q: str, idx: Reader, res: Answer, *, timeout: float = 240, on_wait=None, note: str = "") -> Answer:
    """內容題：送 LLM、自主驗證、換引用。on_wait(秒數) 在等待期間被呼叫（給串流送心跳）。
    note：附在問題前的說明（長表分段篩選時說明這是第幾段，見 _generate_table）。"""
    if res.mode != "pending":
        return res
    if res.tables and not note:
        return _generate_table(q, idx, res, timeout=timeout, on_wait=on_wait)
    t0 = time.monotonic()
    hits = res.hits
    ft = bool(res.fulltext_text)
    ctx = _context(hits, idx) + (f"\n\n[{len(hits) + 1}] {res.fulltext_text}" if ft else "")
    # 核對用的原文要包含 LLM 看得到的摘錄標頭（文件編號、名稱、出處）：只比內文時，正確寫出的文件編號、
    # 「文件屬性」這類出處說明都會被判核對不到，白白重寫到驗證上限
    texts = [f"{doc_head(idx.docs[h.chunk.doc_no])}｜{cite_label(idx, h.chunk) if h.chunk.kind == 'meta' else 'PDF'}\n"
             f"{h.chunk.text}" for h in hits] + ([res.fulltext_text] if ft else [])
    kin = [{j for j, o in enumerate(hits) if j != i and o.chunk.doc_no == h.chunk.doc_no
            and o.chunk.pdf_sha == h.chunk.pdf_sha and h.chunk.page_start and o.chunk.page_start
            and o.chunk.page_start <= h.chunk.page_end and h.chunk.page_start <= o.chunk.page_end}
           for i, h in enumerate(hits)] + ([set()] if ft else [])
    scope = _section_scope(idx, res.sections)
    user = (f"摘錄：\n{ctx}\n\n" + (f"{scope}{_section_layout(res.sections)}\n\n" if scope else "")
            + (f"{note}\n\n" if note else "") + f"問題：{q}")

    def left() -> float:
        return timeout - (time.monotonic() - t0)

    r = llm.complete(SYSTEM, user, timeout=timeout, on_wait=on_wait, temperature=ANSWER_TEMPERATURE)
    text, llm_s = r.text, r.seconds
    # 自主驗證，最多 MAX_VERIFY 輪。每一輪：
    #   1. 逐行核對原文（_verify，程式比對，不花 LLM）：數字、代碼、人名要出現在該行引用的摘錄
    #   2. 只複述問題或標頭、沒有內容的行（_echo_lines，程式檢查）
    #   3. 原文都對得上，再請 LLM 審查員判斷每一行是不是在回答問題（_off_topic，多數決）
    #   4. 問的是整個章節（res.sections）：程式核對節內每一條都有寫到（_uncovered）、而且一條一行（_crowded）
    #   5. 都過了，請 LLM 對照摘錄找漏掉的（_fill_missing）；有補 -> 下一輪再從 1 驗起
    # 部分行答非所問 -> 直接刪掉（不必重寫）；有核對不到的字詞、整個回答答非所問、漏了條、或好幾條擠在一行
    # -> 附上原因重寫，再驗下一輪。第 MAX_VERIFY 輪還不過：核對不到的行標出來。
    log: list[str] = []
    filled = False
    first = None
    bad: dict[int, list[str]] = {}
    for rnd in range(1, MAX_VERIFY + 1):
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        text = "\n".join(lines)
        bad = _verify(text, texts, q, kin)
        first = bad if first is None else first
        off: list[int] = []
        asked = ""
        echo = [] if bad or _unspecified(lines) else _echo_lines(lines, texts, q, kin)
        if echo and len(echo) < len(lines):
            log.append(f"第 {rnd} 輪：刪掉只複述標頭的行：{'；'.join(lines[i] for i in echo)}")
            lines = [ln for i, ln in enumerate(lines) if i not in echo]
            text = "\n".join(lines)
            echo = []
        if echo:
            off = list(range(len(lines)))
        elif not bad and not _unspecified(lines) and left() > MIN_CALL_S:
            try:
                off, s, asked = _off_topic(q, lines, timeout=left(), on_wait=on_wait, log=log,
                                           scope=scope or (TABLE_JUDGE_NOTE if note else ""))
                llm_s += s
            except Exception as e:  # noqa: BLE001 -- 檢查失敗不擋回答
                log.append(f"第 {rnd} 輪：答題檢查失敗（{type(e).__name__}）")
        if off and len(off) < len(lines):
            log.append(f"第 {rnd} 輪：刪掉答非所問的行：{'；'.join(lines[i] for i in off)}")
            text = "\n".join(ln for i, ln in enumerate(lines) if i not in off)
            off = []
        miss = [] if bad or off else _uncovered(text, res.sections)
        crowd = [] if bad or off or miss else _crowded(text, hits, res.sections)
        if not bad and not off and not miss and not crowd:
            # 寫的都對、都是問到的，再查有沒有漏：補上的行下一輪照樣逐行核對、審查（長表的一段在迴圈後補漏）
            if not note and not filled and rnd < MAX_VERIFY and left() > MIN_CALL_S:
                filled = True
                try:
                    text2, added, s = _fill_missing(user, text, texts, q, kin, timeout=left(), on_wait=on_wait,
                                                    rows=False)
                    llm_s += s
                except Exception as e:  # noqa: BLE001 -- 補漏失敗不擋回答
                    log.append(f"第 {rnd} 輪：補漏失敗（{type(e).__name__}）")
                    added = 0
                if added:
                    log.append(f"第 {rnd} 輪：補上漏掉的 {added} 行，再驗一輪")
                    text = text2
                    continue
            log.append(f"第 {rnd} 輪：通過")
            break
        log.append(f"第 {rnd} 輪：" + ("核對不到原文 " + str(bad) if bad else
                                      "只複述了標頭，沒有內容" if echo else "整個回答答非所問" if off else
                                      "漏了章節條文：" + "、".join(miss) if miss else "；".join(crowd)))

        def settle() -> str:
            """不再重寫時用目前這版收尾：核對不到的標出來；其他情況（審查員判整個答非所問、漏條、排列不對）照目前這版。
            每行都核對得到原文時不改說「文件未規定」：原文核對是程式逐字比的，這版寫的確實是文件內容；
            審查員看不到原文，可能拿模型自己「記得」的說法否定正確答案。"""
            if bad:
                return "\n".join(ln + (f"（核對不到原文：{'、'.join(bad[i])}）" if i in bad else "")
                                 for i, ln in enumerate(lines))
            return text
        if rnd == MAX_VERIFY or left() < MIN_CALL_S:
            if off and not bad:
                log.append("審查員判整個答非所問，但每行都核對得到原文：保留這版")
            text = settle()
            break
        if bad:
            detail = "\n".join(f"「{lines[i].strip()}」核對不到：{'、'.join(b)}" for i, b in bad.items())
            fix = (f"{user}\n\n你上一版的回答：\n{text}\n\n以下各行寫了它所引用的摘錄裡沒有的字詞：\n{detail}\n"
                   "請重寫：每一行只寫該行引用的摘錄原文裡有的內容，人名、單位、數字照原文；"
                   "摘錄沒寫的就不要寫，也不要把某一欄的人名配到別的職稱。")
        elif miss or crowd:
            why = f"上一版漏了這幾條：{'、'.join(miss)}" if miss else f"上一版{'；'.join(crowd)}"
            fix = (f"{user}\n\n你上一版的回答：\n{text}\n\n問題問的是整個章節，{why}\n"
                   f"請重寫：{_section_layout(res.sections)}照摘錄原文寫，行尾標出摘錄編號。")
        elif echo:
            fix = (f"{user}\n\n你上一版的回答：\n{text}\n\n上一版只寫了文件編號、名稱或發行狀態，沒有寫出問題要的內容。\n"
                   "請重寫：寫出摘錄原文裡回答這個問題的內容；摘錄完全沒寫才回答「文件未規定」。")
        else:
            fix = (f"{user}\n\n你上一版的回答：\n{text}\n\n審查結果：這個回答沒有回答問題，或夾帶了問題沒要的資訊。\n"
                   + (f"審查員認為問題要的是：{asked}\n" if asked else "")
                   + "請重寫：只回答問題要的那一件事；摘錄完全沒寫才回答「文件未規定」。")
        try:
            r2 = llm.complete(SYSTEM, fix, timeout=left(), on_wait=on_wait, temperature=ANSWER_TEMPERATURE)
        except Exception as e:  # noqa: BLE001 -- 重寫失敗（gateway 中途斷線、逾時）：不丟掉已經有的回答
            log.append(f"第 {rnd} 輪：重寫失敗（{type(e).__name__}），用目前這版收尾")
            text = settle()
            break
        llm_s += r2.seconds
        text = r2.text
    if note and not bad and left() > MIN_CALL_S:        # 長表的一段：補漏（逐列再檢查一次）
        try:
            text, added, s = _fill_missing(user, text, texts, q, kin, timeout=left(), on_wait=on_wait)
            llm_s += s
            log.append(f"補漏：補上 {added} 列")
        except Exception as e:  # noqa: BLE001 -- 補漏失敗不擋回答
            log.append(f"補漏失敗（{type(e).__name__}）")
    body = [ln for ln in text.strip().splitlines() if ln.strip()]
    if note and not _unspecified(body):                  # 長表的一段：每一列都要引用摘錄，沒引用的是雜訊（照抄問題等）
        body = [ln for ln in body if _CITE_RE.search(ln)] or ["文件未規定"]
        text = "\n".join(body)
    if _unspecified(body):
        text = "文件未規定。（已查閱的相關文件中沒有寫到這件事）"
    text = textnorm.to_traditional(text)
    if res.sections:          # 整節題照 PDF 排列：條號開頭，不加「- 」
        text = re.sub(r"(?m)^-\s*(?=\d+(?:\.\d+)+\s)", "", text)
    text = _mark_unreleased(text, hits, idx)
    text, cited, clauses = _replace_cites(text, hits, idx, fulltext=ft)
    return Answer(text, "llm", hits, llm_seconds=llm_s, grounded=not bad, sections=res.sections,
                  cited_clauses=clauses,
                  note=f"第一版核對不到：{first}" if first else "",
                  seconds=round(res.seconds + time.monotonic() - t0, 2), cited=cited,
                  fulltext_text=res.fulltext_text, fulltext=res.fulltext, verify_log=log)


# ---------------------------------------------------------------- 只複述標頭、補漏
DUP_COVER = 0.8       # 補上的一行，依據的原文有這麼多已經被回答涵蓋 -> 算重複


def _cited(line: str, n: int, kin: list[set[int]]) -> list[int]:
    """這一行引用的摘錄（0 起），加上同一頁的摘錄（kin，同 _verify）。"""
    ks = {int(x) - 1 for x in _CITE_RE.findall(line) if 1 <= int(x) <= n}
    return sorted(ks | {j for k in ks for j in kin[k]})


def _strip_common(a: str, b: str, n: int = 2) -> str:
    """從 a 反覆拿掉跟 b 最長的相同片段（≥ n 字），剩下的是 b 沒有的部分。"""
    while a:
        m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
        if m.size < n:
            break
        a = a[:m.a] + a[m.a + m.size:]
    return a


_ALNUM_TOK_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-_/][A-Za-z0-9]+)*")


def _echo_lines(lines: list[str], texts: list[str], q: str, kin: list[set[int]]) -> list[int]:
    """只複述問題或所引用摘錄的標頭（文件編號、名稱、未發行、文件屬性）、沒有別的內容的行（0 起）。
    英數字整個詞比對（「3.2」不是「SOP-QA-032」的一截），中文扣掉跟問題、標頭相同的片段；
    沒有新的英數字詞、中文剩不到 2 個字 -> 沒有回答任何東西。
    審查員只看切不切題，看不出「SOP-QA-009：未發行」其實什麼都沒答，所以由程式檢查。"""
    from ..query.search import _word

    def alnum(s: str) -> set[str]:
        return {_word(t) for t in _ALNUM_TOK_RE.findall(s)}

    def cjk(s: str) -> str:
        return "".join(_CJK_RUN.findall(_word(s)))
    out = []
    for i, ln in enumerate(lines):
        ks = _cited(ln, len(texts), kin) or list(range(len(texts)))
        refs = [q] + [texts[k].split("\n", 1)[0] for k in ks]
        body = _CITE_RE.sub("", ln)
        new = alnum(body) - set().union(*(alnum(r) for r in refs))
        rest = cjk(body)
        for r in refs:
            rest = _strip_common(rest, cjk(r))
        if not new and len(rest) < 2:
            out.append(i)
    return out


def _source_span(line: str, src: str) -> set[int]:
    """這一行的內容對到原文的哪些位置（都先 _word；連續 3 字以上相同的片段才算，零星的兩字詞不算）。"""
    from ..query.search import _word
    a, b = _word(_CITE_RE.sub("", line)), _word(src)
    return {j for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if m.size >= 3
            for j in range(m.b, m.b + m.size)}


def _fill_missing(user: str, text: str, texts: list[str], q: str, kin: list[set[int]], *,
                  timeout: float, on_wait=None, rows: bool = True) -> tuple[str, int, float]:
    """答完、驗證過之後，請 LLM 拿摘錄對照自己的回答，把問題要的、卻漏掉的補上（清單、表格常只答一部分）。
    補上的行一樣要逐行核對原文（_verify），核對不到的、跟已寫的重複的不收。回傳 (新回答, 補了幾行, 秒數)。
    rows：長表的一段，逐列對照；否則是一般內容題。"""
    from ..query.search import _word, common_len
    lines = [] if _unspecified([ln for ln in text.splitlines() if ln.strip()]) else \
        [ln for ln in text.splitlines() if ln.strip()]
    ask = ("請逐列對照摘錄的每一列：符合問題、但上面回答沒有列出的列，照同樣格式補列出來，"
           "行尾標出摘錄編號；沒有漏的就只回答「無」。" if rows else
           "請對照摘錄檢查上面的回答有沒有漏：問題要的內容裡，摘錄有寫、但回答沒寫到的（例如清單或表格只寫了一部分），"
           "照同樣格式補寫成新的行，行尾標出摘錄編號；不要重寫已經有的行，不要補問題沒要的資訊；沒有漏的就只回答「無」。")
    r = llm.complete(SYSTEM, f"{user}\n\n你的回答：\n{chr(10).join(lines) or '（沒有列出任何一列）'}\n\n{ask}",
                     timeout=timeout, on_wait=on_wait, temperature=ANSWER_TEMPERATURE)
    have = {_word(_CITE_RE.sub("", ln)) for ln in lines}
    body = "".join(have)

    def known(ln: str) -> bool:
        w = _word(_CITE_RE.sub("", ln))
        if w in have or rows:
            return w in have
        # 一般內容題：依據的原文已經被回答涵蓋 -> 是換句話重寫，不是漏的。
        # 比的是原文位置不是字面：兩列字面幾乎一樣（「組立課…1次/96小時」「射出課…1次/96小時」）卻是原文不同的列
        ks = _cited(ln, len(texts), kin)
        span = {(k, j) for k in ks for j in _source_span(ln, texts[k])}
        if not span:                # 對不到原文（核對會再擋）：退回比字面
            return common_len(w, body) >= 0.8 * len(w)
        cover = {(k, j) for k in ks for old in lines for j in _source_span(old, texts[k])}
        return len(span & cover) >= DUP_COVER * len(span)
    new = [ln.strip() for ln in r.text.splitlines() if _CITE_RE.search(ln) and not known(ln)]
    if new:
        bad = _verify("\n".join(new), texts, q, kin)
        new = [ln for i, ln in enumerate(new) if i not in bad]
    return "\n".join(lines + new) or text, len(new), r.seconds


# ---------------------------------------------------------------- 長表分段篩選
TABLE_JUDGE_NOTE = ("這個問題是從一張表裡篩選出符合的列，回答的每一行是表格的一列；符合的每一列都是問題要的。"
                    "一格裡有好幾個值（例如「I/III」）時，只要其中一個是問題要的就算符合")
TABLE_WORKERS = 6         # 長表分段篩選同時跑幾段
TABLE_BATCH_CHARS = 2000  # 長表每段多長：一段太長（幾十列）時 LLM 只挑出少數幾列，漏掉大半


def _generate_table(q: str, idx: Reader, res: Answer, *, timeout: float, on_wait=None) -> Answer:
    """問整張長表（search 判定的 res.tables）：表格切成小段，每段各自走一次 generate（同一套逐行核對、
    審查、最多 MAX_VERIFY 輪），只列這一段裡符合問題的列；平行執行後依表格順序合併、去掉重複列（相鄰段重疊一行）。
    一次把幾百列交給 LLM 篩選，它只會挑出前面幾列。"""
    from concurrent.futures import ThreadPoolExecutor

    from ..query.search import _word
    t0 = time.monotonic()
    table = res.tables[0]
    batches: list[list] = [[]]
    for cid in table.ids:
        c = idx.chunks[cid]
        if batches[-1] and sum(len(x.text) for x in batches[-1]) + len(c.text) > TABLE_BATCH_CHARS:
            batches.append([])
        batches[-1].append(c)
    label = table.label(idx)
    cols = (f"照表頭欄位順序（{table.head}）用「｜」分隔，第一格是這一列的編號，不要在格子前面加欄位名稱，"
            if table.head else "")

    def run(k: int, batch: list) -> Answer:
        note = (f"這個問題問的是整張表「{label}」。表很長，分 {len(batches)} 段給你，以下摘錄是第 {k + 1} 段。"
                "只根據這一段，逐列檢查每一列，符合問題的列全部列出，一列都不要漏；有幾列就寫幾列，不受「1～3 行」限制。"
                "一格裡有好幾個值（例如「I/III」）時，只要其中一個是問題要的就算符合。"
                f"每一列寫成一行，{cols}照原文寫（OCR 明顯的辨識錯字，例如 I 被辨識成 l 或 1，寫成正確的字），"
                "行尾標出摘錄編號。這一段沒有符合的列，就只回答「文件未規定」。")
        sub = Answer("", "pending", [Hit(c, 1.0, ["table"]) for c in batch])
        return generate(q, idx, sub, timeout=max(1.0, timeout - (time.monotonic() - t0)), on_wait=on_wait,
                        note=note)

    with ThreadPoolExecutor(TABLE_WORKERS) as ex:
        outs = list(ex.map(run, range(len(batches)), batches))
    hits: list[Hit] = []
    lines: list[str] = []
    seen: set[str] = {_word(table.head)}       # LLM 有時把表頭當成一列抄出來
    cited: list[tuple[int, list[int]]] = []
    clauses: dict[int, str] = {}
    log: list[str] = []
    for k, o in enumerate(outs):
        base = len(hits)
        hits += o.hits
        log += [f"第 {k + 1}/{len(outs)} 段：{x}" for x in o.verify_log]
        if o.mode != "llm" or o.text.startswith("文件未規定"):
            continue
        for ln in o.text.splitlines():
            ln = re.sub(r"\s*[|｜]\s*", "｜", re.sub(r"^\s*-\s*", "", ln)).strip()    # 各段寫法統一成表格列
            if ln and (key := _word(re.sub(r"【[^】]*】", "", ln))) not in seen:
                seen.add(key)
                lines.append(ln)
        cited += [(base + i, p) for i, p in o.cited]
        clauses.update({base + i: cl for i, cl in o.cited_clauses.items()})
    text = "\n".join(lines) or "文件未規定。（已查閱的相關文件中沒有寫到這件事）"
    return Answer(text, "llm", hits, llm_seconds=sum(o.llm_seconds for o in outs),
                  grounded=all(o.grounded is not False for o in outs), tables=res.tables,
                  note="；".join(o.note for o in outs if o.note), cited=cited, cited_clauses=clauses,
                  seconds=round(res.seconds + time.monotonic() - t0, 2), verify_log=log)


MAX_VERIFY = 3        # 驗證上限（回答 1 次 + 重寫最多 2 次）
QUOTE_MIN = 6         # 問題跟文件有這麼長的逐字相同（_flat 後）就做全文核對；6 個字以上逐字相同幾乎不會是巧合
FULLTEXT_LINES = 30   # 全文核對最多列幾份文件給 LLM（其餘只報份數）
MIN_CALL_S = 15       # 總期限剩不到這麼多秒，就不再多問一次 LLM（直接用目前的回答收尾）
# 回答、重寫、補漏用 0：同一題每次一樣（預設溫度下同一請求每次寫法不同，有時會漏掉一份文件）。
# 審查員不設：多數決要的是彼此獨立的票
ANSWER_TEMPERATURE = 0.0

# 審查員先寫出「問題要的是什麼」再逐行判斷：比只輸出行號抓得到更多答非所問的行。
# 數值遮成 #：審查員看不到原文，會拿模型自己「記得」的數字否定正確答案；數字對不對已由 _verify 逐字核對。
# 以評測集校準：只遮數值（不遮代碼）＋多數決，誤刪最少
JUDGE = """你是審查員。先用一句話寫出「問題要的是什麼」，再逐行判斷回答有沒有在提供這個。
回答裡的數字已遮成 #（數字由程式逐字核對過原文，不用管數字是多少）。
答非所問、或附帶問題沒要的資訊（其他規定、流程、文件屬性、版本、日期、表單清單…）的行不合格。
同一件事在不同文件（例如不同單位）各有規定、分行列出，都算有回答。
某一行說明問題要的東西文件沒寫（文件未規定）時，其他行提供的別的資訊就是沒問到的。
同一行裡夾帶了問題沒要的資訊，該行也不合格。
輸出格式（兩行）：
問題要的是：…
不合格：行號（逗號分隔；全部合格寫「無」）"""


def _unspecified(lines: list[str]) -> bool:
    return len(lines) == 1 and "文件未規定" in lines[0]


# ---------------------------------------------------------------- 整節題
def _section_scope(idx: Reader, sections: list) -> str:
    """問題問的整個章節（search 依文件標題結構判定）-> 給答題與審查員的一句說明；沒有 -> ""。
    審查員只看問題時會把「抽樣計畫作業指導書的定義」讀成「抽樣計畫的定義」，把節裡其他條當成沒問到刪掉。"""
    if not sections:
        return ""
    parts = [s.label(idx) + (f"（{'、'.join(f'{n} {t}' for n, t in s.items)}）" if s.items else "") for s in sections]
    return f"這個問題問的是文件的整個章節：{'；'.join(parts)}，該節的每一條都是問題要的。"


def _section_layout(sections: list) -> str:
    """問整個章節時的排列要求：跟 PDF 一樣一條一行、行首寫條號。"""
    eg = next((f"（例如「{n} {t}：…」）" for s in sections for n, t in s.items[:1]), "")
    return f"請照 PDF 原文的排列逐條列出這一節：一條一行，行首寫條號{eg}，不要把好幾條併成一行。"


def _crowded(text: str, hits: list[Hit], sections: list) -> list[str]:
    """問整個章節時，哪幾行把節內好幾條擠在同一行（程式核對：這一行實際用到哪幾條，同 _evidence_clauses）。"""
    nums = {n for s in sections for n, _ in s.items}
    if not nums:
        return []
    out = []
    for ln in text.splitlines():
        used: list[str] = []
        for k in dict.fromkeys(int(x) for x in _CITE_RE.findall(ln)):
            if 1 <= k <= len(hits):
                used += [n for n in _evidence_clauses(ln, hits[k - 1].chunk) if n in nums and n not in used]
        if len(used) >= 2:
            out.append(f"「{_CITE_RE.sub('', ln).strip()[:20]}…」把 {'、'.join(used)} 擠在同一行")
    return out


def _uncovered(text: str, sections: list) -> list[str]:
    """問整個章節時，節內哪幾條答案沒寫到（程式核對，不花 LLM）：條文名稱（含括號裡的縮寫，如 CR）
    有連續 min(長度, 4) 個字出現在答案裡就算寫到。"""
    from ..query.search import _word, common_len
    body = _word(_CITE_RE.sub("", text))
    miss = []
    for s in sections:
        for num, name in s.items:
            parts = [_word(p) for p in [name, re.sub(r"[(（][^)）]*[)）]", "", name),
                                        *re.findall(r"[(（]([^)）]+)[)）]", name)]]
            if any(common_len(p, body) >= min(len(p), 4) for p in parts if p):
                continue
            miss.append(f"{num} {name}")
    return miss


# ---------------------------------------------------------------- 審查員
def _off_topic(q: str, lines: list[str], *, timeout: float, on_wait=None,
               log: list[str] | None = None, scope: str = "") -> tuple[list[int], float, str]:
    """LLM 自我檢查：回傳 (答非所問的行（0 起）, 花的秒數, 審查員寫的「問題要的是…」)。
    只看問題和回答，不看摘錄（原文核對已由 _verify 做過）；scope：問題問的整個章節（_section_scope）。
    審查員本身也會誤判，所以用多數決（majority）。"""
    t0 = time.monotonic()
    asked = ""

    def once() -> set[int]:
        nonlocal asked
        got, w, raw = judge_once(q, lines, scope=scope, timeout=max(1.0, timeout - (time.monotonic() - t0)),
                                 on_wait=on_wait)
        asked = asked or w
        if log is not None:
            log.append("審查：" + " ".join(raw.split())[-80:])
        return got
    return majority(once), time.monotonic() - t0, asked


def majority(vote) -> list[int]:
    """審查員投票：第一票沒標 -> 全部合格；前兩票一樣 -> 照它；不一樣才投第三票，兩票以上標出的行才刪。
    比「兩票都要標才刪」更不怕審查員單次不穩定，又只在前兩票不一致時多問一次。"""
    votes = [vote()]
    if not votes[0]:
        return []
    votes.append(vote())
    if votes[0] != votes[1]:
        votes.append(vote())
    return sorted(i for i in set().union(*votes) if sum(i in v for v in votes) >= 2)


# 數值（96、1.1、2,000、100pcs 的 100），不含代碼裡的（SIP-QA101-A1、SOP-QA-001：前面是字母或連字號）
_VALUE_RE = re.compile(r"(?<![A-Za-z0-9\-_])\d+(?:[.,]\d+)*(?![0-9\-_])")


def judge_once(q: str, lines: list[str], *, prompt: str = "", scope: str = "", timeout: float,
               on_wait=None, mask: bool = True) -> tuple[set[int], str, str]:
    """審查員判一次：(不合格的行（0 起）, 「問題要的是…」, 審查員原文)。
    mask：回答裡的數值遮成 #。審查員要判的只是切不切題，用不到數字；給它看數字，它會拿自己「記得」的值
    否定正確答案。代碼裡的數字不遮（「SIP-QA#-A#」認不出是表單編號）。"""
    def show(ln: str) -> str:
        t = re.sub(r"^-\s*", "", _CITE_RE.sub("", ln).strip())
        return _VALUE_RE.sub("#", t) if mask else t
    numbered = "\n".join(f"{i}. {show(ln)}" for i, ln in enumerate(lines, 1))
    r = llm.complete(prompt or JUDGE, f"問題：{q}\n" + (f"（{scope}）\n" if scope else "") + f"\n回答：\n{numbered}",
                     timeout=timeout, on_wait=on_wait)
    # 取最後一個「不合格：」行（逐行寫理由時前面會有「第 2 行：不合格，…」），只取開頭的行號清單：
    # 審查員常在後面加說明（「8,9（第 8 行…版次1.2…）」），說明裡的數字不是行號
    m = re.findall(r"(?m)^\s*不合格[:：](.*)", r.text) or re.findall(r"不合格[:：](.*)", r.text)
    # 照抄格式說明的「行號（1）」「第 2 行」也要讀得到
    head = re.sub(r"^[\s(（]*(?:行號|第)?[\s(（:：]*", "", m[-1] if m else r.text)
    nums = re.match(r"[\s\d,，、和及]*", head).group(0)
    got = {int(n) - 1 for n in re.findall(r"\d+", nums) if 1 <= int(n) <= len(lines)}
    w = re.search(r"問題要的是[:：]\s*(.+)", r.text)
    return got, w.group(1).strip() if w else "", r.text


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
