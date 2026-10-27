"""OCR 頁面 -> 檢索段落（chunk）。

典型 ISO 文件：會審表（發行/廢止審核）、封面＋修訂履歷、本文（每頁重複表頭；條文編號 1. / 5.1 / 5.1.1）。
做法：
  - 會審表、封面各成一段（kind=issue / revision），整頁保留。
  - 本文頁去掉頁首頁尾樣板（ingest.template 從整批文件學），全部頁接成一條流，依條文編號切段，
    每段記錄起訖頁與所屬章節（例「5.作業內容 > 5.3」）；太短往後併，太長在行邊界切開（相鄰段重疊一行）。
  - 條文編號先正規化（norm_heading）：「1.0 目的」「一、範圍」「第一章」都當第一層標題。
  - 每段前面加上下文標頭給 embedding / BM25：「SOP-QA-001 樣品保存作業規範｜5.作業內容」。
表格類文件（檢驗規範、承認單、一覽表）沒有「目的/範圍/權責…」條文結構，表格裡的「2. 巡檢4H/次」會被誤認成標題。
文件類別（metadata 的 category）是表單類、或本文的 ISO 章節名稱少於 2 個 → 不找條文，每頁依行切段（kind=table，引用只標頁碼）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_CHARS = 120
MAX_CHARS = 700

ISSUE_RE = re.compile(r"會審表|會簽")
COVER_RE = re.compile(r"制\s*/?\s*修[定訂]日期|制訂部門|修訂頁次")
ISO_SECTION_RE = re.compile(r"^\s*\d{1,2}\s*[\.、．]?\s*(目的|範圍|范围|適用範圍|适用范围|權責|权责|職責|职责|定義|定义|"
                            r"作業內容|作业内容|作業程序|作业程序|作業流程|作业流程|內容|内容|程序|參考文件|参考文件|"
                            r"相關文件|相关文件|名詞|名词|流程圖|流程图)", re.M)
FORM_CLASS_RE = re.compile(r"Inspection Standard|檢驗通則|SIP|標準檢驗程序|Checklist|點檢表|Approval Sheet|承認書")
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s*[\.、．]?\s*(\S.*)$")


@dataclass
class Chunk:
    kind: str                 # body / issue / revision
    text: str
    page_start: int
    page_end: int
    section: str = ""
    clause: str = ""          # 起始條文編號，例 5.7
    headings: list[str] = field(default_factory=list)


_CN_NUM = {c: i for i, c in enumerate("一二三四五六七八九十", 1)}
_CN_HEAD_RE = re.compile(r"^(?:第\s*([一二三四五六七八九十]{1,3}|\d{1,2})\s*[章條条節节]|([一二三四五六七八九十]{1,3})\s*[、\.．])\s*")
# 「1.0 目的」「5.0流程圖」是第一層；「3.0mm」「1.05」不是
_ZERO_TAIL_RE = re.compile(r"^(\d{1,2})\.0(?:\.0)*\.?(?=\s|[㐀-鿿:：、]|$)\s*")


def _cn_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    if len(s) == 1:
        return _CN_NUM[s]
    if s[0] == "十":                                  # 十一 ~ 十九
        return 10 + _CN_NUM.get(s[1], 0)
    return _CN_NUM[s[0]] * 10 + (_CN_NUM.get(s[2], 0) if len(s) > 2 else 0)    # 二十、二十一


def norm_heading(line: str) -> str:
    """條文編號寫法統一成「1. 標題」：1.0 目的／一、範圍／一.目的／第一章 管理方針 -> 1. …；其他行原樣。"""
    t = line.strip()
    m = _CN_HEAD_RE.match(t)
    if m:
        return f"{_cn_int(m.group(1) or m.group(2))}. {t[m.end():].lstrip(':：、.． ')}"
    m = _ZERO_TAIL_RE.match(t)
    if m:
        return f"{m.group(1)}. {t[m.end():].lstrip(':：、 ')}"
    return t


def _is_heading(line: str) -> tuple[str, str] | None:
    m = HEADING_RE.match(norm_heading(line))
    if not m:
        return None
    num, title = m.group(1), m.group(2)
    # 「1次/96小時」「2022/8/30」之類不是標題：編號後面緊接的不能是數字/單位/斜線
    if re.match(r"^[\d/%:.\-]", title) or len(num) > 11:
        return None
    # ISO 文件的第一層章節不會超過 20（目的/範圍/權責/定義/作業內容…）；
    # 「50元/月」「120 天」這種表格內容不是標題
    if int(num.split(".")[0]) > 20 or re.match(r"^[元天日月年時小分個件次%％]", title):
        return None
    if "." not in num and not line.strip()[len(num):].lstrip().startswith((".", "、", "．")):
        # 「1 目的」也算，但「12 2019/04/24 ...」這種表格列不算
        if len(title) > 25 or re.search(r"\d{4}/\d", title):
            return None
    return num, title


class HeadingSeq:
    """條文編號要照順序走：第一層只能是「目前章節」到「目前 +2」（容許 OCR 漏掉一個標題）。
    第 4 章裡冒出來的「1. 課級及以上主管」「2. 品保部」是表格/清單項目，不是標題。"""

    def __init__(self, start_top: int = 0):
        self.top = start_top

    def accept(self, h: tuple[str, str] | None) -> tuple[str, str] | None:
        if not h:
            return None
        t = int(h[0].split(".")[0])
        # 第一層標題（沒有小數點）很短：「5. 作業內容」「6.作業流程:」；
        # 表格列「5 製程條件記錄表 5. 客戶特殊要求項」很長 -> 不是標題
        if "." not in h[0] and len(h[1].strip()) > 12:
            return None
        if self.top and not (self.top <= t <= self.top + 2):
            return None
        # 第一個標題不限從 1 開始：有些 PDF 只掃了後半（從「5.作業內容」開始），前面幾頁不在檔案裡
        self.top = t
        return h


def headings_in(text: str, start_clause: str = "") -> list[str]:
    seq = HeadingSeq(int(start_clause.split(".")[0]) if start_clause[:1].isdigit() else 0)
    return [h[0] for ln in text.splitlines() if (h := seq.accept(_is_heading(ln)))]


def _page_kind(text: str, idx: int) -> str:
    head = text[:400]
    if idx <= 1 and ISSUE_RE.search(head):
        return "issue"
    if idx <= 2 and COVER_RE.search(text):
        return "revision"
    return "body"


def is_heading_line(line: str) -> bool:
    return _is_heading(line) is not None


def is_table_doc(pages: list[dict], doc_class: str = "") -> bool:
    if doc_class and FORM_CLASS_RE.search(doc_class):
        return True
    body = "\n".join(norm_heading(ln) for i, p in enumerate(pages) if _page_kind(p["text"], i) == "body"
                     for ln in p["text"].splitlines())
    return len(set(ISO_SECTION_RE.findall(body))) < 2


def chunk_pages(pages: list[dict], doc_class: str = "", templates=None, sha: str = "") -> list[Chunk]:
    """templates：ingest.template.Templates（整批文件學出來的頁首頁尾）；沒給就只用這份文件自己學。"""
    if templates is None:
        from .template import Templates
        templates = Templates.learn({sha: [p["text"] for p in pages]}, is_heading_line)
    table = is_table_doc(pages, doc_class)
    chunks: list[Chunk] = []
    body: list[tuple[int, str]] = []          # (頁碼, 行)
    page_lines = {p["page"]: [ln.strip() for ln in p["text"].splitlines() if ln.strip()] for p in pages}
    for i, p in enumerate(pages):
        kind = _page_kind(p["text"], i)
        lines = page_lines[p["page"]]
        if kind != "body":
            text = "\n".join(lines)
            if text.strip():
                chunks.append(Chunk(kind, text, p["page"], p["page"],
                                    section="文件發行/廢止會審表" if kind == "issue" else "封面與修訂履歷"))
            continue
        for ln in templates.strip(sha, lines, is_heading_line):
            body.append((p["page"], ln))
    chunks.extend(_split_rows(body) if table else _split_body(body))
    return chunks


def _split_rows(body: list[tuple[int, str]]) -> list[Chunk]:
    """表格類：不找條文標題；每頁各自依行（≈表格列）切到 MAX_CHARS，相鄰段重疊一行，不跨頁。"""
    out: list[Chunk] = []
    for page in dict.fromkeys(p for p, _ in body):
        lines = [ln for p, ln in body if p == page]
        start = 0
        while start < len(lines):
            size, end = 0, start
            while end < len(lines) and (size + len(lines[end]) <= MAX_CHARS or end == start):
                size += len(lines[end])
                end += 1
            out.append(Chunk("table", "\n".join(lines[start:end]), page, page, section="表格"))
            if end >= len(lines):
                break
            start = max(end - 1, start + 1)
    return [c for c in out if len(c.text.strip()) >= 15]


def _split_body(body: list[tuple[int, str]]) -> list[Chunk]:
    # 1) 依條文標題分塊
    blocks: list[dict] = []
    top = ""
    cur = None
    seq = HeadingSeq()
    for page, ln in body:
        h = seq.accept(_is_heading(ln))
        if h and (cur is None or len(cur["lines"]) > 0):
            num, title = h
            if "." not in num:
                top = f"{num}.{title}"
            cur = {"clause": num, "section": top if "." not in num else f"{top} > {num}", "lines": [], "pages": []}
            blocks.append(cur)
        if cur is None:
            cur = {"clause": "", "section": "", "lines": [], "pages": []}
            blocks.append(cur)
        cur["lines"].append(ln)
        cur["pages"].append(page)
    # 2) 太短往後併（合併後不超過 MAX_CHARS）；跨第一層章節時章節名稱串起來
    merged: list[dict] = []
    for b in blocks:
        prev_len = len("".join(merged[-1]["lines"])) if merged else 0
        if merged and prev_len < MIN_CHARS and prev_len + len("".join(b["lines"])) <= MAX_CHARS:
            m = merged[-1]
            m["lines"] += b["lines"]
            m["pages"] += b["pages"]
            top_m, top_b = m["section"].split(" > ")[0], b["section"].split(" > ")[0]
            if top_b and top_b not in m["section"].split("、"):
                m["section"] = f"{m['section']}、{top_b}" if m["section"] else top_b
        else:
            merged.append({**b, "lines": list(b["lines"]), "pages": list(b["pages"])})
    # 3) 太長在行邊界切開，重疊一行
    out: list[Chunk] = []
    for b in merged:
        lines, pages = b["lines"], b["pages"]
        start = 0
        while start < len(lines):
            size, end = 0, start
            while end < len(lines) and (size + len(lines[end]) <= MAX_CHARS or end == start):
                size += len(lines[end])
                end += 1
            out.append(Chunk("body", "\n".join(lines[start:end]), pages[start], pages[end - 1],
                             section=b["section"], clause=b["clause"]))
            if end >= len(lines):
                break
            start = max(end - 1, start + 1)
    return [c for c in out if len(c.text.strip()) >= 15]


def doc_header(meta: dict, section: str = "") -> str:
    head = " ".join(x for x in [meta.get("iso_no") or "", _clean_title(meta)] if x)
    return f"{head}｜{section}" if section else head


def _clean_title(meta: dict) -> str:
    t = (meta.get("title") or "").strip()
    iso = meta.get("iso_no") or ""
    if iso and t.startswith(iso):
        t = t[len(iso):].lstrip(" -_")
    return t


def meta_card(meta: dict) -> str:
    """文件屬性卡：讓「生效日期/ISO體系/版本/誰制定」這類問題也能從檢索找到。"""
    rows = [("文件編號", meta.get("doc_no")), ("ISO編號", meta.get("iso_no")), ("文件名稱", meta.get("title")),
            ("系統版本", meta.get("sys_version")), ("文件版本", meta.get("doc_version")),
            ("系統狀態", meta.get("status")), ("文件類別", meta.get("category")),
            ("ISO體系", meta.get("iso_system")), ("文件類型", meta.get("doc_type")),
            ("制定部門", meta.get("dept")), ("制定人", meta.get("owner")),
            ("生效日期", meta.get("effective_date")), ("發行日期", meta.get("release_date")),
            ("到期日期", meta.get("expire_date")), ("建立人員", meta.get("created_by")),
            ("建立日期", meta.get("created_on")), ("修改人員", meta.get("modified_by")),
            ("修改日期", meta.get("modified_on")), ("編輯者", meta.get("editor")),
            ("文件說明", meta.get("description")),
            ("關聯ISO文件", "、".join(meta.get("related_iso_docs") or []) or None),
            ("附加檔案", "、".join(f["filename"] for f in meta.get("files") or []) or None)]
    return "\n".join(f"{k}：{v}" for k, v in rows if v)
