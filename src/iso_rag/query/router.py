"""規則式問題分析（不呼叫 LLM）：文件代碼、屬性條件、意圖。

意圖：
  list       「ISO 14001 的三階管理辦法有哪些」 -> 直接查屬性表
  attribute  「SOP-QA-001 的生效日期？」        -> 直接回屬性（若問題只問屬性）
  content    其他                                -> 檢索 + LLM
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..index import textnorm
from ..index.store import Reader

LIST_RE = re.compile(r"有哪些|有那些|列出|清單|列表|幾份|多少份|哪幾份|哪些文件|所有.*文件|全部.*文件")
# 屬性詞 -> metadata 欄位
ATTRS: list[tuple[str, str, str]] = [
    (r"生效(日期|時間)?|何時生效|什麼時候生效", "effective_date", "生效日期"),
    (r"發行日期|何時發行|什麼時候發行", "release_date", "發行日期"),
    (r"到期", "expire_date", "到期日期"),
    (r"系統版本", "sys_version", "系統版本"),
    (r"文件版本|版本|版次|第幾版", "doc_version", "文件版本"),
    (r"狀態|是否(已)?發行|有沒有發行", "status", "系統狀態"),
    (r"ISO\s*體系|哪個體系|體系", "iso_system", "ISO體系"),
    (r"文件類型|幾階|哪一階|哪階", "doc_type", "文件類型"),
    (r"文件類別|類別", "category", "文件類別"),
    (r"制[定訂]部門|哪個部門(制[定訂]|負責)|權責部門", "dept", "制定部門"),
    (r"制[定訂]人|誰(制[定訂]|寫)的", "owner", "制定人"),
    (r"建立(人員|者)|誰建立", "created_by", "建立人員"),
    (r"建立日期", "created_on", "建立日期"),
    (r"修改(人員|者)|誰修改", "modified_by", "修改人員"),
    (r"修改日期|最後修改", "modified_on", "修改日期"),
    (r"編輯者", "editor", "編輯者"),
    (r"文件編號", "doc_no", "文件編號"),
    (r"ISO\s*編號", "iso_no", "ISO編號"),
    (r"文件名稱|叫什麼|名稱", "title", "文件名稱"),
    (r"附件|附檔|檔案名稱", "files", "附加檔案"),
    (r"關聯(ISO)?文件", "related_iso_docs", "關聯ISO文件"),
]
# 明講階層的才當篩選條件；「程序/辦法/表單」在內容題裡常是一般名詞（「登記表的表單編號」），
# 只有列表題才拿來篩。
DOC_TYPES_STRONG = {"一階": "一階手冊", "二階": "二階程序", "三階": "三階管理辦法", "四階": "四階表單"}
# 「外來文件」在內容題裡也常是一般名詞（「外來文件轉換的條件」「外來文件管理辦法」），拿來篩會把答案所在的
# 文件篩掉；只在列表題當篩選條件
DOC_TYPES_WEAK = {"外來文件": "外來文件&其他", "手冊": "一階手冊", "程序": "二階程序", "管理辦法": "三階管理辦法",
                  "辦法": "三階管理辦法", "表單": "四階表單"}
# 只用在列表題（「是否已發行」這類屬性題不能當篩選）
STATES: list[tuple[str, str]] = [(r"未發行|新建立|草稿|尚未發行", "新建立"), (r"已發行", "已發行"), (r"審核中|簽核中", "審核中")]
FILLER_RE = re.compile(r"[的是為嗎呢？?。,，、\s请請問告訴我一下多少什麼哪個何時幾]|"
                       r"這份|此份|文件|規範|作業|辦法|程序|是什麼|是多少|為何")


@dataclass
class Analysis:
    query: str
    intent: str
    codes: list[str] = field(default_factory=list)          # 正規化後的代碼 key
    doc_nos: list[str] = field(default_factory=list)        # 代碼對應到的文件編號
    attrs: list[tuple[str, str]] = field(default_factory=list)   # (欄位, 中文名)
    iso_system: str | None = None                           # 例 "14001"
    doc_type: str | None = None
    attr_only: bool = False
    status: str | None = None                                # 列表題指定的狀態
    focus_docs: list[str] = field(default_factory=list)     # 問題明確指向的文件（只檢索這些）
    focus_term: str = ""
    focused: bool = False                                    # search() 是否真的鎖定了焦點文件
    sections: list = field(default_factory=list)            # search() 判定問題問的整個章節（search.Section）
    tables: list = field(default_factory=list)              # search() 判定問題問的整張長表（search.Table）


# ---------------------------------------------------------------- 焦點文件
# 標題裡的通用詞：拿掉之後剩下的才是能「指名」一份文件的詞（ORT、量測設備、教育訓練…）
_GENERIC_TAIL = ["控制程序", "管理程序", "作業程序", "管理辦法", "作業辦法", "作業規範", "管理手冊", "程序", "辦法",
                 "規範", "手冊", "管理", "控制", "作業", "過程", "一覽表", "清單"]
_GENERIC_ASCII = {"sop", "iso", "sip", "pdf"}
_focus_cache: dict[str, dict[str, set[str]]] = {}


def _title_terms(d: dict) -> set[str]:
    t = textnorm.norm(d["title"] or "")
    if d.get("iso_no"):
        t = t.replace(textnorm.norm(d["iso_no"]), " ")
    t = re.sub(r"[-_]?[a-z]\d{1,2}(版\S*)?\b|\d{6,8}|[_\-]", " ", t)       # 版本、日期
    terms = {w for w in re.findall(r"[a-z]{2,}", t) if w not in _GENERIC_ASCII}
    for run in re.findall(f"[{textnorm.CJK}]+", t):
        core = run
        changed = True
        while changed:                       # 反覆去掉結尾通用詞：量測設備管理過程控制程序 -> 量測設備
            changed = False
            for g in _GENERIC_TAIL:
                gn = textnorm.norm(g)
                if core.endswith(gn) and len(core) > len(gn):
                    core, changed = core[: -len(gn)], True
        if len(core) >= 3:   # 2 字（生產、客戶、採購…）太常出現在一般問句，會誤鎖
            terms.add(core)
    return terms


def _focus_index(idx: Reader) -> dict[str, set[str]]:
    """詞 -> 擁有這個詞的 ISO 文件群（同 ISO 編號的多份文件算一群）。只留 1 群獨有的詞。"""
    key = str(idx.path)
    if key not in _focus_cache:
        owners: dict[str, set[str]] = {}
        groups: dict[str, set[str]] = {}
        for d in idx.docs.values():
            g = d["iso_key"] or d["doc_no"]
            groups.setdefault(g, set()).add(d["doc_no"])
            for term in _title_terms(d):
                owners.setdefault(term, set()).add(g)
        _focus_cache.clear()
        _focus_cache[key] = {t: groups[next(iter(gs))] for t, gs in owners.items() if len(gs) == 1}
    return _focus_cache[key]


def find_focus(q: str, idx: Reader) -> tuple[list[str], str]:
    nq = textnorm.norm(q)
    hits = [(t, docs) for t, docs in _focus_index(idx).items()
            if (re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", nq) if t.isascii() else t in nq)]
    groups = {frozenset(docs) for _, docs in hits}
    if len(groups) != 1:                     # 沒有、或同時指到多份 -> 不鎖定
        return [], ""
    return sorted(next(iter(groups))), max((t for t, _ in hits), key=len)


_printed_cache: dict[str, dict[str, str]] = {}


def _printed_codes(idx: Reader) -> dict[str, str]:
    """紙本編號 -> 文件編號。有些文件的屬性沒有登記編號，使用者卻用印在文件上的編號問（SOP-QC-0001），
    沒有這張對照就找不到文件。
    從資料取，證據由強到弱，每層都只收屬於唯一一份文件的代碼：
      1. 檔名或標題裡的代碼，也出現在這份文件自己的頁面文字（只看檔名會錯：檔名可能沿用別份文件的編號）；
      2. 只在檔名／標題（頁面 OCR 沒讀出編號），且沒有別份文件在第 1 層認領；
      3. 前兩層都沒有編號的文件：頁首重複印的代碼（出現在至少一半、且 ≥2 頁）。"""
    key = str(idx.path)
    if key not in _printed_cache:
        pages: dict[str, dict[str, set[int]]] = {}       # 代碼 -> 文件 -> 頁
        for c in idx.chunks.values():
            if c.kind != "meta":
                for k in textnorm.find_codes(c.text):
                    pages.setdefault(k, {}).setdefault(c.doc_no, set()).update(range(c.page_start, c.page_end + 1))
        owners: dict[str, set[str]] = {}
        named: dict[str, set[str]] = {}
        for d in idx.docs.values():
            stem = Path(d["pdf_path"]).stem if d["pdf_path"] else ""
            for k in set(textnorm.find_codes(stem) + textnorm.find_codes(d["title"] or "")):
                named.setdefault(k, set()).add(d["doc_no"])
                if d["doc_no"] in pages.get(k, {}):
                    owners.setdefault(k, set()).add(d["doc_no"])
        out = {k: next(iter(ds)) for k, ds in named.items() if len(ds) == 1 and k not in owners}
        out.update({k: next(iter(ds)) for k, ds in owners.items() if len(ds) == 1})
        have = set(out.values()) | {d["doc_no"] for d in idx.docs.values() if d["iso_key"]}
        header: dict[str, set[str]] = {}
        for k, by_doc in pages.items():
            if k in named:
                continue
            for n, ps in by_doc.items():
                if n not in have and len(ps) >= max(2, (idx.docs[n]["pages"] or 1) / 2):
                    header.setdefault(k, set()).add(n)
        out.update({k: next(iter(ds)) for k, ds in header.items() if len(ds) == 1})
        _printed_cache.clear()
        _printed_cache[key] = out
    return _printed_cache[key]


def analyze(q: str, idx: Reader) -> Analysis:
    a = Analysis(query=q, intent="content")
    a.codes = textnorm.find_codes(q)
    for key in a.codes:
        for d in idx.docs.values():
            if key in (d["iso_key"], textnorm.code_key(d["doc_no"])) or \
                    (d["iso_key"] and key.startswith(d["iso_key"]) and len(key) - len(d["iso_key"]) <= 2):
                if d["doc_no"] not in a.doc_nos:
                    a.doc_nos.append(d["doc_no"])
    printed = _printed_codes(idx)
    for key in a.codes:
        n = printed.get(key)
        if n and n not in a.doc_nos and not any(key in (d["iso_key"], textnorm.code_key(d["doc_no"]))
                                                 for d in idx.docs.values()):
            a.doc_nos.append(n)
    if a.doc_nos:
        a.focus_docs, a.focus_term = list(a.doc_nos), "文件代碼"
    else:
        a.focus_docs, a.focus_term = find_focus(q, idx)
    m = re.search(r"(9001|14001|45001|27001)", q)
    if m:
        a.iso_system = m.group(1)
    is_list = bool(LIST_RE.search(q))
    for k, v in list(DOC_TYPES_STRONG.items()) + (list(DOC_TYPES_WEAK.items()) if is_list else []):
        if k in q:
            a.doc_type = v
            break
    for pat, f, label in ATTRS:
        if re.search(pat, q, re.I) and (f, label) not in a.attrs:
            a.attrs.append((f, label))
    if LIST_RE.search(q) and not a.doc_nos and _list_explains(q):
        a.intent = "list"
        a.status = next((st for pat, st in STATES if re.search(pat, q)), None)
    elif a.attrs and a.doc_nos:
        a.intent = "attribute"
        rest = q
        for c in re.finditer(textnorm.CODE_RE, q):
            rest = rest.replace(c.group(0), "")
        for pat, _, _ in ATTRS:
            rest = re.sub(pat, "", rest, flags=re.I)
        for d in a.doc_nos:
            t = idx.docs[d]["title"] or ""
            for part in re.split(r"[\s_\-]+", t):
                if len(part) >= 2:
                    rest = rest.replace(part, "")
        rest = FILLER_RE.sub("", rest)
        a.attr_only = len(rest) <= 3
        if a.attr_only and _pdf_disagrees(q, idx, a):
            a.attr_only = False
    return a


def _pdf_disagrees(q: str, idx: Reader, a: Analysis) -> bool:
    """屬性題自我核對：問句裡的欄位詞（「版次」「生效日期」）也是這份 PDF 裡的欄位標籤，屬性值卻不在同一段
    -> 屬性資料與 PDF 不一致（常見於屬性沒跟著改版更新），改交給檢索＋LLM，以 PDF 為準。
    看的是資料對不對得上，不是問句怎麼問。"""
    words = {m.group(0) for pat, _, _ in ATTRS for m in [re.search(pat, q, re.I)] if m and len(m.group(0)) >= 2}
    for n in a.doc_nos:
        d = idx.docs[n]
        if not d.get("pdf_path"):
            continue
        vals = {str(v).split()[0] for f, _ in a.attrs if (v := d.get(f)) and str(v).strip()}
        for c in idx.chunks.values():
            if c.doc_no != n or c.kind == "meta":
                continue
            for w in words:
                # 欄位標籤：表格的一格（「版次 | …」），或後面接冒號（「生效日期：」）；一般用語不算
                label = re.search(rf"(?m)(?:^|[|｜])\s*{re.escape(w)}\s*(?:[|｜:：]|$)|{re.escape(w)}\s*[:：]", c.text)
                if label and not any(re.search(rf"(?<![0-9A-Za-z]){re.escape(v)}(?![0-9A-Za-z])", c.text)
                                     for v in vals):
                    return True
    return False


def _list_explains(q: str) -> bool:
    """列表題是「依屬性篩選」：拿掉列表詞、篩選條件（體系、階層、狀態）和虛詞後，問題應該沒剩什麼。
    還剩下內容（「不合格控制出現在」「提到不合格品維修」）表示屬性篩選回答不了，交給檢索＋LLM。
    跟屬性題的 attr_only 同一個自我檢查；否則「哪些文件有提到 <名稱>」會把名稱裡的「程序」「管理辦法」
    當篩選條件，列出整個階層的文件。"""
    rest = LIST_RE.sub("", q)
    pats = ([r"ISO", r"9001|14001|45001|27001"] + list(DOC_TYPES_STRONG) + list(DOC_TYPES_WEAK)
            + [p for p, _ in STATES] + [r"哪些|那些|所有|全部|目前|現在|份"])
    for pat in pats:
        rest = re.sub(pat, "", rest, flags=re.I)
    return len(FILLER_RE.sub("", rest)) <= 3


def filter_docs(a: Analysis, idx: Reader) -> list[dict]:
    out = []
    for d in idx.docs.values():
        if a.iso_system and a.iso_system not in (d["iso_system"] or ""):
            continue
        if a.doc_type and d["doc_type"] != a.doc_type:
            continue
        if a.status and d["status"] != a.status:
            continue
        out.append(d)
    return sorted(out, key=lambda d: (d["iso_no"] or "~", d["doc_no"]))
