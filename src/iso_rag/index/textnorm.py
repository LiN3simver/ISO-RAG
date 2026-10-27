"""檢索用的文字正規化與斷詞（只用在索引 key，顯示仍用原文）。

- NFKC：全形英數 -> 半形、康熙部首 -> 一般字。
- 繁簡統一成「簡體」當 key（t2s 是多對一，比 s2t 穩；OCR 會把「穴數」讀成「穴数」，
  使用者也可能打簡體）。
- CJK 切 bigram（單字 run 保留 unigram）；英數保留整個 token。
- 文件代碼（SOP-QA-001、SIP-QA101-A1）另外產生一個去掉符號的
  整體 token（sopqa001），讓「SOP QA 001」「sop-qa-001」都能精準命中。
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

CJK = "㐀-䶿一-鿿豈-﫿"
_CJK_RUN = re.compile(f"[{CJK}]+")
_ALNUM = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
# 代碼：字母開頭、至少一個連字號或字母數字混合，例：sop-qa-001 / qerp-01-d / diso-000101 / sip-qa101-a1
CODE_RE = re.compile(r"(?<![a-z0-9])([a-z]{1,6}(?:(?:\s*[-_]\s*|\s+)[a-z0-9]{1,6}){1,3})(?![a-z0-9])", re.I)


@lru_cache(maxsize=1)
def _t2s():
    from opencc import OpenCC
    return OpenCC("t2s")


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    return _t2s().convert(t)


def code_key(code: str) -> str:
    """'SOP-QA-001' / 'sop qa 001' / 'SOP_QA_001' -> 'sopqa001'"""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", code).lower())


def find_codes(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text or "")
    out = []
    for m in CODE_RE.finditer(t):
        k = code_key(m.group(1))
        if any(c.isdigit() for c in k) and len(k) >= 4:
            out.append(k)
    return out


def tokens(text: str) -> list[str]:
    t = norm(text)
    out: list[str] = []
    for run in _CJK_RUN.findall(t):
        out.extend([run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)])
    out.extend(_ALNUM.findall(t))
    out.extend(find_codes(text))
    return out


def fts_text(text: str) -> str:
    """給 SQLite FTS5（unicode61）用：token 以空白分隔；符號拿掉避免被再切開。"""
    return " ".join(re.sub(r"[^0-9a-z㐀-鿿豈-﫿]", "", tok) or tok for tok in tokens(text))


@lru_cache(maxsize=1)
def _s2t():
    from opencc import OpenCC
    return OpenCC("s2tw")


def to_traditional(text: str) -> str:
    """答案統一成繁體（原文是簡體的文件，模型常照抄簡體）。
    OpenCC 會把「台」改成「臺」，台灣慣用「台」，先換成佔位字再還原。"""
    keep = ""   # Unicode 私用區字元，不會出現在文件裡
    return _s2t().convert((text or "").replace("台", keep)).replace(keep, "台")
