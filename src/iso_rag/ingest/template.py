"""頁面樣板（頁首表頭、頁尾、公司名稱、版權聲明）：每次 build 從整批文件統計學出來，不寫死規則。

每種版型的表頭、頁尾寫法都不同（「頁 次 4/7」「分 頁 次 3/17」、公司名稱、版權聲明…），寫死正規式維護不完，
所以用統計學（以「整行」為單位；數字遮成 0，所以「頁數 3/17」「頁數 4/17」是同一行）：
  R1 同一份 PDF 內：≥3 頁的文件裡，出現在 ≥ max(2, 一半) 頁的行 = 這份文件的樣板（頁首頁尾、重複表頭）
  R2 跨文件：在 ≥ CROSS_DOCS 份 PDF 裡都是 R1 樣板的行 = 共通樣板，連只有 1、2 頁的文件也套用
     （公司名稱、版權聲明這種每種版型自己的固定字）
  條文標題（「1.目的」「5.0 流程圖」）永遠不當樣板：它們帶結構，而且每份文件都有。
  太短的行（中文 < 2 字且英數 < 4 個）不判斷，「OK」「■」「×」這種留著。
只刪整行，不刪行裡的片段：片段（例如部門名稱「品保部」）在本文裡常是真的內容。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from ..index import textnorm

CROSS_DOCS = 2
_DIGIT = re.compile(r"\d")
_SPACE = re.compile(r"[\s|]+")
_CJK = re.compile(f"[{textnorm.CJK}]")
_ALNUM = re.compile(r"[a-z0-9]")


def line_key(line: str) -> str:
    return _DIGIT.sub("0", _SPACE.sub("", textnorm.norm(line)))


def _judged(key: str) -> bool:
    return len(_CJK.findall(key)) >= 2 or len(_ALNUM.findall(key)) >= 4


@dataclass
class Templates:
    per_doc: dict[str, set[str]] = field(default_factory=dict)     # pdf sha -> R1 樣板行
    common: set[str] = field(default_factory=set)                  # R2 共通樣板行

    @classmethod
    def learn(cls, docs: dict[str, list[str]], is_heading=None) -> Templates:
        """docs：pdf sha -> 每頁的頁面文字。is_heading(line) -> bool：條文標題不當樣板。"""
        t = cls()
        r1_docs: Counter = Counter()
        for sha, pages in docs.items():
            n = len(pages)
            if n < 3:
                continue
            df: Counter = Counter()
            for text in pages:
                df.update({line_key(ln) for ln in text.splitlines() if ln.strip()})
            need = max(2, math.ceil(0.5 * n))
            keys = {k for k, c in df.items() if c >= need and _judged(k)}
            if is_heading:
                heads = {line_key(ln) for text in pages for ln in text.splitlines() if is_heading(ln)}
                keys -= heads
            t.per_doc[sha] = keys
            r1_docs.update(keys)
        t.common = {k for k, c in r1_docs.items() if c >= CROSS_DOCS}
        return t

    def is_template(self, sha: str, line: str) -> bool:
        k = line_key(line)
        return k in self.common or k in self.per_doc.get(sha, ())

    def strip(self, sha: str, lines: list[str], is_heading=None) -> list[str]:
        return [ln for ln in lines if not self.is_template(sha, ln) or (is_heading and is_heading(ln))]

    def stats(self) -> dict:
        return {"docs": len(self.per_doc), "common": len(self.common),
                "per_doc_mean": round(sum(map(len, self.per_doc.values())) / max(1, len(self.per_doc)), 1)}

    def examples(self, k: int = 30) -> list[str]:
        cnt = Counter(x for v in self.per_doc.values() for x in v)
        return [x for x, _ in cnt.most_common(k)]


def by_sha(pages_of: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {sha: [p["text"] for p in pages] for sha, pages in pages_of.items()}

