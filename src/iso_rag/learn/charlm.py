"""字元 bigram 語言模型：判斷一段文字「像不像這批文件會出現的字串」。

- 語料取自文件本身（文字層頁、人工校正頁、OCR 高信心框），不需要外部字典，新文件進來也會跟著學。
- key 先經 textnorm.norm（NFKC、轉簡體），再把數字併成 0、英文字母併成 a：料號、日期、版次
  不會因為「沒看過這串數字」被扣分。空白去掉（OCR 斷行位置不固定）。
- Witten-Bell 平滑：P(b|a) = (c(ab) + T(a)·P(b)) / (c(a) + T(a))，T(a) = a 後面接過幾種字；
  unigram 用 add-one，沒看過的字也有保底機率。
"""
from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter

from ..index import textnorm

_DIGIT = re.compile(r"\d")
_ALPHA = re.compile(r"[a-z]")
_WS = re.compile(r"\s+")


def lm_key(text: str) -> str:
    return _WS.sub("", _ALPHA.sub("a", _DIGIT.sub("0", textnorm.norm(text))))


def is_cjk(ch: str) -> bool:
    return "㐀" <= ch <= "鿿" or "豈" <= ch <= "﫿"


class CharLM:
    def __init__(self, uni: dict | None = None, bi: dict | None = None):
        self.uni = Counter(uni or {})
        self.bi = Counter(bi or {})
        self.finish()

    def add(self, text: str):
        k = lm_key(text)
        self.uni.update(k)
        self.bi.update(k[i:i + 2] for i in range(len(k) - 1))

    def finish(self) -> CharLM:
        self.n = sum(self.uni.values())
        self.v = len(self.uni)
        self.ctx: Counter = Counter()
        self.types: Counter = Counter()
        for ab, c in self.bi.items():
            self.ctx[ab[0]] += c
            self.types[ab[0]] += 1
        return self

    @property
    def empty(self) -> bool:
        return self.n == 0

    def p_uni(self, ch: str) -> float:
        return (self.uni.get(ch, 0) + 1) / (self.n + self.v + 1)

    def p(self, a: str, b: str) -> float:
        t = self.types.get(a, 0)
        if not t:
            return self.p_uni(b)
        return (self.bi.get(a + b, 0) + t * self.p_uni(b)) / (self.ctx[a] + t)

    def score(self, text: str) -> tuple[float, float]:
        """(每字平均 log2 機率, 中文 bigram 沒看過的比例)。太短或模型是空的 -> (0, 0)。"""
        k = lm_key(text)
        if len(k) < 2 or self.empty:
            return 0.0, 0.0
        lp = sum(math.log2(self.p(a, b)) for a, b in zip(k, k[1:])) / (len(k) - 1)
        cjk = [a + b for a, b in zip(k, k[1:]) if is_cjk(a) and is_cjk(b)]
        oov = sum(1 for g in cjk if g not in self.bi) / len(cjk) if cjk else 0.0
        return lp, oov

    def order_gain(self, text: str) -> float | None:
        """字序帶來的增益：雙字組模型比單字模型好多少（每字 log2）。正常文字的字序有意義，增益大；
        打亂、Big5/GBK 錯置的字一個個都是中文但前後不搭，增益接近 0 或負。跟字罕不罕見無關，
        所以組織圖、人名清單這種本身分數就低的頁也不會拉鬆門檻。太短或模型是空的 -> None。"""
        k = lm_key(text)
        if len(k) < 2 or self.empty:
            return None
        n = len(k) - 1
        bi = sum(math.log2(self.p(a, b)) for a, b in zip(k, k[1:])) / n
        uni = sum(math.log2(self.p_uni(b)) for b in k[1:]) / n
        return bi - uni

    def unseen_chars(self, text: str, min_chars: int = 20) -> float | None:
        """非 ASCII 的字裡，語料從沒出現過的比例。Big5 被當 GBK 解碼會冒出日文假名和罕見字
        （「ゅンW嘿」「瓣悔靴鳓」），英數多的頁分數和字序增益都不低，只有這個看得出來。字太少 -> None。"""
        k = [c for c in lm_key(text) if not c.isascii()]
        if len(k) < min_chars or self.empty:
            return None
        return sum(self.uni.get(c, 0) == 0 for c in k) / len(k)

    def dumps(self) -> bytes:
        return zlib.compress(json.dumps({"uni": self.uni, "bi": self.bi}, ensure_ascii=False).encode("utf-8"))

    @classmethod
    def loads(cls, blob: bytes) -> CharLM:
        d = json.loads(zlib.decompress(blob))
        return cls(d["uni"], d["bi"])
