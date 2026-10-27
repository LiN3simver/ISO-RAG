"""看影像的掃描狀況挑 OCR 策略（情境模型），新版型第一頁就能選對。

版型勝場要試錯好幾頁才學得會，而且同一個版型裡有蓋章和沒蓋章的頁要的策略本來就不同。
最佳策略跟「這一頁的狀況」有關（紅章 → redch、褪色 → loose、乾淨／歪斜 → base），
所以從影像量出狀況，直接預測每個策略比 base 好多少：

  特徵（OCR 之前就算得出來，約 0.1 秒）：
    red_ratio  紅色像素比例（受控章）            bg_level  紙色（灰底、泛黃）
    ink_ratio  墨跡比例                          ink_level 最深的墨有多深（褪色時偏淡）
    contrast   紙與墨的差                        sharp     筆畫邊緣銳利度（模糊、低解析時低）
    stroke     筆畫粗細（低解析放大、模糊時變粗）  skew      文字列歪斜角度
  模型：kNN（標準化特徵、距離加權），回傳各策略相對 base 的預估增益。
        資料來自 learn selfsup（文字層頁 → 模擬劣化 → 6 個策略都跑 → 跟文字層比正確率）。
  用法（adaptive）：預估某策略比 base 多 ≥ CTX_MIN_GAIN 就先試它，其餘照版型自己的順序。
  評估（trainer.fit）：留一份文件交叉驗證，以「重播」模擬自適應迴圈
        （每頁 6 個策略的結果都有，照決策規則挑、算最後正確率／嘗試次數／秒數）。
"""
from __future__ import annotations

import cv2
import numpy as np

COND_FEATURES = ("red_ratio", "ink_ratio", "ink_level", "bg_level", "contrast", "sharp", "stroke", "skew")
K = 8
CTX_MIN_GAIN = 0.01


def condition_features(img: np.ndarray, dpi: int = 300) -> dict:
    """img：pdfium 渲染的 BGR 頁面影像（dpi）。"""
    f = 100.0 / dpi
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    b, g, r = (small[..., i].astype(np.int16) for i in range(3))
    red = ((r - np.maximum(g, b)) > 50) & (r > 100)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bg = float(np.median(gray))
    dark = gray < bg - 40
    ink_ratio = float(dark.mean())
    ink_level = float(np.percentile(gray[dark], 10)) if dark.any() else bg
    # 銳利度、筆畫粗細看原解析度的中間一塊（頁首頁尾常是 logo、空白）
    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = G.shape
    c = G[H // 4: 3 * H // 4, W // 6: 5 * W // 6]
    cb = float(np.median(c))
    ink = (c < cb - 40).astype(np.uint8)
    contrast = max(bg - ink_level, 1.0)
    lap = np.abs(cv2.Laplacian(c, cv2.CV_32F))
    sharp = float(np.percentile(lap[ink.astype(bool)], 90)) / contrast if ink.any() else 0.0
    edge = ink & ~cv2.erode(ink, np.ones((3, 3), np.uint8))
    stroke = float(2.0 * ink.sum() / max(int(edge.sum()), 1)) * 72.0 / dpi     # pt
    return {"red_ratio": float(red.mean()), "ink_ratio": ink_ratio, "ink_level": ink_level / 255.0,
            "bg_level": bg / 255.0, "contrast": contrast / 255.0, "sharp": sharp, "stroke": stroke,
            "skew": _skew(dark)}


def _skew(dark: np.ndarray) -> float:
    """文字列歪斜（度，絕對值）：轉一點角度後水平投影最「尖」的那個角度。"""
    if dark.sum() < 500:
        return 0.0
    m = dark.astype(np.uint8) * 255
    h, w = m.shape
    best, ang = -1.0, 0.0
    for a in np.arange(-3.0, 3.01, 0.5):
        rot = cv2.warpAffine(m, cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0), (w, h))
        prof = rot.sum(axis=1).astype(np.float64)
        v = float(np.var(prof))
        if v > best:
            best, ang = v, float(a)
    return abs(ang)


class ContextModel:
    """kNN：特徵 -> 各策略相對 base 的預估正確率增益。"""

    def __init__(self, X: np.ndarray, gains: np.ndarray, variants: list[str], secs: np.ndarray | None = None):
        self.variants = list(variants)
        self.X = np.asarray(X, float)
        self.gains = np.asarray(gains, float)
        self.secs = None if secs is None else np.asarray(secs, float)
        self.mean = self.X.mean(0) if len(self.X) else np.zeros(len(COND_FEATURES))
        self.std = self.X.std(0) if len(self.X) else np.ones(len(COND_FEATURES))
        self.std[self.std < 1e-9] = 1.0
        self.Z = (self.X - self.mean) / self.std

    @classmethod
    def fit(cls, groups: list[dict]) -> ContextModel:
        """groups：[{"cond": dict, "acc": {策略: 正確率}, "seconds": {策略: 秒}}]，只用 6 個策略都有的組。"""
        from .adaptive import VARIANTS
        vs = list(VARIANTS)
        full = [g for g in groups if all(v in g["acc"] for v in vs) and g.get("cond")]
        X = np.array([[g["cond"][k] for k in COND_FEATURES] for g in full], float).reshape(-1, len(COND_FEATURES))
        G = np.array([[g["acc"][v] - g["acc"]["base"] for v in vs] for g in full], float).reshape(-1, len(vs))
        S = np.array([[g["seconds"].get(v, 0.0) for v in vs] for g in full], float).reshape(-1, len(vs))
        return cls(X, G, vs, S)

    @property
    def n(self) -> int:
        return len(self.X)

    def gain(self, cond: dict) -> dict[str, float]:
        if self.n == 0:
            return {v: 0.0 for v in self.variants}
        z = (np.array([cond[k] for k in COND_FEATURES], float) - self.mean) / self.std
        d = np.sqrt(((self.Z - z) ** 2).sum(1))
        k = min(K, self.n)
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 0.5)
        g = (self.gains[idx] * w[:, None]).sum(0) / w.sum()
        return dict(zip(self.variants, map(float, g)))

    def first(self, cond: dict) -> tuple[str, float]:
        """預估最好的策略與它比 base 多多少。"""
        g = self.gain(cond)
        v = max(g, key=lambda x: (g[x], x == "base"))
        return v, g[v]

    def to_dict(self) -> dict:
        return {"variants": self.variants, "X": self.X.tolist(), "gains": self.gains.tolist(),
                "secs": None if self.secs is None else self.secs.tolist(), "features": list(COND_FEATURES)}

    @classmethod
    def from_dict(cls, d: dict | None) -> ContextModel | None:
        if not d or tuple(d.get("features") or ()) != COND_FEATURES:
            return None
        return cls(np.array(d["X"], float).reshape(-1, len(COND_FEATURES)),
                   np.array(d["gains"], float).reshape(-1, len(d["variants"])), d["variants"],
                   None if d.get("secs") is None else np.array(d["secs"], float))


def ctx_order(base_order: list[str], ctx: ContextModel | None, cond: dict | None) -> tuple[list[str], str]:
    """情境模型排整個順序：預估某策略比 base 好 ≥ CTX_MIN_GAIN 就先試它，否則 base 先；
    其餘依預估增益排，版型順序（勝場）只拿來打破平手。回傳 (順序, 說明)。
    整個順序都交給情境模型，是因為新版型的版型順序只是全域先驗，對個別頁的狀況常常排錯。"""
    if ctx is None or cond is None or ctx.n == 0:
        return base_order, ""
    g = ctx.gain(cond)
    v = max(g, key=lambda x: (g[x], x == "base"))
    first = v if v != "base" and g[v] >= CTX_MIN_GAIN else "base"
    rest = sorted((x for x in base_order if x != first), key=lambda x: (-round(g.get(x, 0.0), 3), base_order.index(x)))
    return [first] + rest, (f"ctx:{first}+{g[first]:.3f}" if first != "base" else "")


# ---------------------------------------------------------------- 重播（評估用）
def replay(samples: dict[str, dict], order: list[str], *, new: bool, target: float, max_tries: int,
           min_tries: int, min_gain: float) -> dict:
    """samples：策略 -> {"acc", "quality", "rank", "seconds", "suspicious"}（該頁每個策略都跑過）。
    照 adaptive.ocr_image 的停止規則模擬（不含時間上限），回傳 {"acc", "first_acc", "tries", "seconds", "variant"}。"""
    tried = [order[0]]
    for v in order[1:]:
        top = max(tried, key=lambda x: samples[x]["rank"])
        if len(tried) >= max_tries:
            break
        if len(tried) >= min_tries and samples[top]["quality"] >= target and not samples[top]["suspicious"]:
            break
        if len(tried) >= 2 and not new and \
                samples[tried[-1]]["rank"] - max(samples[x]["rank"] for x in tried[:-1]) < min_gain:
            break
        tried.append(v)
    best = max(tried, key=lambda x: (samples[x]["rank"], -samples[x]["seconds"]))
    return {"acc": samples[best]["acc"], "first_acc": samples[order[0]]["acc"], "tries": len(tried),
            "seconds": sum(samples[x]["seconds"] for x in tried), "variant": best,
            "oracle": max(s["acc"] for s in samples.values())}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ("acc", "first_acc", "tries", "seconds", "oracle")
    out = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    out["n"] = len(rows)
    return out

