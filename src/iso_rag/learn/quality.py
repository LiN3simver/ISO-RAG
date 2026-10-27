"""OCR 頁面品質：抽特徵 → 估計「字元正確率」(0~1)，不需要標準答案，所以新版型也能自己評。

特徵（FEATURES）：
  conf_*         OCR 信心（以字數加權）
  drop_ratio     信心 < 0.5 被丟掉的字比例
  ink_uncovered  頁面上「像文字的墨跡」沒有被任何偵測框蓋到的比例 —— 偵測漏字（例如受控章壓字）
                 時 OCR 信心照樣很高，只有這個特徵看得出來
  valid/garbage  字元組成：中文、英數、標點 vs 私用區、�、拉丁亂碼（壞掉的 CMap 常見）
  frag_ratio     單字框比例（框被切碎）
  lm_logp/oov2   字元語言模型（charlm）：這串字像不像這批文件會出現的字

估計器：
  heuristic  預設：信心 × 有效字比例 × 無亂碼
  residual   heuristic + 學到的修正量（有上下限）。修正量主要用「同一頁不同策略之間」的差異訓練
             （特徵與正確率都先減掉該頁平均）：自監督標準答案（文字層）跟畫面本來就不會 100% 一樣，
             每頁的天花板不同，直接回歸絕對正確率會學到「哪一頁天花板高」，換到真實掃描頁就失準。
             要在「留一份文件」交叉驗證中贏過 heuristic 才能上線。
"""
from __future__ import annotations

import base64
import json
import math
import re
import unicodedata
import zlib
from collections import Counter

import cv2
import numpy as np

from ..index import textnorm
from .charlm import CharLM, is_cjk

ANALYSIS_DPI = 100            # 墨跡/版型分析用的解析度（夠看版面，又快）
FEATURES = ("conf_mean", "conf_low", "conf_p10", "drop_ratio", "ink_uncovered", "valid_ratio",
            "garbage_ratio", "frag_ratio", "log_chars", "lm_logp", "oov2")
HEURISTIC_TARGET, HEURISTIC_REVIEW = 0.93, 0.85
INK_REF_DEFAULT = 0.17        # 真實掃描頁 ink_uncovered 的 p95（照片、簽名、logo 也是沒被框到的墨跡）
_SYMBOLS = set("°℃℉±×÷√∞≤≥≠≈‰※★☆●○◎■□◆◇△▲▽▼→←↑↓⇒·…—–‐―〜～§¶†‡")


def _valid(c: str) -> bool:
    o = ord(c)
    return (is_cjk(c) or 0x20 <= o < 0x7F or c in _SYMBOLS or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF
            or 0x2460 <= o <= 0x24FF or 0x2160 <= o <= 0x217F or 0x0370 <= o <= 0x03FF
            or unicodedata.category(c).startswith("P"))


def _garbage(c: str) -> bool:
    o = ord(c)
    if c in _SYMBOLS:          # ×（U+00D7）、÷ 落在下面的拉丁範圍，但表格裡的 × 是正常符號
        return False
    # 0xC0–0x24F：帶重音的拉丁字母——中文文件裡出現幾乎都是編碼錯置（UTF-8/Big5 被當 Latin-1 讀）
    return c == "�" or unicodedata.category(c) in ("Co", "Cn", "Cs", "Cc") or 0xC0 <= o <= 0x24F


def _w(w) -> tuple[str, float, float, float, float, float]:
    if isinstance(w, dict):
        return w["text"], w["score"], w["x0"], w["y0"], w["x1"], w["y1"]
    return w.text, w.score, w.x0, w.y0, w.x1, w.y1


# ---------------------------------------------------------------- 影像
def analysis_image(img: np.ndarray, dpi: float) -> np.ndarray:
    f = ANALYSIS_DPI / dpi
    if abs(f - 1) < 1e-6:
        return img
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def ink_layers(small: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(像文字的墨跡, 水平格線, 垂直格線)。用紅色通道：紅色印章在紅色通道接近白紙，不算墨跡。"""
    r = small[..., 2]                    # pdfium 渲染是 BGR：紅色在第 3 個色版
    med = float(np.median(r))
    otsu, _ = cv2.threshold(np.ascontiguousarray(r), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu < 1:                         # 純黑白影像 Otsu 會給 0
        otsu = 128
    ink = (r < min(otsu, med - 35)).astype(np.uint8)
    h, w = ink.shape
    lh = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 25, 8), 1)))
    lv = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 25, 8))))
    lh, lv = lh.astype(bool), lv.astype(bool)
    return ink.astype(bool) & ~(lh | lv), lh, lv


def ink_uncovered(small: np.ndarray, boxes_pt: list[tuple[float, float, float, float]]) -> float:
    text, _, _ = ink_layers(small)
    total = int(text.sum())
    if total < 200:                      # 幾乎空白頁
        return 0.0
    cover = np.zeros(text.shape, np.uint8)
    s = ANALYSIS_DPI / 72.0
    for x0, y0, x1, y1 in boxes_pt:
        cv2.rectangle(cover, (int(x0 * s) - 2, int(y0 * s) - 2), (int(x1 * s) + 2, int(y1 * s) + 2), 1, -1)
    return float((text & ~cover.astype(bool)).sum()) / total


# ---------------------------------------------------------------- 特徵
def base_features(words: list, text: str, *, dropped_chars: int = 0, ink_unc: float = 0.0,
                  scored: bool = True) -> dict:
    """不需要語言模型的特徵（OCR 當下就算好存起來；LM 特徵在 fit 時用當時的 LM 重算）。"""
    ws = [_w(w) for w in words]
    lens = np.array([max(len(t), 1) for t, *_ in ws], dtype=float)
    sc = np.array([s for _, s, *_ in ws], dtype=float)
    if scored and len(ws):
        tot = lens.sum()
        conf_mean = float((sc * lens).sum() / tot)
        conf_low = float(lens[sc < 0.8].sum() / tot)
        conf_p10 = float(np.percentile(np.repeat(sc, lens.astype(int)), 10))
    else:
        tot = float(len(text))
        conf_mean, conf_low, conf_p10 = (1.0, 0.0, 1.0) if not scored else (0.0, 1.0, 0.0)
    chars = [c for c in text if not c.isspace()]
    n = len(chars)
    return {
        "conf_mean": conf_mean, "conf_low": conf_low, "conf_p10": conf_p10,
        "drop_ratio": dropped_chars / (tot + dropped_chars) if (tot + dropped_chars) else 0.0,
        "ink_uncovered": ink_unc,
        "valid_ratio": sum(map(_valid, chars)) / n if n else 0.0,
        "garbage_ratio": sum(map(_garbage, chars)) / n if n else 0.0,
        "frag_ratio": float(np.mean([len(t) == 1 for t, *_ in ws])) if ws else 0.0,
        "log_chars": math.log1p(n) / 8,
    }


def heuristic(f: dict) -> float:
    return f["conf_mean"] * f["valid_ratio"] * (1 - f["garbage_ratio"])


# ---------------------------------------------------------------- 文字層檢查
def text_layer_check(text: str, lm: CharLM | None = None, lm_floor: float | None = None,
                     gain_floor: float | None = None, unseen_ceil: float | None = None) -> tuple[bool, str]:
    """文字層可不可以信。壞掉的 CMap 會吐出 (cid:123)、私用區字元、拉丁亂碼，或「每個字都是中文
    但連起來不成話」（Big5/GBK 錯置）——最後一種只有語言模型抓得到。"""
    s = text.strip()
    if "(cid:" in s:
        return False, "cid"
    f = base_features([], s, scored=False)
    if f["garbage_ratio"] > 0.05:
        return False, f"garbage={f['garbage_ratio']:.2f}"
    if f["valid_ratio"] < 0.85:
        return False, f"valid={f['valid_ratio']:.2f}"
    if lm is not None and not lm.empty and lm_floor is not None:
        lp, _ = lm.score(s)
        if lp < lm_floor:
            return False, f"lm={lp:.2f}<{lm_floor:.2f}"
    if lm is not None and not lm.empty and gain_floor is not None:
        g = lm.order_gain(s)
        if g is not None and g < gain_floor:
            return False, f"order={g:.2f}<{gain_floor:.2f}"
    if lm is not None and not lm.empty and unseen_ceil is not None:
        u = lm.unseen_chars(s)
        if u is not None and u > unseen_ceil:
            return False, f"unseen={u:.2f}>{unseen_ceil:.2f}"
    return True, "ok"


# ---------------------------------------------------------------- 評分用：與標準答案比
def _ev(s: str) -> str:
    # 「|」是表格還原（ingest.tables）加的欄位分隔，不是讀到的字
    return re.sub(r"[\s|]+", "", textnorm.norm(s))


def accuracy(pred: str, gold: str) -> float:
    """字元 bigram F1（不受表格閱讀順序影響；繁簡、全半形先統一）。"""
    a, b = _ev(pred), _ev(gold)
    A = Counter(a[i:i + 2] for i in range(len(a) - 1))
    B = Counter(b[i:i + 2] for i in range(len(b) - 1))
    o = sum((A & B).values())
    if not o:
        return 0.0
    p, r = o / sum(A.values()), o / sum(B.values())
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------- 模型
RESIDUAL_CAP = 0.15
Z_CAP = 3.0                   # 真實頁的特徵可能落在訓練分布外很遠，標準化後截在 ±3
# 只在「同一頁」比較才有意義的特徵：多抓到字代表這個策略漏得少，但跨頁比只是在比頁面長短，
# 放進絕對品質會讓正常的短頁被判成要人工確認
PAGE_FEATS = ("log_chars",)


BETWEEN_LAM = 20.0


def fit_residual(X: np.ndarray, y: np.ndarray, groups: list, lam: float = 1.0, between: bool = True):
    """heuristic 沒解釋到的部分，兩段 ridge 加起來：
      頁內：同頁置中後學「哪個策略的輸出比同頁其他策略好」（挑策略靠這個）。
      頁間：每頁特徵平均 -> 每頁平均殘差，學「哪一頁讀得比較差」。頁內置中會丟掉跨頁訊號，而沒被框到的墨跡、
        低信心比例跟正確率的相關都比平均信心強，所以另外補一段。正則化較強、不含字數這類只在同頁比較有意義的特徵；
        零點之後由 trainer 錨定在真實掃描頁的平均（real_pass_rate 把關）。"""
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std
    h = np.array([heuristic(dict(zip(FEATURES, x))) for x in X])
    r = y - h
    Zc, rc = Z.copy(), r.copy()
    keys = list(dict.fromkeys(groups))
    Zg, rg = [], []
    for g in keys:
        idx = [i for i, x in enumerate(groups) if x == g]
        Zg.append(Z[idx].mean(0))
        rg.append(r[idx].mean())
        Zc[idx] -= Z[idx].mean(0)
        rc[idx] -= r[idx].mean()
    w = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Z.shape[1]), Zc.T @ rc)
    if between and len(keys) >= 10:
        Zg, rg = np.array(Zg), np.array(rg)
        page = np.array([k in PAGE_FEATS for k in FEATURES])
        Zg[:, page] = 0.0
        Zg0, rg0 = Zg - Zg.mean(0), rg - rg.mean()
        w = w + np.linalg.solve(Zg0.T @ Zg0 + BETWEEN_LAM * np.eye(Z.shape[1]), Zg0.T @ rg0)
    return mean, std, w


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std
    b = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - b))
    return mean, std, w, b


class QualityModel:
    def __init__(self, kind: str = "heuristic", *, version: str = "heuristic", lm: CharLM | None = None,
                 mean=None, std=None, w=None, b: float = 0.0, target: float = HEURISTIC_TARGET,
                 review: float = HEURISTIC_REVIEW, lm_floor: float | None = None, prior: dict | None = None,
                 ink_ref: float = INK_REF_DEFAULT, ctx=None, gain_floor: float | None = None,
                 unseen_ceil: float | None = None):
        self.kind, self.version, self.lm = kind, version, lm
        self.mean = None if mean is None else np.asarray(mean, float)
        self.std = None if std is None else np.asarray(std, float)
        self.w = None if w is None else np.asarray(w, float)
        self.b = b
        self.target, self.review, self.lm_floor = target, review, lm_floor
        self.gain_floor = gain_floor     # 文字層字序增益的下限（CharLM.order_gain）
        self.unseen_ceil = unseen_ceil   # 文字層「沒看過的字」比例上限（CharLM.unseen_chars）
        self.prior = prior or {}
        self.ink_ref = ink_ref           # 真實掃描頁 ink_uncovered 的 p90（版型自己的紀錄不足時用）
        self.ctx = ctx                   # learn.context.ContextModel：看掃描狀況挑策略（沒有就只看版型勝場）

    def features(self, base: dict, text: str) -> dict:
        f = dict(base)
        f["lm_logp"], f["oov2"] = self.lm.score(text) if self.lm is not None else (0.0, 0.0)
        return f

    def _z(self, f: dict) -> np.ndarray:
        return np.clip((np.array([f[k] for k in FEATURES], float) - self.mean) / self.std, -Z_CAP, Z_CAP)

    def predict(self, f: dict) -> float:
        """絕對品質（估計正確率）：決定要不要再試別的策略、要不要請人確認。"""
        if self.kind == "heuristic":
            return heuristic(f)
        z = self._z(f)
        if self.kind == "residual":
            page = np.array([k in PAGE_FEATS for k in FEATURES])
            adj = float(np.clip(z[~page] @ self.w[~page], -RESIDUAL_CAP, RESIDUAL_CAP))
            return float(np.clip(heuristic(f) + adj, 0.0, 1.0))
        return float(np.clip(self.b + z @ self.w, 0.0, 1.0))

    def rank(self, f: dict) -> float:
        """同一頁多個策略之間挑哪個：絕對品質 + 只在同頁比較才有意義的特徵。"""
        q = self.predict(f)
        if self.kind != "residual":
            return q
        z = self._z(f)
        return q + sum(float(z[i] * self.w[i]) for i, k in enumerate(FEATURES) if k in PAGE_FEATS)

    def check_text_layer(self, text: str) -> tuple[bool, str]:
        return text_layer_check(text, self.lm, self.lm_floor, self.gain_floor, self.unseen_ceil)

    def to_params(self) -> bytes:
        d = {"kind": self.kind, "version": self.version, "b": self.b, "target": self.target, "review": self.review,
             "lm_floor": self.lm_floor, "gain_floor": self.gain_floor, "unseen_ceil": self.unseen_ceil,
             "prior": self.prior, "ink_ref": self.ink_ref, "features": list(FEATURES),
             "mean": None if self.mean is None else self.mean.tolist(),
             "std": None if self.std is None else self.std.tolist(),
             "w": None if self.w is None else self.w.tolist(),
             "lm": base64.b64encode(self.lm.dumps()).decode() if self.lm is not None else None,
             "ctx": self.ctx.to_dict() if self.ctx is not None else None}
        return zlib.compress(json.dumps(d).encode())

    @classmethod
    def from_params(cls, blob: bytes) -> QualityModel:
        d = json.loads(zlib.decompress(blob))
        if d.get("features") and tuple(d["features"]) != FEATURES:
            raise ValueError(f"模型特徵 {d['features']} 與程式 {FEATURES} 不一致，請重新 learn fit")
        from .context import ContextModel
        lm = CharLM.loads(base64.b64decode(d["lm"])) if d.get("lm") else None
        return cls(d["kind"], version=d["version"], lm=lm, mean=d["mean"], std=d["std"], w=d["w"], b=d["b"],
                   target=d["target"], review=d["review"], lm_floor=d["lm_floor"], prior=d.get("prior"),
                   ink_ref=d.get("ink_ref", INK_REF_DEFAULT), ctx=ContextModel.from_dict(d.get("ctx")),
                   gain_floor=d.get("gain_floor"), unseen_ceil=d.get("unseen_ceil"))

    @classmethod
    def load_active(cls, store) -> QualityModel:
        v = store.model_active()
        return cls.from_params(store.model_params(v)) if v else cls()
