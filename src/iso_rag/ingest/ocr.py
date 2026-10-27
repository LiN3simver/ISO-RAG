"""掃描頁 OCR：pypdfium2 渲染 + RapidOCR（onnxruntime，免 torch）。

- 300 DPI：比 200/400 DPI 的辨識結果好。
- 座標轉回 PDF point（72 dpi），之後切段/判斷頁首頁尾用。
- 依「y 中心落在目前這行字高內」合併成行。
- NFKC 正規化：OCR 常吐康熙部首等相容字元，不正規化會讓關鍵字比對失敗。
- 方向：整頁讀不到字或信心低時（橫印直放的表單），試 90/270/180 度，取「字數×平均信心」最高者。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pypdfium2 as pdfium

DPI = 300
MIN_SCORE = 0.5


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float


@dataclass
class PageOCR:
    words: list[Word]
    rotation: int          # 讀取時把影像轉了幾度（0/90/180/270）
    width: float           # 轉正後的頁寬（point）
    height: float

    @property
    def quality(self) -> float:
        if not self.words:
            return 0.0
        n = sum(len(w.text) for w in self.words)
        return n * (sum(w.score for w in self.words) / len(self.words))


@lru_cache(maxsize=4)
def engine(version: str = "v6", engine_type: str = "onnxruntime"):
    from ..netfix import use_os_certs
    use_os_certs()  # 首次會從 modelscope 下載模型
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    et = EngineType(engine_type)
    if version == "v6":
        params = {"Det.ocr_version": OCRVersion.PPOCRV6, "Rec.ocr_version": OCRVersion.PPOCRV6,
                  "Det.model_type": ModelType.SMALL, "Rec.model_type": ModelType.SMALL}
    elif version == "v5-server":
        params = {"Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
                  "Det.model_type": ModelType.SERVER, "Rec.model_type": ModelType.SERVER,
                  "Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.CH}
    else:  # v5 mobile
        params = {"Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
                  "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE,
                  "Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.CH}
    params.update({"Det.engine_type": et, "Rec.engine_type": et, "Cls.engine_type": et,
                   "Global.log_level": "error"})
    return RapidOCR(params=params)


def render(pdf_path: str, page_no: int, dpi: int = DPI) -> np.ndarray:
    doc = pdfium.PdfDocument(pdf_path)
    try:
        img = doc[page_no].render(scale=dpi / 72.0).to_numpy()
    finally:
        doc.close()
    return np.ascontiguousarray(img[:, :, :3])


def _ocr(img: np.ndarray, rotation: int, dpi: int, eng) -> PageOCR:
    k = {0: 0, 90: 3, 180: 2, 270: 1}[rotation]   # 順時針 rotation 度
    im = np.ascontiguousarray(np.rot90(img, k)) if k else img
    res = eng(im)
    scale = dpi / 72.0
    words = []
    if res.boxes is not None:
        for box, text, score in zip(res.boxes, res.txts, res.scores):
            text = unicodedata.normalize("NFKC", text or "").strip()
            if score < MIN_SCORE or not text:
                continue
            xs, ys = box[:, 0], box[:, 1]
            words.append(Word(float(xs.min() / scale), float(ys.min() / scale),
                              float(xs.max() / scale), float(ys.max() / scale), text, float(score)))
    return PageOCR(words, rotation, float(im.shape[1] / scale), float(im.shape[0] / scale))


def ocr_page(pdf_path: str, page_no: int, *, dpi: int = DPI, version: str = "v6",
             engine_type: str = "onnxruntime", try_rotations: bool = True) -> PageOCR:
    eng = engine(version, engine_type)
    img = render(pdf_path, page_no, dpi)
    best = _ocr(img, 0, dpi, eng)
    if try_rotations and _looks_rotated(best):
        for rot in (90, 270, 180):
            cand = _ocr(img, rot, dpi, eng)
            if cand.quality > best.quality * 1.3:
                best = cand
    return best


def _looks_rotated(p: PageOCR) -> bool:
    """直放的橫印頁：偵測框多半是又高又窄的「直條」，或幾乎讀不到字。"""
    if len(p.words) < 5:
        return True
    tall = sum(1 for w in p.words if (w.y1 - w.y0) > 2.0 * (w.x1 - w.x0) and len(w.text) > 1)
    avg = sum(w.score for w in p.words) / len(p.words)
    return tall / len(p.words) > 0.3 or avg < 0.75


def to_lines(words: list[Word]) -> list[list[Word]]:
    if not words:
        return []
    items = sorted(words, key=lambda w: ((w.y0 + w.y1) / 2, w.x0))
    lines: list[list[Word]] = [[items[0]]]
    for w in items[1:]:
        cur = lines[-1]
        cy = (w.y0 + w.y1) / 2
        ref_cy = sum((x.y0 + x.y1) / 2 for x in cur) / len(cur)
        ref_h = sum(x.y1 - x.y0 for x in cur) / len(cur)
        if abs(cy - ref_cy) <= ref_h * 0.6:
            cur.append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def page_text(words: list[Word], sep: str = " ") -> str:
    return "\n".join(sep.join(w.text for w in line) for line in to_lines(words))


# ---------------------------------------------------------------- 表格勾選記號
MARK = "■"


def filled_marks(img: np.ndarray, boxes_pt: list[tuple[float, float, float, float]], dpi: int = DPI) -> list[Word]:
    """找出表格裡的實心方塊（檢規的 IQC/首件/巡檢欄「■＝要檢驗」），補成 Word("■")。

    OCR 只認得出「×」，「■」整個不見，「進料 ■、首件 ■、巡檢 ×」會變成只剩「x」。
    條件：邊長 1.2~5 mm、長寬比接近 1、填滿率 > 0.8、沒有被任何文字框蓋到（粗體字、logo 會被 OCR 框住）、
    周圍一圈是白紙（照片裡的暗塊周圍是灰階紋理，誤報多出在照片和頁緣）、不貼頁緣。
    """
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    ink = (gray < 110).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    px = dpi / 25.4
    s = dpi / 72.0
    lo, hi = 1.2 * px, 5.0 * px
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (lo <= w <= hi and lo <= h <= hi and 0.7 <= w / h <= 1.4 and area / (w * h) > 0.8):
            continue
        pad = max(int(0.2 * w), 3)
        if x - pad < 0 or y - pad < 0 or x + w + pad > ink.shape[1] or y + h + pad > ink.shape[0]:
            continue
        ring = gray[y - pad:y + h + pad, x - pad:x + w + pad].astype(np.float32)
        ring[pad:pad + h, pad:pad + w] = np.nan                       # 只看外圈
        if np.nanmean(ring < 200) > 0.25:      # 外圈暗點比例：勾選記號 ≤ 0.15，照片暗塊 ≥ 0.38
            continue
        cx, cy = (x + w / 2) / s, (y + h / 2) / s
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes_pt):
            continue
        out.append(Word(x / s, y / s, (x + w) / s, (y + h) / s, MARK, 1.0))
    return out
