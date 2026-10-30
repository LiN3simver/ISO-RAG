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


# ---------------------------------------------------------------- 選項控件（電子表單的單選鈕／勾選框）
OPTIONS_V = 1
OPTION = {("o", True): "◉", ("o", False): "○", ("s", True): "■", ("s", False): "□"}
_OPTION_CHARS = set("○◯◎●◉□☐■☑☒✓✔")       # OCR 對控件的各種讀法，拆詞時一律換成偵測結果


def _controls(img: np.ndarray, dpi: int) -> list[tuple[int, int, int, int, str, bool]]:
    """影像上的選項控件 [(x0, y0, x1, y1 像素, 'o' 圓 / 's' 方, 有沒有勾)]。

    電子表單截圖的「◉新增 ○修改 ○取消」：OCR 會把控件吞掉或讀成雜字，變成「新增修改取消」，
    問「勾的是哪個」就無從回答。控件的判斷全靠幾何形狀：
      - 邊長 1.6~4.5 mm、外框圍起的形狀（圓的填滿率 ≈ 0.785、方 ≈ 1），框線與中心之間留白；
      - 同一列至少兩個同形同大小才算（選項成組出現，單獨的「口」字不算）；
      - 勾選的那顆常是藍底、空隙不白，改認「與同組空心控件同形同大小、中心有墨水、轉 180 度不變」。
    """
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    n, lab, stats, _ = cv2.connectedComponentsWithStats((gray < 225).astype(np.uint8), connectivity=8)
    px = dpi / 25.4
    lo, hi = 1.6 * px, 4.5 * px
    k3 = np.ones((3, 3), np.uint8)
    hollow, solid = [], []
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        if not (lo <= w <= hi and lo <= h <= hi and 0.85 <= w / h <= 1.18):
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        comp = cv2.morphologyEx(np.pad(comp, 2), cv2.MORPH_CLOSE, k3)[2:-2, 2:-2]   # 淡灰外框有斷點
        ff = np.pad(1 - comp, 1, constant_values=1)
        cv2.floodFill(ff, None, (0, 0), 2)
        filled = (ff[1:-1, 1:-1] != 2).astype(np.uint8)        # 外框圍起來的區域
        fa = float(filled.mean())
        if fa < 0.7:
            continue
        s = min(w, h)
        a1 = cv2.erode(filled, k3, iterations=max(int(0.13 * s), 2)).astype(bool)
        a2 = cv2.erode(filled, k3, iterations=max(int(0.27 * s), 3)).astype(bool)
        gap = a1 & ~a2
        if not a2.any() or not gap.any():
            continue
        g = gray[y:y + h, x:x + w]
        center = float((g[a2] < 160).mean())
        if 0.1 <= center < (0.6 if fa <= 0.9 else 0.5):
            continue           # 中心只有一部分墨水：圈號數字、掃描雜點，不是空框也不是實心點
        band = gray[y + 2:y + h - 2] < 160
        if _blank(band, x - 1, -1, s) < 0.5 * s or _blank(band, x + w, 1, s) < 0.2 * s:
            continue           # 控件左右要留白；英文字母 o/D/e 總有一邊緊貼別的字
        c = (x, y, x + w, y + h, "s" if fa > 0.9 else "o", fa, center)
        if np.median(g[gap]) >= 200:
            hollow.append(c)
        elif np.abs(g.astype(np.float32) - g[::-1, ::-1]).mean() < 0.12 * 255:
            solid.append(c)        # 圓點、實心方塊轉 180 度不變；字不會

    def mates(c, pool):
        sz, cy = c[3] - c[1], (c[1] + c[3]) / 2
        return [d for d in pool if d is not c and d[4] == c[4] and abs((d[1] + d[3]) / 2 - cy) < 0.3 * sz
                and abs((d[3] - d[1]) - sz) <= 0.1 * sz + 1 and abs(d[5] - c[5]) <= 0.04]

    keep = [c for c in hollow if mates(c, hollow)]
    out = [(*c[:5], c[6] >= 0.1) for c in keep]
    out += [(*c[:5], True) for c in solid if c[6] >= 0.1 and mates(c, keep)]
    return out


def _blank(band: np.ndarray, start: int, step: int, limit: int) -> int:
    """從 start 往 step 方向數連續幾欄沒有墨水（最多 limit；碰到頁緣算全白）。"""
    for n in range(limit):
        x = start + n * step
        if not 0 <= x < band.shape[1]:
            return limit
        if band[:, x].any():
            return n
    return limit


def option_marks(img: np.ndarray, words: list[Word], dpi: int = DPI) -> tuple[list[Word], int]:
    """把選項控件補回文字：「新增修改取消 ○停用」→「◉新增 ○修改 ○取消 ○停用」。回傳 (新 words, 控件數)。

    控件常落在 OCR 合併的詞框裡，要把詞拆開：依影像上控件之間各段的墨水寬度估字數，切在最近的字界。
    """
    s = dpi / 72.0
    ctl = sorted(((x0 / s, y0 / s, x1 / s, y1 / s, OPTION[shape, on])
                  for x0, y0, x1, y1, shape, on in _controls(img, dpi)), key=lambda c: c[0])
    ctl = [c for c in ctl if _line_sized(c, words)]
    if not ctl:
        return words, 0
    gray = img.mean(axis=2) if img.ndim == 3 else img
    used: set[int] = set()
    out: list[Word] = []
    for w in words:
        inside = [i for i, c in enumerate(ctl) if i not in used
                  and w.x0 <= (c[0] + c[2]) / 2 <= w.x1 and w.y0 <= (c[1] + c[3]) / 2 <= w.y1]
        if not inside:
            out.append(w)
            continue
        used.update(inside)
        out += _split(w, [ctl[i] for i in inside], gray, s)
    out += [Word(c[0], c[1], c[2], c[3], c[4], 1.0) for i, c in enumerate(ctl) if i not in used]
    return out, len(ctl)


def _line_sized(c: tuple, words: list[Word]) -> bool:
    """控件跟所在那行字一樣高、垂直置中。字的部件（「品」下面的口）只有半個字高、「。」貼底線，
    同列又同大小，群組條件擋不掉，靠這一關排除。"""
    h, cy = c[3] - c[1], (c[1] + c[3]) / 2
    row = [w for w in words if w.y0 <= cy <= w.y1 and w.x1 > c[0] and w.x0 < c[2] + 4 * h]   # 包住它或右邊的標籤
    if not row:
        return False
    w = min(row, key=lambda w: max(w.x0 - c[2], 0))
    return h >= 0.55 * (w.y1 - w.y0) and abs(cy - (w.y0 + w.y1) / 2) <= 0.2 * (w.y1 - w.y0)


def _split(w: Word, cs: list[tuple], gray: np.ndarray, s: float) -> list[Word]:
    text = "".join(ch for ch in w.text if ch not in _OPTION_CHARS and not ch.isspace())
    if text in ("O", "o", "0"):                     # 整個詞就是控件本身被讀成字母
        text = ""
    wt = [1.0 if unicodedata.east_asian_width(ch) in "WF" else 0.5 for ch in text]
    # 控件之間每一段的墨水寬度（像素）
    y0, y1 = int(w.y0 * s), int(w.y1 * s)
    edges = [w.x0] + [v for c in cs for v in (c[0], c[2])] + [w.x1]
    ink = []
    for a, b in zip(edges[::2], edges[1::2]):
        cols = np.flatnonzero((gray[y0:y1, int(a * s) + 2:max(int(b * s) - 2, int(a * s) + 2)] < 160).any(axis=0))
        ink.append(float(cols[-1] - cols[0] + 1) if cols.size else 0.0)
    total = sum(ink)
    cum_w = np.cumsum([0.0] + wt)                    # 字界（以字寬權重計）
    cuts, acc = [], 0.0
    for v in ink[:-1]:
        acc += v
        target = acc / total * cum_w[-1] if total else 0.0
        cut = int(np.argmin(np.abs(cum_w - target)))
        cuts.append(max(cut, cuts[-1] if cuts else 0))
    bounds = [0] + cuts + [len(text)]
    out = []
    if bounds[1] > 0:
        out.append(Word(w.x0, w.y0, cs[0][0], w.y1, text[:bounds[1]], w.score))
    for k, c in enumerate(cs):
        x1 = cs[k + 1][0] if k + 1 < len(cs) else w.x1
        out.append(Word(c[0], w.y0, x1, w.y1, c[4] + text[bounds[k + 1]:bounds[k + 2]], w.score))
    return out
