"""文字層頁：每一行的座標，以及嵌入圖片裡的字（混合頁）。

文字層只給純文字：沒有座標就無法跟 OCR 頁一起做版型分析（頁首頁尾、版型分群）；
嵌在文件裡的圖片（流程圖、組織圖、系統截圖、掃描表單）沒有文字層，不另外 OCR 就查不到。

做法：
  - 行座標：get_text_range() 的第 i 個字對應第 i 個 char，每行取字框聯集；
    座標經 FPDF_PageToDevice 轉成「轉正後、左上原點」的 pt，跟 OCR 頁同一個座標系（含 /Rotate 90/270 的頁）。
  - 圖片：頁面上的影像物件（面積 ≥ IMG_MIN_AREA）裁出來 OCR；字的中心落在文字層行框內的丟掉
    （「可搜尋 PDF」整頁是圖、上面疊一層文字：只補文字層漏掉的部分）。
"""
from __future__ import annotations

import ctypes
from dataclasses import asdict

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from ..learn import quality
from . import ocr

IMG_MIN_AREA = 0.02        # 影像物件至少佔頁面 2%（logo、小圖示不 OCR）
IMG_PAD_PT = 4.0
MIN_UNCOVERED_PX = 150     # 100 dpi 下「像字、但文字層沒蓋到」的墨跡至少這麼多才 OCR（約 4 個字）
# Word 匯出的框底色、底圖也是影像物件，字其實在文字層上，整塊 OCR 很慢又幾乎沒有新字
# —— 先量圖裡有沒有文字層沒蓋到的字跡，只裁那一塊 OCR。
_K = 20                    # PageToDevice 用整數像素，放大 20 倍再除回來


class _Mapper:
    """PDF 使用者座標（左下原點、未轉正）-> 轉正後的頁面 pt（左上原點）。"""

    def __init__(self, page: pdfium.PdfPage):
        self.page = page
        self.w, self.h = page.get_size()
        self.sx, self.sy = int(self.w * _K), int(self.h * _K)
        self.dx, self.dy = ctypes.c_int(), ctypes.c_int()

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        pdfium_c.FPDF_PageToDevice(self.page.raw, 0, 0, self.sx, self.sy, 0, x, y,
                                   ctypes.byref(self.dx), ctypes.byref(self.dy))
        return self.dx.value / _K, self.dy.value / _K

    def box(self, l: float, b: float, r: float, t: float) -> tuple[float, float, float, float]:
        pts = [self(l, b), self(r, t), self(l, t), self(r, b)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)


def text_lines(page: pdfium.PdfPage) -> tuple[str, list[ocr.Word]]:
    """(文字層原文, 每行一個 Word：行框 + 行文字，score=1.0)。"""
    tp = page.get_textpage()
    text = tp.get_text_range() or ""
    n = tp.count_chars()
    m = _Mapper(page)
    lines: list[ocr.Word] = []
    start = 0
    for i in range(len(text) + 1):
        if i < len(text) and text[i] not in "\r\n":
            continue
        seg = text[start:i]
        boxes = []
        for k in range(start, min(i, n)):
            if text[k].isspace():
                continue
            l, b, r, t = tp.get_charbox(k)
            if r > l and t > b:
                boxes.append(m.box(l, b, r, t))
        if seg.strip() and boxes:
            lines.append(ocr.Word(min(x[0] for x in boxes), min(x[1] for x in boxes),
                                  max(x[2] for x in boxes), max(x[3] for x in boxes), seg.strip(), 1.0))
        start = i + 1
    return text, lines


def image_regions(page: pdfium.PdfPage) -> list[tuple[float, float, float, float]]:
    """頁面上夠大的影像物件（轉正後 pt）。"""
    m = _Mapper(page)
    area = m.w * m.h
    out = []
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
        try:
            l, b, r, t = obj.get_bounds()
        except Exception:  # noqa: BLE001
            continue
        x0, y0, x1, y1 = m.box(l, b, r, t)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, m.w), min(y1, m.h)
        if (x1 - x0) * (y1 - y0) >= IMG_MIN_AREA * area:
            out.append((x0, y0, x1, y1))
    return _dedup(out)


def _dedup(boxes):
    """大部分（≥80%）落在另一個較大框裡的框不重複 OCR（疊在一起的截圖）。"""
    def inter(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    out = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        area = (b[2] - b[0]) * (b[3] - b[1])
        if not any(inter(o, b) >= 0.8 * area for o in out):
            out.append(b)
    return out


def uncovered_regions(small: np.ndarray, regions, layer_boxes, dpi: int = quality.ANALYSIS_DPI):
    """每個圖片區塊裡「像字、文字層沒蓋到」的墨跡外框（pt）；墨跡太少的區塊略過。"""
    import cv2
    text, _, _ = quality.ink_layers(small)
    s = dpi / 72.0
    cover = np.zeros(text.shape, np.uint8)
    for x0, y0, x1, y1 in layer_boxes:
        cv2.rectangle(cover, (int(x0 * s) - 2, int(y0 * s) - 2), (int(x1 * s) + 2, int(y1 * s) + 2), 1, -1)
    unc = text & ~cover.astype(bool)
    out = []
    for x0, y0, x1, y1 in regions:
        a, b, c, d = int(x0 * s), int(y0 * s), int(x1 * s) + 1, int(y1 * s) + 1
        sub = unc[b:d, a:c]
        if int(sub.sum()) < MIN_UNCOVERED_PX:
            continue
        ys, xs = np.nonzero(sub)
        out.append((max(x0, (a + xs.min()) / s - 8), max(y0, (b + ys.min()) / s - 8),
                    min(x1, (a + xs.max()) / s + 8), min(y1, (b + ys.max()) / s + 8)))
    return out


def ocr_regions(img: np.ndarray, regions, layer_boxes, dpi: int = ocr.DPI) -> list[ocr.Word]:
    """img：轉正後整頁影像（dpi）。逐區塊裁切 OCR，回傳頁面座標（pt）的字；中心落在文字層行框內的丟掉。"""
    eng = ocr.engine()
    s = dpi / 72.0
    H, W = img.shape[:2]
    out: list[ocr.Word] = []
    for x0, y0, x1, y1 in regions:
        a, b = max(int((x0 - IMG_PAD_PT) * s), 0), max(int((y0 - IMG_PAD_PT) * s), 0)
        c, d = min(int((x1 + IMG_PAD_PT) * s), W), min(int((y1 + IMG_PAD_PT) * s), H)
        if c - a < 20 or d - b < 20:
            continue
        crop = np.ascontiguousarray(img[b:d, a:c])
        p = ocr._ocr(crop, 0, dpi, eng)
        for w in p.words:
            w = ocr.Word(w.x0 + a / s, w.y0 + b / s, w.x1 + a / s, w.y1 + b / s, w.text, w.score)
            cx, cy = (w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2
            if any(lx0 - 1 <= cx <= lx1 + 1 and ly0 - 1 <= cy <= ly1 + 1 for lx0, ly0, lx1, ly1 in layer_boxes):
                continue
            if any(abs(o.x0 - w.x0) < 2 and abs(o.y0 - w.y0) < 2 and o.text == w.text for o in out):
                continue          # 重疊的影像物件
            out.append(w)
    return out


def text_page(pdf: str, page_no: int, layer: str | None = None, *, hybrid: bool = True) -> dict:
    """文字層頁的快取紀錄（page_no 從 0 起算）。text = 文字層原文 +（有的話）圖片裡讀到的字。"""
    doc = pdfium.PdfDocument(pdf)
    try:
        pg = doc[page_no]
        raw, lines = text_lines(pg)
        w, h = pg.get_size()
        regions = image_regions(pg) if hybrid else []
    finally:
        doc.close()
    layer = raw if layer is None else layer
    words = [dict(asdict(x), src="layer") for x in lines]
    img_words: list[ocr.Word] = []
    boxes = [(x.x0, x.y0, x.x1, x.y1) for x in lines]
    if regions:
        regions = uncovered_regions(ocr.render(pdf, page_no, dpi=quality.ANALYSIS_DPI), regions, boxes)
    if regions:
        img = ocr.render(pdf, page_no)
        img_words = ocr_regions(img, regions, boxes)
    text = layer
    if img_words:
        # 圖片裡的字依圖分段接在文字層後面（同一頁，引用頁碼不變）；截圖裡的表單也還原表格列
        from . import tables
        grid = tables.detect(img)
        parts, left = [], list(img_words)
        for r in regions:                  # 每個字只歸一個圖（部分重疊的圖不重複輸出）
            ws = [x for x in left if r[0] - IMG_PAD_PT <= (x.x0 + x.x1) / 2 <= r[2] + IMG_PAD_PT
                  and r[1] - IMG_PAD_PT <= (x.y0 + x.y1) / 2 <= r[3] + IMG_PAD_PT]
            left = [x for x in left if x not in ws]
            if ws:
                parts.append(tables.assemble(ws, grid, w)[0])
        text = layer.rstrip() + "\n" + "\n".join(parts)
        words += [dict(asdict(x), src="image") for x in img_words]
    return {"page": page_no + 1, "source": "text", "rotation": 0, "width": w, "height": h, "text": text,
            "words": words, "layer_v": 2, "images": len(regions), "image_words": len(img_words)}
