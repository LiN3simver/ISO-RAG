"""PDF -> 每頁文字（文字層優先，不足才 OCR），結果依檔案 sha256 逐頁快取，可中斷續跑。

- 文字層要先通過品質檢查才採用：壞掉的 CMap 會給出 (cid:..)、亂碼，或「字都是中文但連起來不成話」。
- 其餘頁走自適應 OCR（learn.adaptive）：依版型挑策略、估品質、不夠好換策略；OCR_ADAPTIVE=0 退回單一設定。
- 文字層頁另存每行座標並 OCR 嵌入的圖片（ingest.layer）；OCR 頁做表格還原（ingest.tables）。
- 讀快取時套用人工校正（learn.review），快取檔本身不改。
- 舊格式快取在讀取時就地升級（upgrade）：只重抽文字層或重組表格，不重跑 OCR。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pypdfium2 as pdfium

from ..config import CACHE_DIR
from ..fileutil import atomic_write
from . import ocr

MIN_TEXT_CHARS = 20          # 文字層少於這個字數視為掃描頁
OCR_TAG = f"rapidocr-v6-{ocr.DPI}dpi"
ADAPTIVE = os.environ.get("OCR_ADAPTIVE", "1") != "0"


def sha256(path: Path) -> str:
    """檔案沒變（大小、修改時間）就用快取的值，不重讀整個 PDF。"""
    from ..fileutil import sha256 as cached
    return cached(path)


def _cache_path(digest: str, page: int) -> Path:
    return CACHE_DIR / "pages" / digest[:2] / digest / f"p{page + 1:03d}.{OCR_TAG}.json"


def load_page(digest: str, page_1based: int) -> dict | None:
    """快取中的頁面紀錄（已套用人工校正）；沒有快取 -> None。"""
    from ..learn import review
    cp = _cache_path(digest, page_1based - 1)
    if not cp.exists():
        return None
    return review.apply(json.loads(cp.read_text(encoding="utf-8")), digest)


def extract_page(pdf: Path, digest: str, page: int, log=None) -> dict:
    from ..learn import review
    cp = _cache_path(digest, page)
    if cp.exists():
        rec = json.loads(cp.read_text(encoding="utf-8"))
        new = upgrade(pdf, page, rec)
        if new is not rec:
            atomic_write(cp, json.dumps(new, ensure_ascii=False))
        return review.apply(new, digest)
    t0 = time.monotonic()
    doc = pdfium.PdfDocument(str(pdf))
    try:
        pg = doc[page]
        layer = pg.get_textpage().get_text_range() or ""
        w, h = pg.get_size()
    finally:
        doc.close()
    layer_ok, why = (False, "short") if len(layer.strip()) < MIN_TEXT_CHARS else _check_layer(layer)
    if layer_ok:
        from . import layer as layer_mod
        rec = layer_mod.text_page(str(pdf), page, layer)
    elif ADAPTIVE:
        from ..learn import adaptive
        rec = adaptive.ocr_pdf_page(str(pdf), page, sha=digest, doc=pdf.name, log=log)
    else:
        p = ocr.ocr_page(str(pdf), page)
        img = _rotated(ocr.render(str(pdf), page), p.rotation)
        words = p.words + ocr.filled_marks(img, [(w.x0, w.y0, w.x1, w.y1) for w in p.words])
        rec = {"page": page + 1, "source": "ocr", "rotation": p.rotation,
               "width": p.width, "height": p.height,
               "text": ocr.page_text(words), "words": [asdict(x) for x in words],
               "marks": len(words) - len(p.words)}
    if why != "short":
        rec["layer_check"] = why             # 有文字層但沒採用：記下原因
    rec["seconds"] = round(time.monotonic() - t0, 2)
    atomic_write(cp, json.dumps(rec, ensure_ascii=False))      # OCR 跑完才寫；中斷不會留下半個 JSON
    return review.apply(rec, digest)


def upgrade(pdf: Path, page: int, rec: dict) -> dict:
    """舊格式快取升級（不重跑整頁 OCR）。沒有要升級 -> 原物件。"""
    if rec.get("source") == "text" and rec.get("layer_v") != 2:
        from . import layer as layer_mod
        new = layer_mod.text_page(str(pdf), page)
        for k in ("layer_check", "seconds"):
            if k in rec:
                new[k] = rec[k]
        return new
    if rec.get("source") == "ocr" and rec.get("tables_v") != _tables_v():
        from ..learn import adaptive
        from dataclasses import fields
        words = [ocr.Word(**{f.name: w[f.name] for f in fields(ocr.Word)}) for w in rec["words"]]
        img = _rotated(ocr.render(str(pdf), page), rec.get("rotation", 0))
        lines = rec.get("text_lines", rec["text"])  # 逐行版（表格還原之前的文字）
        added = []
        if rec.get("text_chosen"):                  # 多策略合併補進來的行（沒有座標）
            have = set(rec["text_chosen"].splitlines())
            added = [ln for ln in lines.splitlines() if ln not in have]
        tt = adaptive.table_text(img, words, rec["width"], added)
        new = {**rec, "tables": tt["tables"], "tables_v": adaptive.TABLES_V}
        if tt["tables"]:
            new["text_lines"], new["text"] = lines, tt["text"]
        else:                                       # 新版沒有還原出表格：回到逐行版
            new.pop("text_lines", None)
            new["text"] = lines
        return new
    return rec


def _check_layer(text: str) -> tuple[bool, str]:
    from ..learn import adaptive, quality
    if not ADAPTIVE:
        return quality.text_layer_check(text)
    return adaptive.runtime().model.check_text_layer(text)


def _rotated(img, rotation: int):
    import numpy as np
    k = {0: 0, 90: 3, 180: 2, 270: 1}[rotation]      # 同 ocr._ocr
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def add_marks_to_cache(pdf: Path, log=print) -> int:
    """既有 OCR 快取補上表格勾選記號（不重跑 OCR；只重新渲染頁面找實心方塊）。已補過的頁略過。"""
    from dataclasses import fields
    digest = sha256(pdf)
    n = 0
    page = 0
    while (cp := _cache_path(digest, page)).exists():
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if rec["source"] == "ocr" and "marks" not in rec:
            words = [ocr.Word(**{f.name: w[f.name] for f in fields(ocr.Word)}) for w in rec["words"]]
            img = _rotated(ocr.render(str(pdf), page), rec["rotation"])
            marks = ocr.filled_marks(img, [(w.x0, w.y0, w.x1, w.y1) for w in words])
            rec["marks"] = len(marks)
            if marks:
                words += marks
                rec["words"] = [asdict(w) for w in words]
                rec["text"] = ocr.page_text(words)
                n += len(marks)
            atomic_write(cp, json.dumps(rec, ensure_ascii=False))
        page += 1
    return n


def extract_pdf(pdf: Path, log=None) -> tuple[str, list[dict]]:
    digest = sha256(pdf)
    doc = pdfium.PdfDocument(str(pdf))
    n = len(doc)
    doc.close()
    return digest, [extract_page(pdf, digest, i, log=log) for i in range(n)]


def _tables_v() -> int:
    from ..learn import adaptive
    return adaptive.TABLES_V
