"""自監督訓練資料：有文字層的 PDF 頁 = 免費的標準答案。

  文字層頁 → 渲染 300 dpi → 模擬掃描劣化（AUGS）→ 每個 OCR 策略跑一次 → 與文字層比對得到正確率
得到的 (特徵, 正確率) 用來訓練品質模型；「哪個策略在哪種劣化下贏」用來當新版型的策略先驗。
同一份文件的頁在交叉驗證時整份留出（trainer.py），才量得到「沒看過的文件」上的表現。

輕度模糊/雜訊對 OCR 幾乎沒影響，所以劣化多用重度的（低解析、強模糊、褪色、重壓縮、
紅色受控章壓字、歪斜）——品質模型要看過真的會壞的樣本才學得到。
樣本以 (sha, 頁, 劣化, 策略) 為 key 存進 learn.sqlite，可中斷續跑。
"""
from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np
import pypdfium2 as pdfium

from ..config import PDF_DIR
from ..fileutil import display_name
from ..ingest import ocr
from ..ingest.extract import sha256
from . import adaptive, quality
from .store import Store

AUGS = ("clean", "lowres", "blur", "ghost", "jpeg", "stamp", "skew")
STAMP_BGR = (50, 50, 225)    # 紅色（影像是 pdfium 的 BGR 順序）
STAMP_V = 2                  # 紅章樣本的版本：版本不符時 learn selfsup 會清掉 aug='stamp' 的樣本重做
MIN_GOLD_CHARS = 200


def degrade(img: np.ndarray, aug: str, rng: np.random.Generator) -> np.ndarray:
    im = img.copy()
    h, w = im.shape[:2]
    if aug == "lowres":                      # 75~100 dpi 的掃描檔放大回來
        f = rng.uniform(0.25, 0.33)
        im = cv2.resize(cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h))
    elif aug == "blur":
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(2.5, 3.8))
    elif aug == "ghost":                     # 碳粉不足、淡到快看不見
        a = rng.uniform(0.12, 0.25)
        im = (255 - (255 - im.astype(np.float32)) * a + rng.normal(0, 5, im.shape))
        im = np.clip(im, 0, 255).astype(np.uint8)
    elif aug == "jpeg":
        f = rng.uniform(0.3, 0.4)
        sm = cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(8, 16))])
        im = cv2.resize(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), (w, h))
    elif aug == "stamp":                     # 紅色章（相乘：章底下的黑字還是黑的，跟真的蓋章一樣）
        ov = np.full_like(im, 255)
        for _ in range(int(rng.integers(8, 15))):
            cx, cy = int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.15, 0.9) * h)
            r = int(rng.uniform(0.05, 0.09) * w)
            cv2.circle(ov, (cx, cy), r, STAMP_BGR, int(r * 0.12))
            cv2.putText(ov, "QC PASS", (cx - r, cy + r // 6), cv2.FONT_HERSHEY_SIMPLEX, r / 90, STAMP_BGR,
                        max(int(r * 0.06), 3))
        im = (im.astype(np.float32) * ov.astype(np.float32) / 255).astype(np.uint8)
    elif aug == "skew":
        ang = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        im = cv2.warpAffine(im, m, (w, h), borderValue=(255, 255, 255))
        im = np.clip(im.astype(np.float32) + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8)
    elif aug != "clean":
        raise ValueError(aug)
    return np.ascontiguousarray(im)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def gold_pages(store: Store, log=print) -> list[tuple]:
    """data/pdf 中文字層通過檢查（只用規則，不用 LM，避免拿自己驗自己）的頁 -> gold(origin=textlayer)。"""
    out = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        sha = sha256(pdf)
        doc = pdfium.PdfDocument(str(pdf))
        try:
            for i in range(len(doc)):
                text = doc[i].get_textpage().get_text_range() or ""
                if len(text.strip()) >= MIN_GOLD_CHARS and quality.text_layer_check(text)[0]:
                    store.gold_put(sha, i + 1, "textlayer", pdf.name, text)
                    out.append((pdf, sha, i, text))
        finally:
            doc.close()
    store.commit()
    log(f"文字層標準答案：{len(out)} 頁，{len({x[1] for x in out})} 份文件")
    return out


def _spread(pages: list[tuple], n: int) -> list[tuple]:
    """各文件輪流取頁，讓樣本不集中在同一份文件。"""
    by_doc: dict[str, list] = {}
    for p in pages:
        by_doc.setdefault(p[1], []).append(p)
    out, k = [], 0
    while len(out) < n and any(k < len(v) for v in by_doc.values()):
        for v in by_doc.values():
            # 每份文件從第 2 頁開始挑、間隔取，避開大量相同表頭的連續頁
            idx = [i for i in range(len(v)) if i % 2 == 1] + [i for i in range(len(v)) if i % 2 == 0]
            if k < len(idx) and len(out) < n:
                out.append(v[idx[k]])
        k += 1
    return out


def migrate_stamp(store: Store, log=print):
    """舊版本的印章樣本（顏色通道順序錯誤）清掉重做；以 meta.stamp_v 判斷是否已遷移。"""
    if store.meta_get("stamp_v") == str(STAMP_V):
        return
    n = store.db.execute("DELETE FROM selfsup WHERE aug = 'stamp'").rowcount
    store.db.execute("DELETE FROM cond WHERE aug = 'stamp'")
    store.meta_put("stamp_v", str(STAMP_V))
    store.commit()
    log(f"清掉舊的藍章樣本 {n} 筆（改成紅章重做）")


def backfill_cond(store: Store, log=print) -> int:
    """已有自監督樣本、還沒有掃描狀況特徵的組：重新渲染＋同一個種子劣化，算特徵（不跑 OCR）。"""
    from . import context
    have = store.cond_rows()
    groups = {(r[0], r[1], r[2]) for r in store.db.execute("SELECT DISTINCT sha, page, aug FROM selfsup")}
    todo = sorted(groups - set(have))
    pdfs = {sha256(p): p for p in PDF_DIR.glob("*.pdf")} if todo else {}
    n = 0
    for sha, page, aug in todo:
        pdf = pdfs.get(sha)
        if pdf is None:
            continue
        img = degrade(ocr.render(str(pdf), page - 1), aug, np.random.default_rng(_seed(sha, page - 1, aug)))
        store.cond_put(sha, page, aug, context.condition_features(img))
        n += 1
    store.commit()
    if n:
        log(f"補算掃描狀況特徵：{n} 組")
    return n


def run(store: Store, *, pages: int = 12, augs: int = 3, variants: list[str] | None = None,
        only: set[str] | None = None, log=print) -> int:
    """產生自監督樣本（可中斷續跑）。only：只從這些 PDF（sha）挑頁（例如只做新加入的文件）。"""
    from . import context
    migrate_stamp(store, log)
    variants = variants or list(adaptive.VARIANTS)
    gold = gold_pages(store, log)
    if only:
        gold = [g for g in gold if g[1] in only]
    sample = _spread(gold, pages)
    todo = [(p, a) for p in sample for a in ["clean"] + _pick(p, augs - 1)]
    log(f"自監督：{len(sample)} 頁 × {augs} 種劣化 × {len(variants)} 策略（已做過的略過）")
    n, t0 = 0, time.monotonic()
    for (pdf, sha, pno, gold), aug in todo:
        if all(store.selfsup_has(sha, pno + 1, aug, v) for v in variants):
            continue
        img = degrade(ocr.render(str(pdf), pno), aug, np.random.default_rng(_seed(sha, pno, aug)))
        small = quality.analysis_image(img, ocr.DPI)
        store.cond_put(sha, pno + 1, aug, context.condition_features(img))
        row = []
        for v in variants:
            if store.selfsup_has(sha, pno + 1, aug, v):
                continue
            a = adaptive.run_attempt(img, small, 0, v)
            acc = quality.accuracy(a.text, gold)
            store.selfsup_put(sha, pno + 1, aug, v, acc, round(a.seconds, 2), a.base, a.text)
            store.commit()
            row.append(f"{v}={acc:.3f}")
            n += 1
        store.commit()
        log(f"  {display_name(pdf.name)[:28]:28} p{pno + 1:02d} {aug:7} " + " ".join(row)
            + f"  ({time.monotonic() - t0:.0f}s)")
    return n


def _pick(p: tuple, k: int) -> list[str]:
    rest = list(AUGS[1:])
    rng = np.random.default_rng(_seed(p[1], p[2], "augs"))
    return [str(x) for x in rng.choice(rest, size=min(k, len(rest)), replace=False)]
