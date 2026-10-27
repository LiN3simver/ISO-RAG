"""人工確認（資料飛輪）：品質低或新版型的頁 → 匯出頁面圖 + OCR 文字 → 人改好 → 匯入。

  learn review --export N   data/learn/review/ 產生 <文件>_p003.png（150 dpi）與同名 .txt
                            品質最低的頁先匯出
  （直接改 .txt 存檔；沒改就代表 OCR 正確）
  learn accept              讀回 .txt：
                              - 有改 → data/learn/corrections/<sha>/p003.txt（extract 讀快取時套用）
                              - 沒改 → 確認
                              兩種都進 gold（origin=human），下次 learn fit 就是真實掃描頁的訓練資料
原始 OCR 快取不改寫（raw 不可變，校正另存並標記）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from .store import LEARN_DIR, Store

REVIEW_DIR = LEARN_DIR / "review"
CORR_DIR = LEARN_DIR / "corrections"


def correction_path(sha: str, page: int) -> Path:
    return CORR_DIR / sha / f"p{page:03d}.txt"


def apply(rec: dict, sha: str) -> dict:
    """extract 讀頁面紀錄時套用人工校正（快取檔本身不動）。"""
    p = correction_path(sha, rec["page"])
    if not p.exists():
        return rec
    return {**rec, "raw_text": rec["text"], "text": p.read_text(encoding="utf-8"), "corrected": True}


def _norm(s: str) -> str:
    return "\n".join(ln.rstrip() for ln in s.strip().splitlines())


def pending(store: Store) -> list[dict]:
    """待確認的頁，品質由低到高。"""
    return sorted(store.review_rows("open"), key=lambda r: r["quality"])


def export(store: Store, limit: int = 20, log=print) -> list[Path]:
    from ..ingest import ocr
    from ..ingest.extract import load_page

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for r in pending(store)[:limit]:
        if not r["pdf"] or not Path(r["pdf"]).exists():
            log(f"  略過 {r['doc']} p{r['page']}：找不到 PDF")
            continue
        rec = load_page(r["sha"], r["page"])
        if rec is None:
            continue
        base = REVIEW_DIR / f"{Path(r['pdf']).stem}_p{r['page']:03d}"
        img = ocr.render(r["pdf"], r["page"] - 1, dpi=150)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        base.with_suffix(".png").write_bytes(buf.tobytes())        # cv2.imwrite 不吃中文路徑
        base.with_suffix(".txt").write_text(rec["text"], encoding="utf-8")
        base.with_suffix(".json").write_text(json.dumps(
            {"sha": r["sha"], "page": r["page"], "doc": r["doc"], "pdf": r["pdf"], "reason": r["reason"],
             "quality": r["quality"], "ocr_text": rec["text"]}, ensure_ascii=False, indent=1), encoding="utf-8")
        out.append(base.with_suffix(".txt"))
        log(f"  {base.name}  原因={r['reason']} 品質={r['quality']:.3f}")
    return out


def accept(store: Store, log=print) -> dict[str, int]:
    done = REVIEW_DIR / "done"
    n = {"corrected": 0, "confirmed": 0}
    for meta_p in sorted(REVIEW_DIR.glob("*.json")):
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        txt_p = meta_p.with_suffix(".txt")
        if not txt_p.exists():
            continue
        text = _norm(txt_p.read_text(encoding="utf-8"))
        changed = text != _norm(meta["ocr_text"])
        if changed:
            cp = correction_path(meta["sha"], meta["page"])
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(text, encoding="utf-8")
        store.gold_put(meta["sha"], meta["page"], "human", meta["doc"], text)
        status = "corrected" if changed else "confirmed"
        store.review_set(meta["sha"], meta["page"], status)
        n[status] += 1
        done.mkdir(parents=True, exist_ok=True)
        for p in (meta_p, txt_p, meta_p.with_suffix(".png")):
            if p.exists():
                shutil.move(str(p), done / p.name)
        log(f"  {txt_p.name}: {status}")
    store.commit()
    return n
