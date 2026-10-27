"""命令列入口：python -m iso_rag.cli <command>

  extract  data/pdf 全部 PDF 做文字擷取／自適應 OCR（逐頁快取）
  build    擷取 → 切段 → embedding + FTS5 → 寫新索引並切換
  ask      問一題（--no-llm 只看檢索）
  search   只看檢索結果與分數
  learn    OCR 自主學習：status / scan / selfsup / cond / fit / promote / rollback / review / accept / merge
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    if not sys.stdout.isatty():          # 導到檔案/管線時 Windows 預設 cp950，中文會變亂碼
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="iso_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("extract")
    p.add_argument("--marks", action="store_true", help="既有 OCR 快取補上表格勾選記號 ■（不重跑 OCR）")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--no-llm", action="store_true")
    p = sub.add_parser("search")
    p.add_argument("question")
    p = sub.add_parser("learn")
    lsub = p.add_subparsers(dest="lcmd", required=True)
    lsub.add_parser("status", help="版型、策略、模型、待確認清單總覽")
    lp = lsub.add_parser("scan", help="用既有 OCR 快取建立版型與品質分數（不重跑 OCR）")
    lp.add_argument("--reset", action="store_true")
    lp = lsub.add_parser("selfsup", help="文字層自監督樣本（可續跑）")
    lp.add_argument("--pages", type=int, default=12)
    lp.add_argument("--augs", type=int, default=3)
    lp.add_argument("--variants", default="")
    lsub.add_parser("cond", help="自監督樣本補算掃描狀況特徵（不跑 OCR）")
    lsub.add_parser("fit", help="訓練 -> candidate + 評測閘門")
    lp = lsub.add_parser("promote")
    lp.add_argument("--version")
    lp.add_argument("--force", action="store_true")
    lsub.add_parser("rollback")
    lp = lsub.add_parser("review", help="待人工確認的頁（--export 產生圖檔與文字檔）")
    lp.add_argument("--export", type=int, default=0)
    lsub.add_parser("accept", help="匯入人工確認／校正")
    lsub.add_parser("merge", help="既有快取補做多策略合併（不重跑 OCR）")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        from .index.build import build
        build()
        return 0
    if args.cmd == "extract":
        return _extract(args.marks)
    if args.cmd == "learn":
        return _learn(args)

    from .index.store import Reader
    from .query.router import analyze
    idx = Reader()
    a = analyze(args.question, idx)
    print(f"[intent={a.intent} codes={a.codes} docs={a.doc_nos} attrs={[x[1] for x in a.attrs]} "
          f"iso={a.iso_system} type={a.doc_type} attr_only={a.attr_only}]")
    if args.cmd == "search":
        from .answer.answer import cite_label
        from .query.search import search
        hits = search(a, idx)
        print(f"[focus={a.focus_docs} term={a.focus_term!r} focused={a.focused}]")
        for h in hits:
            print(f"{h.score:.4f} {cite_label(idx, h.chunk):35} {h.why} | {h.chunk.text[:70]!r}")
        return 0
    from .answer.answer import answer
    res = answer(args.question, idx, a, use_llm=not args.no_llm)
    print(f"[mode={res.mode} {res.seconds}s llm={res.llm_seconds:.1f}s grounded={res.grounded}]\n{res.text}")
    if res.fulltext_text:
        print(f"[全文核對] {res.fulltext_text.splitlines()[0]}")
    for x in res.verify_log:
        print(f"[驗證] {x}")
    return 0


def _extract(marks: bool = False) -> int:
    import time

    from .config import PDF_DIR
    from .ingest.extract import add_marks_to_cache, extract_pdf

    t0 = time.monotonic()
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        t = time.monotonic()
        if marks:
            print(f"{pdf.name[:40]:40} 補上 ■ {add_marks_to_cache(pdf)} 個", flush=True)
            continue
        _, pages = extract_pdf(pdf, log=print)
        ocr_n = sum(p["source"] == "ocr" for p in pages)
        new = [p["page"] for p in pages if p.get("new_layout")]
        fixed = [p["page"] for p in pages if p.get("corrected")]
        print(f"{pdf.name[:40]:40} pages={len(pages):3} ocr={ocr_n:3} "
              f"{'新版型頁=' + str(new) + ' ' if new else ''}{'人工校正頁=' + str(fixed) + ' ' if fixed else ''}"
              f"{time.monotonic() - t:6.1f}s", flush=True)
    print(f"TOTAL {time.monotonic() - t0:.0f}s")
    return 0


def _learn(args) -> int:
    from .learn import review, selfsup, trainer
    from .learn.store import Store

    store = Store()
    log = lambda s: print(s, flush=True)  # noqa: E731
    if args.lcmd == "status":
        trainer.status(store, log)
    elif args.lcmd == "scan":
        trainer.scan(store, reset=args.reset, log=log)
    elif args.lcmd == "selfsup":
        selfsup.run(store, pages=args.pages, augs=args.augs,
                    variants=[v for v in args.variants.split(",") if v] or None, log=log)
    elif args.lcmd == "cond":
        selfsup.migrate_stamp(store, log)
        print(f"補算 {selfsup.backfill_cond(store, log)} 組")
    elif args.lcmd == "fit":
        v = trainer.fit(store, log)
        r = next(r for r in store.model_rows() if r["version"] == v)
        print(json.dumps({"version": v, "metrics": r["metrics"], "gate": r["gate"]}, ensure_ascii=False, indent=1))
        print(f"\n閘門：{r['gate']['status']}" + ("  -> 可執行 learn promote" if r["gate"]["status"] == "PASS" else ""))
    elif args.lcmd == "promote":
        try:
            print(f"上線：{trainer.promote(store, args.version, force=args.force)}")
        except RuntimeError as e:
            print(e)
            return 1
    elif args.lcmd == "rollback":
        print(f"退回：{trainer.rollback(store) or 'heuristic（沒有更早的版本）'}")
    elif args.lcmd == "review":
        rows = review.pending(store)
        for r in rows:
            print(f"  {r['doc'][:36]:36} p{r['page']:03d} 原因={r['reason']:22} 品質={r['quality']:.3f}")
        print(f"共 {len(rows)} 頁待確認")
        if args.export:
            paths = review.export(store, args.export, log)
            print(f"已匯出 {len(paths)} 頁到 {review.REVIEW_DIR}；改好 .txt 後執行 learn accept")
    elif args.lcmd == "accept":
        print(review.accept(store, log))
    elif args.lcmd == "merge":
        from .learn import adaptive
        print(adaptive.merge_cache(store, log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
