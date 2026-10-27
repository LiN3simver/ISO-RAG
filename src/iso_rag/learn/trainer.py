"""訓練、評測閘門、上線/回滾，以及用既有快取建立版型（scan）。

  learn scan      既有 OCR 快取 -> 版型指紋、品質分數、待確認清單（不重跑 OCR）
  learn fit       語言模型 + 品質模型（residual）-> candidate，附評測指標與閘門結果
  learn promote   閘門 PASS 才換上（--force 可強制）；learn rollback 退回上一版

評測閘門（門檻寫死；指標缺漏是 INCOMPLETE 不是 PASS）：
  - 留一份文件交叉驗證：該文件的頁連同它的文字都不進語言模型與回歸，量的是「沒看過的文件」
  - regret：同一頁的多個策略中，模型挑的那個比真正最好的差多少（自適應 OCR 實際靠這個做決定）
  - spearman：預測品質與真實正確率的等級相關
  - 文字層檢查：真文字層不能誤殺；合成的壞文字層（拉丁亂碼、Big5/GBK 錯置、打亂、私用區、cid）要抓到
  - real_pass_rate：現有的正常掃描頁不能被大量判成「需人工確認」
  - 情境模型（learn.context）：留一份文件，用「重播」模擬新版型的自適應迴圈，
    比較「只看先驗順序」與「先看掃描狀況」的最後正確率、嘗試次數，不能比較差
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict

import numpy as np

from ..config import CACHE_DIR, PDF_DIR
from ..fileutil import display_name
from ..ingest import ocr
from ..ingest.extract import load_page, sha256
from . import adaptive, context, layout, quality, selfsup
from .charlm import CharLM, is_cjk
from .quality import FEATURES, QualityModel
from .store import Store

LM_FEATS = ("lm_logp", "oov2")
BASE_FEATS = tuple(f for f in FEATURES if f not in LM_FEATS)
TARGET, REVIEW = quality.HEURISTIC_TARGET, quality.HEURISTIC_REVIEW   # residual 的絕對水準跟 heuristic 相同
LM_MARGIN = 1.0
GAIN_MARGIN = 0.3
UNSEEN_MARGIN = 0.02


# ---------------------------------------------------------------- 既有快取 -> 版型
def scan(store: Store, *, reset: bool = False, log=print) -> int:
    if reset:
        for t in ("profiles", "policy"):
            store.db.execute(f"DELETE FROM {t}")
        store.db.execute("DELETE FROM obs WHERE origin = 'scan'")
        store.commit()
    selfsup.gold_pages(store, log)
    rt = adaptive.Runtime(store)       # 同一個連線：版型更新與觀測寫入不能互卡
    n = 0
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        sha = sha256(pdf)
        done = {r[0] for r in store.db.execute("SELECT page FROM obs WHERE sha=? AND origin='scan'", (sha,))}
        page = 0
        line = []
        while (rec := load_page(sha, page + 1)) is not None:
            page += 1
            if rec["source"] != "ocr" or page in done:
                continue
            small = adaptive.rotate(ocr.render(str(pdf), page - 1, dpi=quality.ANALYSIS_DPI), rec["rotation"])
            prof, sim, created = rt.profiles.assign(layout.fingerprint(small), pdf.name, "scan")
            boxes = [(w["x0"], w["y0"], w["x1"], w["y1"]) for w in rec["words"]]
            base = quality.base_features(rec["words"], rec.get("raw_text", rec["text"]),
                                         ink_unc=quality.ink_uncovered(small, boxes))
            feats = rt.model.features(base, rec.get("raw_text", rec["text"]))
            q = rt.model.predict(feats)
            store.obs_put(sha=sha, page=page, variant=rec.get("variant", "base"), origin="scan", profile=prof.id,
                          quality=q, seconds=rec.get("seconds", 0), chars=len(rec["text"]), text_sha="",
                          feats={**feats, "fp_sim": round(sim, 4)}, text=rec.get("raw_text", rec["text"]), chosen=1)
            if q < rt.model.review:
                store.review_put(sha, page, pdf.name, str(pdf), "low_quality", q)
            store.commit()
            line.append(f"p{page}:{prof.id}{'*' if created else ''}:{q:.2f}")
            n += 1
        store.commit()
        if line:
            log(f"  {display_name(pdf.name)[:30]:30} " + " ".join(line))
    log(f"scan：{n} 頁，版型 {len(rt.profiles.items)} 個（* = 該頁開了新版型）")
    return n


# ---------------------------------------------------------------- 語料與樣本
def _ocr_corpus() -> dict[str, list[str]]:
    """快取中的 OCR 頁，只取信心 ≥ 0.9 的框（語言模型的自我訓練語料）。"""
    out: dict[str, list[str]] = defaultdict(list)
    for p in (CACHE_DIR / "pages").glob("*/*/p*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("source") == "ocr":
            out[p.parent.name].append("\n".join(w["text"] for w in rec["words"] if w["score"] >= 0.9))
    return out


def corpus_lm(store: Store, exclude: set[str] = frozenset(), ocr_corpus=None) -> CharLM:
    lm = CharLM()
    for g in store.gold_rows():
        if g["sha"] not in exclude:
            lm.add(g["text"])
    for sha, texts in (ocr_corpus if ocr_corpus is not None else _ocr_corpus()).items():
        if sha not in exclude:
            for t in texts:
                lm.add(t)
    return lm.finish()


def samples(store: Store) -> list[dict]:
    S = [{"group": (r["sha"], r["page"], r["aug"]), "doc": r["sha"], "variant": r["variant"], "acc": r["acc"],
          "seconds": r["seconds"], "base": r["feats"], "text": r["text"], "kind": "selfsup"}
         for r in store.selfsup_rows()]
    human = {(g["sha"], g["page"]): g["text"] for g in store.gold_rows("human")}
    for o in store.obs_rows():
        if (o["sha"], o["page"]) in human:
            S.append({"group": (o["sha"], o["page"], "real"), "doc": o["sha"], "variant": o["variant"],
                      "acc": quality.accuracy(o["text"], human[(o["sha"], o["page"])]), "seconds": o["seconds"],
                      "base": {k: o["feats"][k] for k in BASE_FEATS}, "text": o["text"], "kind": "human"})
    return S


def _matrix(S: list[dict], lm: CharLM) -> np.ndarray:
    m = QualityModel(lm=lm)
    return np.array([[m.features(s["base"], s["text"])[k] for k in FEATURES] for s in S], float)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _regret(S: list[dict], pred: np.ndarray) -> tuple[float, float]:
    """(平均 regret, 模型挑中策略的平均正確率)。只算有 ≥2 個策略的頁。"""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(S):
        groups[s["group"]].append(i)
    reg, acc = [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        pick = max(idx, key=lambda i: pred[i])
        best = max(S[i]["acc"] for i in idx)
        reg.append(best - S[pick]["acc"])
        acc.append(S[pick]["acc"])
    return (float(np.mean(reg)), float(np.mean(acc))) if reg else (float("nan"), float("nan"))


# ---------------------------------------------------------------- 文字層檢查的合成負例
def _corruptions(text: str, rng: np.random.Generator) -> dict[str, str]:
    chars = list(text)
    rng.shuffle(chars)
    return {
        "latin1": text.encode("utf-8").decode("latin-1"),
        "big5_as_gbk": text.encode("big5", "ignore").decode("gbk", "ignore"),
        "shuffle": "".join(chars),
        "pua": "".join(chr(0xE000 + ord(c) % 6400) if is_cjk(c) else c for c in text),
        "cid": " ".join(f"(cid:{ord(c) % 900})" for c in text[:200]),
    }


# ---------------------------------------------------------------- fit / gate
def fit(store: Store, log=print) -> str:
    S = samples(store)
    if not S:
        raise RuntimeError("沒有訓練樣本：先跑 learn selfsup（文字層自監督）或 learn accept（人工校正）")
    y = np.array([s["acc"] for s in S])
    docs = sorted({s["doc"] for s in S})
    corpus = _ocr_corpus()
    golds = store.gold_rows("textlayer")
    rng = np.random.default_rng(0)
    pred_l, rank_l, pred_h = np.zeros(len(S)), np.zeros(len(S)), np.zeros(len(S))
    tl_scores, bad = [], defaultdict(list)            # 文字層：留出文件時的 LM 分數
    tl_gains: list[float] = []                       # 文字層：留出文件時的字序增益
    tl_unseen: list[float] = []                      # 文字層：留出文件時「沒看過的字」比例
    for d in sorted(set(docs) | {g["sha"] for g in golds}):
        lm = corpus_lm(store, {d}, corpus)
        te = np.array([s["doc"] == d for s in S])
        if te.any():
            X = _matrix(S, lm)
            mean, std, w = quality.fit_residual(X[~te], y[~te], [s["group"] for s, t in zip(S, te) if not t])
            m = QualityModel("residual", mean=mean, std=std, w=w)
            pred_l[te] = [m.predict(dict(zip(FEATURES, x))) for x in X[te]]
            rank_l[te] = [m.rank(dict(zip(FEATURES, x))) for x in X[te]]
            pred_h[te] = [quality.heuristic(dict(zip(FEATURES, x))) for x in X[te]]
        for g in golds:
            if g["sha"] == d:
                tl_scores.append(lm.score(g["text"])[0])
                if (og := lm.order_gain(g["text"])) is not None:
                    tl_gains.append(og)
                if (uc := lm.unseen_chars(g["text"])) is not None:
                    tl_unseen.append(uc)
                for k, t in _corruptions(g["text"], rng).items():
                    bad[k].append((t, lm))
    lm_floor = (min(tl_scores) - LM_MARGIN) if tl_scores else None
    gain_floor = (min(tl_gains) - GAIN_MARGIN) if tl_gains else None
    unseen_ceil = (max(tl_unseen) + UNSEEN_MARGIN) if tl_unseen else None
    detect = {k: float(np.mean([not quality.text_layer_check(t, lm, lm_floor, gain_floor, unseen_ceil)[0]
                                for t, lm in v])) for k, v in bad.items()}

    lm = corpus_lm(store, ocr_corpus=corpus)
    X = _matrix(S, lm)
    mean, std, w = quality.fit_residual(X, y, [s["group"] for s in S])
    scan_rows = store.obs_rows("origin = 'scan'")
    tmp = QualityModel(lm=lm)
    R = np.array([[tmp.features({k: o["feats"][k] for k in BASE_FEATS}, o["text"])[k] for k in FEATURES]
                  for o in scan_rows]) if scan_rows else None
    if R is not None:
        # 修正量以「真實掃描頁的平均」為零點：一般的掃描頁維持 heuristic 的水準，只修正偏離常態的頁
        mean = R.mean(0)
    ink_ref = (float(np.percentile(R[:, FEATURES.index("ink_uncovered")], 90)) if R is not None
               else quality.INK_REF_DEFAULT)
    prior = _prior(S)
    groups = ctx_groups(store, S)
    ctx = context.ContextModel.fit(list(groups.values())) if groups else None
    ctx_m = ctx_cv(S, rank_l, pred_l, groups, ink_ref) if groups else {}
    version = "q" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = QualityModel("residual", version=version, lm=lm, mean=mean, std=std, w=w, target=TARGET,
                         review=REVIEW, lm_floor=lm_floor, prior=prior, ink_ref=ink_ref, gain_floor=gain_floor,
                         unseen_ceil=unseen_ceil,
                         ctx=ctx if ctx is not None and ctx.n else None)
    real = [model.predict(dict(zip(FEATURES, x))) for x in R] if R is not None else []

    reg_l, acc_l = _regret(S, rank_l)
    reg_h, acc_h = _regret(S, pred_h)
    base_acc = [s["acc"] for s in S if s["variant"] == adaptive.DEFAULT and
                sum(1 for t in S if t["group"] == s["group"]) >= 2]
    metrics = {
        "n_samples": len(S), "n_docs": len(docs), "n_human": sum(s["kind"] == "human" for s in S),
        "n_real_pages": len(real),
        "cv_mae_learned": round(float(np.mean(np.abs(pred_l - y))), 4),
        "cv_spearman_learned": round(_spearman(pred_l, y), 4),
        "cv_spearman_heuristic": round(_spearman(pred_h, y), 4),
        "cv_regret_learned": round(reg_l, 4), "cv_regret_heuristic": round(reg_h, 4),
        "acc_chosen_learned": round(acc_l, 4), "acc_chosen_heuristic": round(acc_h, 4),
        "acc_always_base": round(float(np.mean(base_acc)), 4) if base_acc else None,
        "textlayer_pages": len(tl_scores),
        "textlayer_false_reject": int(sum(not quality.text_layer_check(g["text"], lm, lm_floor, gain_floor,
                                                                       unseen_ceil)[0] for g in golds)),
        "textlayer_detect": {k: round(v, 3) for k, v in detect.items()},
        "textlayer_detect_rate": round(min(detect.values()), 3) if detect else None,
        "lm_floor": None if lm_floor is None else round(lm_floor, 3),
        "gain_floor": None if gain_floor is None else round(gain_floor, 3),
        "unseen_ceil": None if unseen_ceil is None else round(unseen_ceil, 3),
        "ink_ref": round(ink_ref, 4),
        "real_pass_rate": round(float(np.mean([q >= REVIEW for q in real])), 4) if real else None,
        "real_explore_rate": round(float(np.mean([q < TARGET for q in real])), 4) if real else None,
        "weights": {k: round(float(v), 4) for k, v in zip(FEATURES, w)},
        "variant_prior": {k: round(v, 3) for k, v in prior.items()},
        "ctx_groups": len(groups),
        **ctx_m,
    }
    g = gate(metrics)
    store.model_put(version, "candidate", model.to_params(), metrics, g)
    store.commit()
    return version


def ctx_groups(store: Store, S: list[dict]) -> dict[tuple, dict]:
    """自監督樣本依 (sha, 頁, 劣化) 分組，附上掃描狀況特徵（沒有特徵的組不收）。"""
    conds = store.cond_rows()
    out: dict[tuple, dict] = {}
    for i, s in enumerate(S):
        if s["kind"] != "selfsup" or s["group"] not in conds:
            continue
        g = out.setdefault(s["group"], {"cond": conds[s["group"]], "acc": {}, "seconds": {}, "idx": {},
                                        "doc": s["doc"]})
        g["acc"][s["variant"]] = s["acc"]
        g["seconds"][s["variant"]] = s["seconds"]
        g["idx"][s["variant"]] = i
    return {k: g for k, g in out.items() if len(g["acc"]) == len(adaptive.VARIANTS)}


def ctx_cv(S: list[dict], rank: np.ndarray, pred: np.ndarray, groups: dict[tuple, dict], ink_ref: float,
           new: bool = True) -> dict:
    """留一份文件：先驗順序 vs 情境模型，重播新版型的自適應迴圈。rank/pred 是留一份文件的品質預測。"""
    old_rows, ctx_rows, base_acc = [], [], []
    for d in sorted({g["doc"] for g in groups.values()}):
        train = [g for g in groups.values() if g["doc"] != d]
        test = [g for g in groups.values() if g["doc"] == d]
        if not train or not test:
            continue
        ctx = context.ContextModel.fit(train)
        pri = _prior([S[i] for g in train for i in g["idx"].values()])
        pol = sorted(adaptive.VARIANTS, key=lambda v: (-pri.get(v, 0.0), v != adaptive.DEFAULT, v))
        for g in test:
            smp = {v: {"acc": g["acc"][v], "quality": float(pred[i]), "rank": float(rank[i]),
                       "seconds": g["seconds"][v],
                       "suspicious": S[i]["base"]["ink_uncovered"] > ink_ref + adaptive.INK_MARGIN}
                   for v, i in g["idx"].items()}
            kw = dict(new=new, target=TARGET, max_tries=adaptive.MAX_TRIES_NEW if new else adaptive.MAX_TRIES,
                      min_tries=2 if new else 1, min_gain=adaptive.MIN_GAIN)
            old_rows.append(context.replay(smp, pol, **kw))
            ctx_rows.append(context.replay(smp, context.ctx_order(pol, ctx, g["cond"])[0], **kw))
            base_acc.append(g["acc"]["base"])
    o, c = context.summarize(old_rows), context.summarize(ctx_rows)
    if not o:
        return {}
    return {"replay_n": o["n"], "replay_base_acc": round(float(np.mean(base_acc)), 4),
            "replay_prior_acc": o["acc"], "replay_ctx_acc": c["acc"],
            "replay_prior_first_acc": o["first_acc"], "replay_ctx_first_acc": c["first_acc"],
            "replay_prior_tries": o["tries"], "replay_ctx_tries": c["tries"],
            "replay_prior_seconds": o["seconds"], "replay_ctx_seconds": c["seconds"],
            "replay_oracle_acc": o["oracle"]}


def _prior(S: list[dict]) -> dict[str, float]:
    """自監督資料中，各策略「扣掉時間成本後正確率最高」的比例 -> 新版型的策略先驗。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in S:
        groups[s["group"]].append(s)
    wins: dict[str, float] = defaultdict(float)
    full = [g for g in groups.values() if len(g) == len(adaptive.VARIANTS)]
    for g in full:
        top = max(x["acc"] - adaptive.COST_PER_S * x["seconds"] for x in g)
        best = [x for x in g if x["acc"] - adaptive.COST_PER_S * x["seconds"] >= top - 1e-9]
        for x in best:
            wins[x["variant"]] += 1 / len(best)
    return {v: wins[v] / len(full) for v in adaptive.VARIANTS} if full else {}


def gate(m: dict) -> dict:
    checks = []

    def need(name, op, thr, incomplete=False):
        v = m.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            checks.append({"metric": name, "value": None, "op": op, "threshold": thr, "result": "INCOMPLETE"})
            return
        ok = v >= thr if op == ">=" else v <= thr
        checks.append({"metric": name, "value": v, "op": op, "threshold": round(thr, 4),
                       "result": "PASS" if ok else ("INCOMPLETE" if incomplete else "FAIL")})

    need("n_samples", ">=", 60, incomplete=True)
    need("n_docs", ">=", 3, incomplete=True)
    need("n_real_pages", ">=", 20, incomplete=True)
    need("cv_regret_learned", "<=", (m.get("cv_regret_heuristic") or 0) + 0.002)
    need("cv_spearman_learned", ">=", max(0.3, (m.get("cv_spearman_heuristic") or 0) - 0.02))
    need("textlayer_false_reject", "<=", 0)
    need("textlayer_detect_rate", ">=", 0.95)
    need("real_pass_rate", ">=", 0.90)
    if m.get("replay_prior_acc") is not None:
        need("replay_ctx_acc", ">=", m["replay_prior_acc"] - 0.001)
        need("replay_ctx_tries", "<=", m["replay_prior_tries"] + 0.1)
    res = {c["result"] for c in checks}
    status = "INCOMPLETE" if "INCOMPLETE" in res else ("FAIL" if "FAIL" in res else "PASS")
    return {"status": status, "checks": checks}


def promote(store: Store, version: str | None = None, *, force: bool = False) -> str:
    rows = {r["version"]: r for r in store.model_rows()}
    if version is None:
        cands = [v for v, r in rows.items() if r["status"] == "candidate"]
        if not cands:
            raise RuntimeError("沒有 candidate，請先 learn fit")
        version = max(cands)
    if version not in rows:
        raise RuntimeError(f"沒有模型 {version}")
    st = rows[version]["gate"].get("status")
    if st != "PASS" and not force:
        raise RuntimeError(f"{version} 閘門結果 {st}，不上線（確定要上線請加 --force）")
    cur = store.model_active()
    if cur:
        store.model_status(cur, "retired")
    store.model_status(version, "active")
    store.commit()
    adaptive.reset_runtime()
    return version


def rollback(store: Store) -> str | None:
    cur = store.model_active()
    if not cur:
        raise RuntimeError("目前沒有上線中的模型（已經是 heuristic）")
    store.model_status(cur, "rolled_back")
    prev = [r["version"] for r in store.model_rows() if r["status"] == "retired" and r["version"] < cur]
    if prev:
        store.model_status(max(prev), "active")
    store.commit()
    adaptive.reset_runtime()
    return max(prev) if prev else None


# ---------------------------------------------------------------- 總覽
def status(store: Store, log=print):
    rt = adaptive.Runtime(store)
    log(f"品質模型：{rt.model.version}（{rt.model.kind}） target={rt.model.target} review={rt.model.review}")
    for r in store.model_rows():
        m = r["metrics"]
        log(f"  {r['version']} {r['status']:10} 閘門={r['gate'].get('status')} "
            f"regret {m.get('cv_regret_learned')} vs heuristic {m.get('cv_regret_heuristic')}  "
            f"spearman {m.get('cv_spearman_learned')} vs {m.get('cv_spearman_heuristic')}")
    q = defaultdict(list)
    for pid, qual in store.db.execute("SELECT profile, quality FROM obs WHERE chosen = 1"):
        q[pid].append(qual)
    log(f"版型 {len(rt.profiles.items)} 個：")
    for p in rt.profiles.items:
        order = rt.policy.order(p.id, p.new)
        duels = sum(rt.policy.stats.get((p.id, v), (0, 0))[0] for v in adaptive.VARIANTS)
        qs = q.get(p.id, [])
        docs = "、".join(display_name(d)[:14] for d in p.docs[:3]) + (f" 等 {len(p.docs)} 份" if len(p.docs) > 3 else "")
        log(f"  {p.id} 頁數={p.n:3} {'新' if p.new else '  '} 平均品質={np.mean(qs) if qs else 0:.3f} "
            f"策略順序={order[:3]}（對戰 {duels}） {docs}")
    rows = store.review_rows(None)
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["status"]] += 1
    log(f"待人工確認：{dict(cnt) or 0}；標準答案頁：文字層 {len(store.gold_rows('textlayer'))}、"
        f"人工 {len(store.gold_rows('human'))}")
