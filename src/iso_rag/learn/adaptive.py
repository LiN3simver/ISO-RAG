"""自適應 OCR：依版型學到的策略順序逐一嘗試，品質估計夠好就停（有上限的策略迴圈）。

策略（VARIANTS），在文字層頁上模擬掃描劣化後挑出：
  base     預設設定（300 dpi，RapidOCR 內部縮到最長邊 2000 px）
  loose    偵測門檻放寬：淡字、印章壓字時少漏行
  sharpen  反銳化遮罩：模糊掃描
  norm     對比拉伸（2%~98%）：褪色、灰底
  up       反銳化 + 最長邊 2600 + 放寬偵測：印章、褪色字最有效，但較慢
  redch    只用紅色通道：紅色受控章在紅色通道接近白紙，等於把章拿掉

策略順序（Policy）：每個版型記錄「勝場」——同一頁試了 ≥2 種策略時，品質估計最高者（扣掉時間成本）
得一勝，下一頁先試勝率最高的。新版型用自監督資料算出的全域先驗；熟悉版型用偏向 base 的預設。
第一個策略已達 target 就停，所以熟悉版型的乾淨頁仍是一次 OCR。

「夠好」= 品質 ≥ target 且不可疑。可疑 = 沒被框到的墨跡比這個版型平常的 p90 多 INK_MARGIN 以上：
偵測漏字時 OCR 信心照樣很高，而真實掃描頁本來就有照片、簽名、logo 這些框不到的墨跡，
所以跟同版型平常的水準比，而不是用固定門檻。
停止條件：夠好 / 次數上限（新版型 4、熟悉版型 3）/ 時間上限 / 熟悉版型改善 < MIN_GAIN。
"""
from __future__ import annotations

import difflib
import hashlib
import re
import unicodedata
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from ..ingest import ocr
from . import layout, quality
from .charlm import CharLM, is_cjk
from .quality import QualityModel
from .store import Store


@dataclass(frozen=True)
class Variant:
    prep: str | None
    max_side: int
    box_thresh: float
    unclip: float


VARIANTS: dict[str, Variant] = {
    "base": Variant(None, 2000, 0.5, 1.6),
    "loose": Variant(None, 2000, 0.35, 1.9),
    "sharpen": Variant("sharpen", 2000, 0.5, 1.6),
    "norm": Variant("norm", 2000, 0.5, 1.6),
    "up": Variant("sharpen", 2600, 0.4, 1.8),
    "redch": Variant("redch", 2000, 0.4, 1.8),
}
DEFAULT = "base"
_ENGINE_DEFAULT = Variant(None, 2000, 0.5, 1.6)   # RapidOCR config.yaml 的預設，用完要還原

COST_PER_S = 0.002          # 多花 10 秒 ≈ 品質 0.02
PRIOR_N = 2.0               # 先驗勝率相當於幾場
DEFAULT_PRIOR = {"base": 0.4, "loose": 0.2, "sharpen": 0.1, "norm": 0.1, "up": 0.1, "redch": 0.1}
MAX_TRIES, MAX_TRIES_NEW = 3, 4
MIN_GAIN = 0.02
BUDGET_S = 120.0
INK_MARGIN = 0.05
INK_MIN_OBS = 5
# 多策略合併：不同策略常各漏不同的整行。以選中的文字為主，其他策略讀到、而它對應位置整行沒有的行補進來。
MERGE_COVER = 0.5           # 該行 bigram 在對應差異區塊的覆蓋率低於此 = 選中的策略沒讀到
MERGE_MIN_CHARS = 8
MERGE_LINE_FLOOR = -12.0    # 單行語言模型分數：真實行 -3.5~-9.7，亂碼（乙梁岭梁花…）-15 以下
MERGE_MIN_CJK = 4           # 語言模型把英數併成 a/0，「Pperan o o o ow」這類英數亂碼分數不低，要另外擋


def preprocess(img: np.ndarray, prep: str | None) -> np.ndarray:
    if prep is None:
        return img
    if prep == "redch":
        # pdfium 渲染出來是 BGR，紅色在第 3 個色版
        g = img[..., 2]
    else:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if prep == "sharpen":
            g = cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0, 0), 2.0), -0.8, 0)
        elif prep == "norm":
            lo, hi = np.percentile(g, (2, 98))
            g = np.clip((g.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"未知的前處理 {prep}")
    return np.ascontiguousarray(cv2.cvtColor(np.ascontiguousarray(g), cv2.COLOR_GRAY2RGB))


def rotate(img: np.ndarray, rotation: int) -> np.ndarray:
    k = {0: 0, 90: 3, 180: 2, 270: 1}[rotation]      # 順時針 rotation 度（同 ocr._ocr）
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


@dataclass
class Attempt:
    variant: str
    rotation: int
    words: list[ocr.Word]
    width: float
    height: float
    seconds: float
    base: dict                       # quality.base_features（不含 LM）
    text: str
    quality: float = 0.0
    rank: float = 0.0                # 同頁比較用（quality + 只在同頁有意義的特徵）
    feats: dict = field(default_factory=dict)

    @property
    def page(self) -> ocr.PageOCR:
        return ocr.PageOCR(self.words, self.rotation, self.width, self.height)

    @property
    def text_sha(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def run_attempt(img: np.ndarray, small: np.ndarray, rotation: int, name: str, *, dpi: int = ocr.DPI) -> Attempt:
    """在（已轉正的）img 上跑一個策略。small 是同一張圖的 ANALYSIS_DPI 小圖（已轉正），算墨跡覆蓋用。"""
    v = VARIANTS[name]
    eng = ocr.engine()
    im = preprocess(img, v.prep)
    t0 = time.monotonic()
    eng.max_side_len = v.max_side
    try:
        res = eng(im, box_thresh=v.box_thresh, unclip_ratio=v.unclip)
    finally:   # box_thresh/unclip 會留在引擎上（RapidOCR update_params），還原給 ocr.ocr_page 用
        eng.max_side_len = _ENGINE_DEFAULT.max_side
        eng.update_params(box_thresh=_ENGINE_DEFAULT.box_thresh, unclip_ratio=_ENGINE_DEFAULT.unclip)
    secs = time.monotonic() - t0
    scale = dpi / 72.0
    words, boxes, dropped = [], [], 0
    if res.boxes is not None:
        for box, text, score in zip(res.boxes, res.txts, res.scores):
            text = unicodedata.normalize("NFKC", text or "").strip()
            if not text:
                continue
            xs, ys = box[:, 0] / scale, box[:, 1] / scale
            bb = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            boxes.append(bb)
            if score < ocr.MIN_SCORE:
                dropped += len(text)
                continue
            words.append(ocr.Word(*bb, text, float(score)))
    text = ocr.page_text(words)
    base = quality.base_features(words, text, dropped_chars=dropped, ink_unc=quality.ink_uncovered(small, boxes))
    return Attempt(name, rotation, words, float(img.shape[1] / scale), float(img.shape[0] / scale), secs, base, text)


def score(a: Attempt, model: QualityModel) -> Attempt:
    a.feats = model.features(a.base, a.text)
    a.quality = model.predict(a.feats)
    a.rank = model.rank(a.feats)
    return a


def _key(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s)


def _cover(line: str, block: str) -> float:
    a, b = _key(line), _key(block)
    A = Counter(a[i:i + 2] for i in range(len(a) - 1))
    B = Counter(b[i:i + 2] for i in range(len(b) - 1))
    return sum((A & B).values()) / max(1, sum(A.values()))


def merge_text(main: str, others: list[str], lm: CharLM | None) -> tuple[str, list[str]]:
    """main 為主，others 裡 main 整行沒讀到的行插回對應位置。回傳 (合併後文字, 補進的行)。
    只看對應的差異區塊（整頁比會被同頁的常見詞蓋過，漏掉的行補不回來）；
    沒有語言模型就不合併（擋不住亂碼）。"""
    if lm is None or lm.empty:
        return main, []
    added: list[str] = []
    for other in others:
        ml = [x for x in main.splitlines() if x.strip()]
        ol = [x for x in other.splitlines() if x.strip()]
        out: list[str] = []
        sm = difflib.SequenceMatcher(None, [x.strip() for x in ol], [x.strip() for x in ml], autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            out += ml[j1:j2]
            if op not in ("replace", "delete"):
                continue
            block = "\n".join(ml[j1:j2])
            for x in ol[i1:i2]:
                k = _key(x)
                if len(k) >= MERGE_MIN_CHARS and sum(map(is_cjk, k)) >= MERGE_MIN_CJK \
                        and _cover(x, block) < MERGE_COVER and lm.score(k)[0] >= MERGE_LINE_FLOOR:
                    out.append(x)
                    added.append(x)
        main = "\n".join(out)
    return main, added


class Policy:
    def __init__(self, store: Store, prior: dict[str, float] | None):
        self.store = store
        self.stats = store.policy_rows()          # (版型, 策略) -> (對戰次數, 勝場)
        self.prior = {**DEFAULT_PRIOR, **(prior or {})}

    def win_rate(self, pid: str, v: str, new: bool = True) -> float:
        n, wins = self.stats.get((pid, v), (0, 0.0))
        prior = self.prior if new else DEFAULT_PRIOR
        return (wins + PRIOR_N * prior.get(v, 0.1)) / (n + PRIOR_N)

    def order(self, pid: str, new: bool = True) -> list[str]:
        return sorted(VARIANTS, key=lambda v: (-self.win_rate(pid, v, new), v != DEFAULT, v))

    def duel(self, pid: str, attempts: list[Attempt]):
        """同一頁試了多種策略：每個策略對戰 +1，扣掉時間成本後品質最高者勝場 +1。"""
        if len(attempts) < 2:
            return
        win = max(attempts, key=lambda a: a.rank - COST_PER_S * a.seconds)
        for a in attempts:
            r = 1.0 if a is win else 0.0
            self.store.policy_add(pid, a.variant, r)
            n, w = self.stats.get((pid, a.variant), (0, 0.0))
            self.stats[(pid, a.variant)] = (n + 1, w + r)


class Runtime:
    """一個行程共用一份：品質模型（active 版本）、版型、策略統計。"""

    def __init__(self, store: Store | None = None, model: QualityModel | None = None):
        self.store = store or Store()
        self.model = model or QualityModel.load_active(self.store)
        self.profiles = layout.Profiles(self.store)
        self.policy = Policy(self.store, self.model.prior)
        self._ink: dict[str, list[float]] = {}

    def ink_ref(self, pid: str) -> float:
        """這個版型被採用的 OCR 結果中，ink_uncovered 的 p90（紀錄不足用模型的全域值）。"""
        if pid not in self._ink:
            self._ink[pid] = [o["feats"]["ink_uncovered"] for o in self.store.obs_rows(
                "profile = ? AND chosen = 1 AND origin != 'eval'", (pid,))]
        xs = self._ink[pid]
        return float(np.percentile(xs, 90)) if len(xs) >= INK_MIN_OBS else self.model.ink_ref

    def suspicious(self, pid: str, a: Attempt) -> bool:
        return a.base["ink_uncovered"] > self.ink_ref(pid) + INK_MARGIN

    def remember(self, pid: str, a: Attempt):
        self._ink.setdefault(pid, []).append(a.base["ink_uncovered"])


_rt: Runtime | None = None


def runtime() -> Runtime:
    global _rt
    if _rt is None:
        _rt = Runtime()
    return _rt


def reset_runtime():
    global _rt
    _rt = None


def ocr_image(img: np.ndarray, rt: Runtime, *, sha: str, page: int, doc: str, pdf: str = "",
              dpi: int = ocr.DPI, origin: str = "build", log=None) -> dict:
    """自適應 OCR 一頁（page 從 1 起算）。回傳與 extract 快取相同格式的紀錄，另加 variant/quality/profile/tried。"""
    from .context import condition_features, ctx_order
    t0 = time.monotonic()
    small = quality.analysis_image(img, dpi)
    prof, sim, created = rt.profiles.assign(layout.fingerprint(small), doc, origin)
    rt.store.commit()
    is_new = created or prof.new
    # 掃描狀況（紅章、褪色、模糊…）-> 情境模型預估某策略明顯比 base 好就先試它；其餘照版型勝場
    cond = condition_features(img, dpi) if rt.model.ctx is not None else None
    order, why = ctx_order(rt.policy.order(prof.id, is_new), rt.model.ctx, cond)

    # 1) 第一個策略順便判斷方向（同 ocr.ocr_page：直放的橫印頁）
    first = score(run_attempt(img, small, 0, order[0], dpi=dpi), rt.model)
    if ocr._looks_rotated(first.page):
        for rot in (90, 270, 180):
            cand = score(run_attempt(rotate(img, rot), rotate(small, rot), rot, order[0], dpi=dpi), rt.model)
            if cand.page.quality > first.page.quality * 1.3:
                first = cand
    rot = first.rotation
    img_r, small_r = rotate(img, rot), rotate(small, rot)

    # 2) 有上限的策略迴圈
    attempts = [first]
    max_tries = MAX_TRIES_NEW if is_new else MAX_TRIES
    min_tries = 2 if is_new else 1
    for v in order[1:]:
        top = max(attempts, key=lambda a: a.rank)
        if len(attempts) >= max_tries or time.monotonic() - t0 > BUDGET_S:
            break
        if len(attempts) >= min_tries and top.quality >= rt.model.target and not rt.suspicious(prof.id, top):
            break
        if len(attempts) >= 2 and not is_new and \
                attempts[-1].rank - max(a.rank for a in attempts[:-1]) < MIN_GAIN:
            break
        attempts.append(score(run_attempt(img_r, small_r, rot, v, dpi=dpi), rt.model))
    best = max(attempts, key=lambda a: (a.rank, -a.seconds))
    suspect = rt.suspicious(prof.id, best)

    # 3) 學習：策略勝場、觀測紀錄、待人工確認
    rt.policy.duel(prof.id, attempts)
    for a in attempts:
        rt.store.obs_put(sha=sha, page=page, variant=a.variant, origin=origin, profile=prof.id, quality=a.quality,
                         seconds=round(a.seconds, 2), chars=len(a.text), text_sha=a.text_sha,
                         feats={**a.feats, "fp_sim": round(sim, 4), **({"cond": cond} if cond else {})},
                         text=a.text, chosen=a is best)
    if origin != "eval":
        rt.remember(prof.id, best)
    reasons = (["new_layout"] if created else []) + (["low_quality"] if best.quality < rt.model.review else []) \
        + (["missed_text"] if suspect else [])
    for r in reasons:
        rt.store.review_put(sha, page, doc, pdf, r, best.quality)
    rt.store.commit()
    if log:
        log(f"    p{page:03d} 版型 {prof.id}{'(新)' if created else ''} sim={sim:.2f} {why} "
            + " ".join(f"{a.variant}={a.quality:.3f}/{a.seconds:.0f}s" for a in attempts) + f" -> {best.variant}")
    # 表格勾選記號（品質估計之後才加，模型特徵維持只看 OCR 本身的輸出）
    marks = ocr.filled_marks(img_r, [(w.x0, w.y0, w.x1, w.y1) for w in best.words], dpi)
    words, opts = ocr.option_marks(img_r, best.words + marks, dpi)      # 電子表單的 ◉○■□
    chosen = ocr.page_text(words) if marks or opts else best.text
    # words 維持選中策略的框（座標要一致）；只有 text 合併其他策略漏掉的行
    text, added = merge_text(chosen, [a.text for a in attempts if a is not best], rt.model.lm)
    extra = {"merged": len(added), "text_chosen": chosen} if added else {"merged": 0}
    tt = table_text(img_r, words, best.width, added, dpi)
    if tt["tables"]:                     # 表格版當正式文字，逐行版留在 text_lines
        extra["text_lines"], text = text, tt["text"]
    extra.update(tables=tt["tables"], tables_v=TABLES_V)
    return {"page": page, "source": "ocr", "rotation": rot, "width": best.width, "height": best.height,
            "text": text, "words": [asdict(w) for w in words], **extra,
            "marks": len(marks), "options": opts, "options_v": ocr.OPTIONS_V, "variant": best.variant,
            "quality": round(best.quality, 4), "profile": prof.id, "new_layout": created, "suspect": suspect,
            "tried": [{"variant": a.variant, "quality": round(a.quality, 4), "seconds": round(a.seconds, 2)}
                      for a in attempts], **({"ctx": why} if why else {}),
            "model": rt.model.version}


TABLES_V = 2              # 表格還原的版本（2：表格偵測前先轉正）


def table_text(img: np.ndarray, words: list, width: float, added: list[str], dpi: int = ocr.DPI) -> dict:
    """表格還原（ingest.tables）-> {"tables": 還原出的表格列數, "text": 表格版頁面文字（沒有表格時是空字串）}。
    多策略合併補進來的行（added，沒有座標）接在最後。"""
    from ..ingest import tables
    tb = tables.detect(img, dpi)
    text, rows = tables.assemble(words, tb, width)
    if not rows:
        return {"tables": 0, "text": ""}
    if added:
        text = text.rstrip() + "\n" + "\n".join(added)
    return {"tables": rows, "text": text}


def ocr_pdf_page(pdf_path: str, page_no: int, *, sha: str, doc: str, rt: Runtime | None = None, log=None) -> dict:
    """page_no 從 0 起算（同 ocr.render）。"""
    rt = rt or runtime()
    img = ocr.render(pdf_path, page_no)
    return ocr_image(img, rt, sha=sha, page=page_no + 1, doc=doc, pdf=pdf_path, log=log)


def merge_cache(store: Store | None = None, log=print) -> dict[str, int]:
    """既有快取補做多策略合併（不重跑 OCR：其他策略的文字在 learn.sqlite obs 裡）。
    原文字留在 text_chosen；已有 merged 欄位的頁略過，可重複執行。"""
    import json

    from ..config import CACHE_DIR
    rt = runtime() if store is None else Runtime(store)
    stats = {"pages": 0, "merged_pages": 0, "lines": 0}
    for cp in sorted((CACHE_DIR / "pages").glob("*/*/p*.json")):
        rec = json.loads(cp.read_text(encoding="utf-8"))
        if rec.get("source") != "ocr" or "merged" in rec or len(rec.get("tried") or []) < 2:
            continue
        stats["pages"] += 1
        sha = cp.parent.name
        others = [r["text"] for r in rt.store.obs_rows("sha=? AND page=? AND origin='build' AND variant!=?",
                                                       (sha, rec["page"], rec["variant"]))]
        text, added = merge_text(rec["text"], others, rt.model.lm)
        rec["merged"] = len(added)
        if added:
            rec["text_chosen"], rec["text"] = rec["text"], text
            stats["merged_pages"] += 1
            stats["lines"] += len(added)
            log(f"  {sha[:8]} p{rec['page']:03d} 補 {len(added)} 行：" + " / ".join(x[:30] for x in added))
        cp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return stats
