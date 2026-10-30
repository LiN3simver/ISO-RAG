"""OCR 自主學習的單元測試（不需 OCR 引擎、不碰 data/）。

  python tests/test_learn.py        （沒有 pytest 也能跑；有 pytest 則 pytest tests/）
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iso_rag.learn import adaptive, layout, quality, trainer  # noqa: E402
from iso_rag.learn.charlm import CharLM  # noqa: E402
from iso_rag.learn.store import Store  # noqa: E402

TEXT = ("5.7 ORT留樣的樣品保存期均為3年，每半年清點一次，超過保存期限的樣品，由品保部統一報廢處理。"
        "5.8 保存期限內如需借用樣品，應填寫借用登記表，借用人須妥善保管並於期限內歸還。"
        "1. 目的：為規範樣品保存作業，確保產品品質可追溯。2. 範圍：適用於本公司各類產品樣品。")


def _store() -> Store:
    return Store(Path(tempfile.mkdtemp()) / "t.sqlite")


def test_charlm_prefers_real_text():
    lm = CharLM()
    for _ in range(3):
        lm.add(TEXT)
    lm.finish()
    real, _ = lm.score("樣品保存期均為3年，由品保部統一報廢處理")
    chars = list("樣品保存期均為3年，由品保部統一報廢處理")
    np.random.default_rng(0).shuffle(chars)
    fake, oov = lm.score("".join(chars))
    assert real > fake + 1.0 and oov > 0.5
    # 數字併類：沒看過的料號不扣分
    assert abs(lm.score("保存期均為7年")[0] - lm.score("保存期均為2年")[0]) < 1e-9


def test_text_layer_check_catches_corruptions():
    lm = CharLM()
    lm.add(TEXT * 2)
    lm.finish()
    floor = lm.score(TEXT)[0] - 1.0
    assert quality.text_layer_check(TEXT, lm, floor)[0]
    for name, bad in trainer._corruptions(TEXT, np.random.default_rng(1)).items():
        assert not quality.text_layer_check(bad, lm, floor)[0], name


def test_accuracy_ignores_layout_and_script():
    assert quality.accuracy("樣品保存\n期均為２年", "样品保存期均为2年") == 1.0
    assert quality.accuracy("", "樣品") == 0.0
    assert 0 < quality.accuracy("樣品保存期均為", "樣品保存期均為2年") < 1


def test_ridge_recovers_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 0.5 + 0.1 * X[:, 0] - 0.05 * X[:, 2]
    mean, std, w, b = quality.fit_ridge(X, y, lam=1e-6)
    pred = b + ((X - mean) / std) @ w
    assert np.abs(pred - y).max() < 1e-4


def _write(img, y, x0, x1, h=10):
    """畫一行「字」：一個個小方塊（實心長條會被當成表格線濾掉）。"""
    for x in range(x0, x1 - 6, 10):
        img[y:y + h, x:x + 6] = 0


def test_ink_uncovered_sees_missed_text():
    small = np.full((300, 200, 3), 255, np.uint8)
    _write(small, 50, 20, 180)       # 一行字
    _write(small, 150, 20, 180)      # 另一行（沒被偵測到）
    pt = 72 / quality.ANALYSIS_DPI
    one = [(20 * pt, 50 * pt, 180 * pt, 60 * pt)]
    both = one + [(20 * pt, 150 * pt, 180 * pt, 160 * pt)]
    assert quality.ink_uncovered(small, both) < 0.05
    assert 0.4 < quality.ink_uncovered(small, one) < 0.6


def test_profiles_known_vs_new():
    s = _store()
    prof = layout.Profiles(s)
    page = np.full((1100, 850, 3), 255, np.uint8)
    for y in range(150, 1000, 40):
        _write(page, y, 100, 750, h=12)
    form = np.full((1100, 850, 3), 255, np.uint8)
    for y in range(100, 1000, 120):
        form[y:y + 3, 50:800] = 0
    for x in range(50, 800, 150):
        form[100:1000, x:x + 3] = 0
    p1, _, c1 = prof.assign(layout.fingerprint(page), "a", "test")
    p2, sim, c2 = prof.assign(layout.fingerprint(page[:, ::-1].copy()), "b", "test")
    p3, _, c3 = prof.assign(layout.fingerprint(form), "c", "test")
    assert c1 and not c2 and p2.id == p1.id and sim > layout.MATCH
    assert c3 and p3.id != p1.id
    assert len(layout.Profiles(s).items) == 2      # 存回 SQLite 再讀得到


def test_policy_learns_winner_per_profile():
    s = _store()
    pol = adaptive.Policy(s, None)
    assert pol.order("L1")[0] == "base"
    for _ in range(4):
        a = adaptive.Attempt("base", 0, [], 1, 1, 3.0, {}, "", quality=0.80, rank=0.80)
        b = adaptive.Attempt("redch", 0, [], 1, 1, 3.0, {}, "", quality=0.95, rank=0.95)
        pol.duel("L1", [a, b])
    assert pol.order("L1")[0] == "redch"
    assert pol.order("L2")[0] == "base"            # 別的版型不受影響
    assert adaptive.Policy(s, None).order("L1")[0] == "redch"
    # 學到的先驗只影響新版型；熟悉版型沒有自己的對戰紀錄時維持 base
    pol2 = adaptive.Policy(s, {"norm": 0.9, "base": 0.05})
    assert pol2.order("L3", new=True)[0] == "norm" and pol2.order("L3", new=False)[0] == "base"


def test_preprocess_keeps_shape():
    img = np.random.default_rng(0).integers(0, 255, (60, 40, 3), dtype=np.uint8)
    for v in adaptive.VARIANTS.values():
        out = adaptive.preprocess(img, v.prep)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_gate_semantics():
    good = {"n_samples": 100, "n_docs": 4, "n_real_pages": 50, "cv_regret_learned": 0.01,
            "cv_regret_heuristic": 0.02, "cv_spearman_learned": 0.7, "cv_spearman_heuristic": 0.5,
            "textlayer_false_reject": 0, "textlayer_detect_rate": 1.0, "real_pass_rate": 0.98}
    assert trainer.gate(good)["status"] == "PASS"
    assert trainer.gate({**good, "cv_regret_learned": 0.05})["status"] == "FAIL"
    assert trainer.gate({**good, "n_samples": 10})["status"] == "INCOMPLETE"
    assert trainer.gate({k: v for k, v in good.items() if k != "real_pass_rate"})["status"] == "INCOMPLETE"


def test_filled_marks_finds_checkboxes_not_text_or_photos():
    from iso_rag.ingest import ocr
    img = np.full((600, 800, 3), 255, np.uint8)
    img[100:130, 100:130] = 0                     # ■（2.5 mm @300dpi）
    img[100:130, 300:330] = 0                     # 被文字框蓋到的粗體字 -> 不算
    img[300:500, 400:700] = 120                   # 照片灰塊
    img[380:410, 520:550] = 0                     # 照片裡的暗塊 -> 不算
    s = 72 / 300
    marks = ocr.filled_marks(img, [(290 * s, 90 * s, 340 * s, 140 * s)])
    assert [(round(m.x0 / s), round(m.y0 / s)) for m in marks] == [(100, 100)]
    assert marks[0].text == ocr.MARK


def test_option_marks_restores_radio_buttons_inside_ocr_word():
    import cv2
    from iso_rag.ingest import ocr
    img = np.full((300, 700, 3), 255, np.uint8)
    for x in (100, 140, 240, 280, 370, 410):              # 「類型」「新增」「修改」六個字（墨水塊）
        img[105:135, x:x + 32] = 0
    img[112:116, 180:184] = img[124:128, 180:184] = 0     # 半形冒號
    cv2.circle(img, (215, 120), 13, (190, 200, 230), -1)  # 勾選的那顆：藍底（空隙不是白的）
    cv2.circle(img, (215, 120), 13, (60, 90, 200), 3)
    cv2.circle(img, (215, 120), 6, (30, 60, 180), -1)
    cv2.circle(img, (345, 120), 13, (120, 120, 120), 3)   # 沒勾
    cv2.rectangle(img, (500, 200), (530, 230), (0, 0, 0), 3)   # 單獨一個方框（像「口」）-> 不算
    s = 72 / 300
    words = [ocr.Word(95 * s, 100 * s, 450 * s, 140 * s, "類型:新增修改", 0.9)]
    new, n = ocr.option_marks(img, words)
    assert n == 2
    assert ocr.page_text(new) == "類型: ◉新增 ○修改"


def test_table_docs_are_not_split_by_fake_clauses():
    from iso_rag.ingest import chunker
    form = [{"page": 1, "text": "進料/制程檢驗規範\n序號 規格尺寸 規格上限 規格下限\n"
                                "1. 首件時機:產品開线前\n2. 巡檢4H/次\nA 5.00 5.05 4.95 ■ ■ ×\nB 3.20 3.25 3.15 ■ ■ ×"}]
    proc = [{"page": 1, "text": "1. 目的\n規範留樣作業，確保品質可追溯。\n2. 範圍\n適用於量產產品。\n"
                                "3. 權責\n品保部負責。\n4. 定義\n無。"}]
    assert chunker.is_table_doc(form) and not chunker.is_table_doc(proc)
    assert chunker.is_table_doc(proc, "檢驗通則")       # 文件類別優先
    cs = chunker.chunk_pages(form)
    assert cs and all(c.kind == "table" and c.clause == "" for c in cs)
    assert any("■ ■ ×" in c.text for c in cs)


def test_merge_text_adds_missed_lines_not_garbage():
    lm = CharLM()
    lm.add(TEXT * 3)
    lm.finish()
    adaptive.MERGE_LINE_FLOOR, floor = -5.0, adaptive.MERGE_LINE_FLOOR   # 小語料：沒看過的字約 -9，真實行約 -1
    main = "5.7 ORT留樣的樣品保存期均為3年\n5.8 保存期限內如需借用樣品"
    other = ("5.7 ORT留樣的樣品保存期均為3年\n每半年清點一次，超過保存期限的樣品，由品保部統一報廢處理。\n"
             "乙梁岭梁花梁获校苦龄定垒\n5.8 保存期限內如需借用樣品")
    text, added = adaptive.merge_text(main, [other], lm)
    assert added == ["每半年清點一次，超過保存期限的樣品，由品保部統一報廢處理。"]
    assert text.splitlines()[1] == added[0]                      # 插在對應位置
    assert adaptive.merge_text(main, [main], lm) == (main, [])   # 一樣就不動
    assert adaptive.merge_text(main, [other], None) == (main, [])  # 沒有語言模型不合併
    adaptive.MERGE_LINE_FLOOR = floor


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
