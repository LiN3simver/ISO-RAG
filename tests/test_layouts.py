"""新版型處理的單元測試（不需 OCR 引擎、不碰 data/）：
章節編號正規化、頁首頁尾樣板學習、表格還原、情境模型（看掃描狀況挑策略）、BGR 通道。

  python tests/test_layouts.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iso_rag.ingest import chunker, ocr, tables  # noqa: E402
from iso_rag.ingest.template import Templates, line_key  # noqa: E402
from iso_rag.learn import adaptive, context, quality, selfsup  # noqa: E402


def test_heading_numbering_styles():
    cases = {"1.0 目的:": ("1", "目的:"), "5.0流程圖：": ("5", "流程圖："), "一、範圍": ("1", "範圍"),
             "三.定義:物料管控": ("3", "定義:物料管控"), "第一章：管理方針": ("1", "管理方針"),
             "十二、附件": ("12", "附件"), "5.1.1流程說明": ("5.1.1", "流程說明"), "1. 目的": ("1", "目的")}
    for line, want in cases.items():
        assert chunker._is_heading(line) == want, (line, chunker._is_heading(line))
    for line in ("105.05.13", "3.0mm 以上", "2022/8/30 新版發行"):
        h = chunker._is_heading(line)
        assert h is None or h[0] != "3", (line, h)
    assert chunker.norm_heading("3.0mm 以上") == "3.0mm 以上"


def test_numbered_procedure_is_not_a_table_doc():
    """「1.0 目的 / 2.0 範圍」這種編號要算 ISO 章節，否則整份會被當成表格類（引用沒有條文編號）。"""
    pages = [{"page": 1, "text": "範例科技股份有限公司\n1.0 目的:\n為規範…\n2.0 範圍:\n本廠\n3.0 權責:\n品保部"},
             {"page": 2, "text": "4.0 定義:\n無\n5.0 作業內容:\n5.1 每日點檢"}]
    assert not chunker.is_table_doc(pages)


BODY = ["申請單由部門主管簽核", "倉庫每月盤點一次", "不合格品隔離存放", "量測儀器每年校驗", "教育訓練記錄保存三年"]


def _doc(n, title, footer):
    return [f"{title} | 文件編號 | SOP-X-001\n頁數 | {i} / {n}\n{BODY[i - 1]}\n{footer}" for i in range(1, n + 1)]


def test_templates_learn_headers_footers_not_content():
    footer_a = "本程序為範例科技股份有限公司所有,未經同意不可複印"
    docs = {"a": _doc(5, "人力資源管理作業程序", footer_a), "b": _doc(4, "倉庫作業程序", footer_a),
            "c": ["1.0 目的:\n短文件的內容只有一頁\n" + footer_a]}
    # 每頁都有的條文標題不能當樣板
    docs["a"] = [t + "\n1.0 目的:" for t in docs["a"]]
    t = Templates.learn(docs, chunker.is_heading_line)
    assert t.is_template("a", "頁數 | 3 / 5")                       # 頁碼不同也是同一行（數字遮罩）
    assert t.is_template("a", "人力資源管理作業程序 | 文件編號 | SOP-X-001")
    assert not any(t.is_template("a", b) for b in BODY)               # 本文不動
    assert t.is_template("c", footer_a)                              # 跨文件共通樣板：1 頁的文件也套用
    kept = t.strip("a", ["1.0 目的:", "頁數 | 2 / 5", "真正的內容"], chunker.is_heading_line)
    assert kept == ["1.0 目的:", "真正的內容"]
    assert line_key("頁數 | 2 / 5") == line_key("頁數 3/17".replace("17", "5"))


def _grid_image(rows=3, cols=3, cw=300, ch=150, x0=150, y0=300, dpi=300):
    img = np.full((3508, 2480, 3), 255, np.uint8)
    for r in range(rows + 1):
        cv2.line(img, (x0, y0 + r * ch), (x0 + cols * cw, y0 + r * ch), (0, 0, 0), 4)
    for c in range(cols + 1):
        cv2.line(img, (x0 + c * cw, y0), (x0 + c * cw, y0 + rows * ch), (0, 0, 0), 4)
    s = dpi / 72.0
    return img, (lambda px: px / s)


def test_table_cells_join_wrapped_text_per_row():
    img, pt = _grid_image()
    tb = tables.detect(img)
    assert len(tb) == 1 and len(tb[0].cells) == 9
    W = ocr.Word
    x = lambda c: pt(150 + c * 300 + 20)          # noqa: E731
    y = lambda r, k=0: pt(300 + r * 150 + 20 + k * 50)   # noqa: E731
    words = [W(x(0), y(0), x(0) + 40, y(0) + 10, "量測點", 1.0), W(x(0), y(0, 1), x(0) + 30, y(0, 1) + 10, "位置", 1.0),
             W(x(1), y(0), x(1) + 30, y(0) + 10, "公差", 1.0),
             W(x(0), y(1), x(0) + 20, y(1) + 10, "25", 1.0), W(x(1), y(1), x(1) + 20, y(1) + 10, "350", 1.0),
             W(x(2), y(1), x(2) + 20, y(1) + 10, "18~2", 1.0), W(x(2), y(1, 1), x(2) + 10, y(1, 1) + 10, "2", 1.0),
             W(pt(150), pt(100), pt(400), pt(130), "表格上面的一般文字", 1.0)]
    text, rows = tables.assemble(words, tb, 595.0)
    assert rows == 2
    lines = text.splitlines()
    assert lines[0] == "表格上面的一般文字"
    assert lines[1] == "量測點位置 | 公差"
    assert lines[2] == "25 | 350 | 18~22"


def test_skewed_table_is_deskewed():
    """歪 2.5° 的表格：先轉正再找格線，字的座標跟著轉，列要跟沒歪時一樣。"""
    img, pt = _grid_image(rows=3, cols=3)
    W = ocr.Word
    s = 300 / 72.0
    cells = {(0, 0): "量測點", (0, 1): "公差", (0, 2): "角度", (1, 0): "25", (1, 1): "350", (1, 2): "90°",
             (2, 0): "31", (2, 1): "352", (2, 2): "70°"}
    for ang in (2.5, -2.5):
        H, Wd = img.shape[:2]
        M = cv2.getRotationMatrix2D((Wd / 2, H / 2), ang, 1.0)
        rot = cv2.warpAffine(img, M, (Wd, H), borderValue=(255, 255, 255))
        words = []
        for (r, c), text in cells.items():
            px, py = 150 + c * 300 + 60, 300 + r * 150 + 60
            x, y = M @ np.array([px, py, 1.0])
            words.append(W(x / s - 8, y / s - 4, x / s + 8, y / s + 4, text, 1.0))
        tb = tables.detect(rot)
        assert len(tb) == 1 and tb[0].rot is not None and abs(abs(tb[0].rot[0]) - 2.5) < 0.3
        text, rows = tables.assemble(words, tb, 595.0)
        assert text.splitlines() == ["量測點 | 公差 | 角度", "25 | 350 | 90°", "31 | 352 | 70°"], (ang, text)


def test_no_table_falls_back_to_lines():
    img = np.full((800, 600, 3), 255, np.uint8)
    W = ocr.Word
    words = [W(10, 10, 50, 20, "第一行", 1.0), W(10, 40, 50, 50, "第二行", 1.0)]
    assert tables.assemble(words, tables.detect(img), 595.0) == ("第一行\n第二行", 0)


def test_garbage_and_accuracy_ignore_symbols():
    assert not quality._garbage("×") and quality._garbage("é")
    assert quality.accuracy("項目 | 標準 | 方法", "項目標準方法") == 1.0


def test_bgr_red_channel():
    """pdfium 渲染是 BGR：紅章在第 3 個色版。redch 要把紅章變白、黑字留著。"""
    img = np.full((100, 100, 3), 255, np.uint8)
    img[10:20, :] = selfsup.STAMP_BGR           # 紅章
    img[50:60, :] = (0, 0, 0)                    # 黑字
    out = adaptive.preprocess(img, "redch")
    assert out[15, 50].min() > 200 and out[55, 50].max() < 50
    text, _, _ = quality.ink_layers(img)
    assert not text[15].any()                    # 紅章不算墨跡


def test_condition_features_see_stamps_and_fading():
    rng = np.random.default_rng(0)
    img = np.full((1200, 900, 3), 255, np.uint8)
    for i in range(20):
        cv2.putText(img, "TEXT LINE %d" % i, (60, 80 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    clean = context.condition_features(img)
    stamp = context.condition_features(selfsup.degrade(img, "stamp", rng))
    ghost = context.condition_features(selfsup.degrade(img, "ghost", rng))
    assert stamp["red_ratio"] > clean["red_ratio"] + 0.005
    assert ghost["contrast"] < clean["contrast"] - 0.3


def test_context_model_puts_redch_first_on_stamped_pages():
    rng = np.random.default_rng(1)
    vs = list(adaptive.VARIANTS)
    groups = []
    for i in range(40):
        stamped = i % 2 == 0
        cond = {k: float(rng.normal(0.5, 0.05)) for k in context.COND_FEATURES}
        cond["red_ratio"] = 0.05 if stamped else 0.0
        acc = {v: 0.90 for v in vs}
        if stamped:
            acc["redch"] = 0.96
        groups.append({"cond": cond, "acc": acc, "seconds": {v: 4.0 for v in vs}})
    m = context.ContextModel.fit(groups)
    base_order = ["base", "loose", "sharpen", "norm", "up", "redch"]
    s_cond = dict(groups[0]["cond"])
    c_cond = dict(groups[1]["cond"])
    assert context.ctx_order(base_order, m, s_cond)[0][0] == "redch"
    assert context.ctx_order(base_order, m, c_cond)[0][0] == "base"         # 正常頁 base 先
    m2 = context.ContextModel.from_dict(m.to_dict())
    assert m2.first(s_cond)[0] == "redch"


def test_replay_follows_stop_rules():
    smp = {v: {"acc": 0.9, "quality": 0.95, "rank": 0.95, "seconds": 4.0, "suspicious": False}
           for v in adaptive.VARIANTS}
    smp["redch"].update(acc=0.97, rank=0.97)
    order = ["base", "loose", "redch", "norm", "sharpen", "up"]
    new = context.replay(smp, order, new=True, target=0.93, max_tries=4, min_tries=2, min_gain=0.02)
    assert new["tries"] == 2 and new["acc"] == 0.9                    # 品質夠好，試兩個就停
    ctx = context.replay(smp, ["redch"] + order[:2] + order[3:], new=True, target=0.93, max_tries=4,
                         min_tries=2, min_gain=0.02)
    assert ctx["acc"] == 0.97 and ctx["first_acc"] == 0.97


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
