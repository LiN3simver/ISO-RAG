"""答案自主驗證的單元測試：不呼叫 LLM、不讀 data/（假索引、假 LLM、玩具字元模型）。

  python tests/test_answer.py        （沒有 pytest 也能跑；有 pytest 則 pytest tests/）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iso_rag.answer import answer as ans  # noqa: E402
from iso_rag.index.store import ChunkRow  # noqa: E402
from iso_rag.learn.charlm import CharLM  # noqa: E402
from iso_rag.query import router  # noqa: E402
from iso_rag.query.router import Analysis  # noqa: E402
from iso_rag.query.search import Hit, quoted_span  # noqa: E402


def doc(no, iso, title, sha=None):
    return {"doc_no": no, "iso_no": iso, "iso_key": iso.replace("-", "").lower(), "title": title, "pdf_sha": sha,
            "iso_system": "ISO 9001", "doc_type": "二階程序", "status": "已發行"}


class FakeIdx:
    """Reader 的最小替身：docs、chunks。"""

    def __init__(self, docs, chunks):
        self.docs = {d["doc_no"]: d for d in docs}
        self.chunks = {c.id: c for c in chunks}


def chunk(cid, no, page, text, kind="body", sha="s1", clause=""):
    return ChunkRow(cid, no, kind, page, page, "", clause, text, sha, None)


IDX = FakeIdx(
    [doc("D1", "WI1001", "WI1001-A0-不合格品處理作業辦法", "s1"),
     doc("D2", "QERP-01-Q", "QERP-01-Q 不合格品控制程序 A1", "s2"),
     doc("D3", "SOP-QA-001", "SOP-QA-001 樣品保存作業規範-A1", "s3")],
    [chunk(1, "D1", 1, "不合格品處理作業辦法\n1.目的"),
     chunk(2, "D2", 3, "5.2 依不合格品控制程序處理，必要時召開 MRB", sha="s2"),
     chunk(3, "D3", 4, "5.6 ORT 留樣的樣品保存期均為 3 年", sha="s3"),
     chunk(4, "D3", 4, "5.7 超過保存期限的樣品，由品保部統一報廢", sha="s3"),
     chunk(5, "D2", 5, "測試 report 後交品保部存查", sha="s2"),
     chunk(6, "D1", 0, "文件編號：D1\nISO編號：WI1001\n附加檔案：ORT 表單.pdf", kind="meta")])


# ---------------------------------------------------------------- 引用標籤
def test_page_label():
    assert ans.page_label([4]) == "p.4"
    assert ans.page_label([3, 1, 2]) == "p.1-3"
    assert ans.page_label([1, 3]) == "p.1、3"


def test_replace_cites_dedupes_same_page_and_marks_fulltext():
    hits = [Hit(IDX.chunks[3], 1, []), Hit(IDX.chunks[4], 1, [])]
    text, cited = ans._replace_cites("- 保存 3 年 [1][2]\n- 出現在 1 份 [3]", hits, IDX, fulltext=True)
    lines = text.splitlines()
    assert lines[0].count("SOP-QA-001 p.4") == 1, lines[0]          # 同文件同頁只標一次
    assert "【全文核對】" in lines[1]
    assert (2, []) in cited                                          # 全文核對記成索引 len(hits)


# ---------------------------------------------------------------- 逐行核對
def test_verify_accepts_same_page_sibling_excerpt():
    texts = ["5.6 ORT 留樣保存期 3 年", "5.7 超過保存期限的樣品由品保部統一報廢", "別份文件 保存期限"]
    line = "- 超過保存期限由品保部報廢 [1]"
    assert ans._verify(line, texts, "", [{1}, {0}, set()]) == {}     # 內容在同頁的另一段
    assert ans._verify(line, texts, "", None)                        # 不看同頁段落就會被判核對不到


def _toy_lm():
    lm = CharLM()
    for s in ["總不合格數均少於限值", "在穩定的生產條件下", "核准 李小華 審核 王小明", "審核人員為李小華",
              "樣品保存期限為三年", "由品保部統一報廢處理"] * 20:
        lm.add(s)
    return lm.finish()


def test_ocr_fix_exemption_uses_char_lm():
    ans._lm_cache["lm"] = _toy_lm()
    try:
        src = "之前10批允收，且總不合格數均少淤跟值(選擇性的)"
        assert ans._missing("總不合格數均少於限值", src, src + " 另一段 總不合格數均少於限值") == []
        # 人名換人：原文那段本身是正常文字，不放行
        assert ans._missing("審核人員為陳大文", "審核人員為李小華", "審核人員為李小華 另一段 陳大文") == ["陳大文"]
    finally:
        ans._lm_cache.pop("lm", None)


def test_ocr_fix_needs_context():
    ans._lm_cache["lm"] = _toy_lm()
    try:
        assert not ans._ocr_fixed("", "於限值", "", "少淤跟值")        # 沒有前後文可對齊
    finally:
        ans._lm_cache.pop("lm", None)


# ---------------------------------------------------------------- 列表題自我檢查
def test_list_explains():
    for q in ["二階程序有哪些？", "ISO 14001 的三階管理辦法有哪些？", "已發行的二階程序有哪些？", "四階表單列表"]:
        assert router._list_explains(q), q
    for q in ["哪些文件有提到不合格品處理作業辦法", "「不合格品控制程序」出現在哪些文件"]:
        assert not router._list_explains(q), q


# ---------------------------------------------------------------- 全文核對
def A(q, doc_nos=()):
    return Analysis(query=q, intent="content", doc_nos=list(doc_nos))


def test_fulltext_word_boundary_and_meta_fields():
    text, entries, n = ans._fulltext("ort", A("哪些文件有提到ORT"), IDX, label="ORT")
    assert "「ORT」出現在 2 份文件" in text, text                      # report 不算
    assert "屬性「附加檔案」" in text                             # WI1001 只在屬性的附加檔案欄
    assert {h.chunk.doc_no for h, _ in entries} == {"D1", "D3"}


def test_quoted_span_and_pick_prefers_informative_term():
    hits = [Hit(IDX.chunks[1], 1, [])]
    assert quoted_span("那些文件有提到不合格品處理作業辦法", IDX, hits, 6) == "不合格品处理作业办法"
    # 「SOP-QA-001 的 ORT…」：SOP-QA-001 只出現在點名的文件，選 ORT
    ft = ans._pick_fulltext("SOP-QA-001 的 ORT 留樣保存多久", A("", ["D3"]), IDX, [Hit(IDX.chunks[3], 1, [])])
    assert ft and "「ORT」" in ft[0], ft and ft[0]


# ---------------------------------------------------------------- 驗證迴圈（假 LLM）
class FakeLLM:
    """依序回放回答；審查員（system 是 JUDGE）另有一組回放。記下每次的 user 內容。"""

    def __init__(self, answers, judges):
        self.answers, self.judges, self.calls = list(answers), list(judges), []

    def __call__(self, system, user, **kw):
        self.calls.append((system == ans.JUDGE, user))
        text = (self.judges if system == ans.JUDGE else self.answers).pop(0)
        return type("R", (), {"text": text, "seconds": 0.0})()


def _generate(fake):
    real = ans.llm.complete
    ans.llm.complete = fake
    try:
        res = ans.Answer("", "pending", [Hit(IDX.chunks[3], 1, []), Hit(IDX.chunks[4], 1, [])])
        return ans.generate("ORT 留樣品要保存多久？", IDX, res)
    finally:
        ans.llm.complete = real


def test_loop_drops_line_both_judges_flag():
    fake = FakeLLM(["- 保存期均為 3 年 [1]\n- 由品保部統一報廢 [2]"],
                   ["問題要的是：保存期限\n不合格：2", "問題要的是：保存期限\n不合格：2"])
    out = _generate(fake)
    assert "報廢" not in out.text and "3 年" in out.text, out.text
    assert len(out.cited) == 1


def test_loop_ignores_single_judge_vote():
    fake = FakeLLM(["- 保存期均為 3 年 [1]\n- 由品保部統一報廢 [2]"],
                   ["問題要的是：保存期限\n不合格：2", "問題要的是：保存期限\n不合格：無"])
    out = _generate(fake)
    assert "報廢" in out.text, out.text


def test_loop_rewrites_with_asked_and_stops_at_max_verify():
    # 每一輪都整個答非所問：重寫 MAX_VERIFY-1 次，第 MAX_VERIFY 輪後收尾
    fake = FakeLLM(["- 由品保部統一報廢 [2]"] * ans.MAX_VERIFY,
                   ["問題要的是：保存期限\n不合格：1"] * (2 * ans.MAX_VERIFY))
    out = _generate(fake)
    answers = [u for judge, u in fake.calls if not judge]
    assert len(answers) == ans.MAX_VERIFY, len(answers)                # 1 次回答 + 2 次重寫
    assert "審查員認為問題要的是：保存期限" in answers[1]
    assert out.text.startswith("文件未規定"), out.text
    assert sum(1 for x in out.verify_log if x.startswith("第")) == ans.MAX_VERIFY


def test_loop_keeps_answer_when_rewrite_fails():
    class Down(FakeLLM):
        """第 1 次回答正常，之後的回答呼叫（重寫）都連不上。"""

        def __call__(self, system, user, **kw):
            if system != ans.JUDGE and any(not judge for judge, _ in self.calls):
                raise ConnectionError("gateway 關機")
            return super().__call__(system, user, **kw)
    # 第 1 版有一個數字在原文找不到 -> 要重寫，但重寫時 gateway 斷線
    fake = Down(["- 保存期均為 4 年 [1]"], [])
    out = _generate(fake)
    assert "4 年" in out.text and "核對不到原文：4" in out.text, out.text
    assert any("重寫失敗" in x for x in out.verify_log), out.verify_log


def test_printed_codes_need_own_text_and_fall_back_to_header():
    """沒有 ISO 編號的文件用紙本編號問：檔名＋內文都有的優先；檔名寫錯的用頁首重複印的代碼。"""
    def bare(no, title, path, pages):
        d = doc(no, "", title)
        d.update(iso_key="", pdf_path=path, pages=pages)
        return d
    idx = FakeIdx(
        [bare("D15", "客訴處理程序", "x/SOP-QC-0001_客訴處理程序V1.0.pdf", 4),
         bare("D16", "不合格品管制作業程序", "x/SOP-QC-0001不合格品.pdf", 4),
         bare("D01", "品質手冊", None, 60)],
        [chunk(1, "D15", 1, "文件編碼SOP-QC-0001\n客訴處理程序"),
         chunk(2, "D15", 2, "SOP-QC-0001 分頁次2/4"),
         chunk(3, "D16", 1, "文件編碼 SOP-QC-0002"),
         chunk(4, "D16", 2, "SOP-QC-0002 分頁次2/4"),
         chunk(5, "D01", 54, "客訴處理程序 SOP-QC-0001 品保部")])
    idx.path = "fake-printed"
    router._printed_cache.clear()
    assert router.analyze("編號SOP-QC-0001客訴處理程序的權責單位?", idx).doc_nos == ["D15"]
    assert router.analyze("SOP-QC-0002的適用範圍", idx).doc_nos == ["D16"]
    router._printed_cache.clear()


def test_verify_per_line():
    assert ans._verify("- 保存期為 5 年 [1]", ["保存期均為3年"]) == {0: ["5"]}
    assert ans._verify("- 陳大文說保存期是3年 [1]", ["保存期均為3年", "管理者代表 陳大文"]) == {0: ["陳大文"]}   # 別段搬來
    assert ans._verify("- 作業規範裡寫的保存期是3年 [2]", ["無關", "保存期均為3年", "樣品保存作業規範"],
                       "樣品保存作業規範的保存期？") == {}             # 複述問題的詞不算
    v = ans._verify("- 審核 陳大文 [1]", ["審核 林小美\n核准 陳大文"])
    assert "審核 陳大文" in v.get(0, []), v                         # 職稱配錯人名


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
