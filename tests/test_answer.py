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
     doc("D2", "SOP-NC-001", "SOP-NC-001 不合格品控制程序 A1", "s2"),
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
    text, cited, _ = ans._replace_cites("- 保存 3 年 [1][2]\n- 出現在 1 份 [3]", hits, IDX, fulltext=True)
    lines = text.splitlines()
    assert lines[0].count("SOP-QA-001 p.4") == 1, lines[0]          # 同文件同頁只標一次
    assert "【全文核對】" in lines[1]
    assert (0, [4]) in cited and (1, [4]) in cited
    assert (2, []) in cited                                          # 全文核對記成索引 len(hits)


# ---------------------------------------------------------------- 逐行核對
def test_missing_flags_merged_alnum_token():
    # 「版本 A」「版次 00」被併成「A0」：每個字元都核對得到，整串卻是捏造的；原文有的代碼去掉符號照樣算有
    src = "系統版本：A\n文件版本：00\n3.2 | ASQ/ANSI Z1.4-2003(R2018)\nISO 9001\nWI-1001"
    assert ans._missing("最新版次為 A0", src) == ["a0"]
    for ok in ["版次 3.2 依 Z1.4-2003(R2018)", "ISO9001", "WI1001", "系統版本 A，文件版本 00"]:
        assert ans._missing(ok, src) == [], ok


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
        q = self.judges if system == ans.JUDGE else self.answers
        # 回放用完：補漏回「無」、審查員回「全部合格」
        text = q.pop(0) if q else ("不合格：無" if system == ans.JUDGE else "無")
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
    # 前兩票不一致 -> 第三票；只有一票標出的行不刪
    fake = FakeLLM(["- 保存期均為 3 年 [1]\n- 由品保部統一報廢 [2]"],
                   ["問題要的是：保存期限\n不合格：2", "問題要的是：保存期限\n不合格：無",
                    "問題要的是：保存期限\n不合格：無"])
    out = _generate(fake)
    assert "報廢" in out.text, out.text
    assert sum(1 for judge, _ in fake.calls if judge) == 3


def test_majority_vote():
    def votes(*vs):
        it = iter(vs)
        return ans.majority(lambda: next(it))
    assert votes(set()) == []                                  # 第一票沒標：不再問
    assert votes({1}, {1}) == [1]                              # 前兩票一樣：照它
    assert votes({0, 1}, {1}, {0}) == [0, 1]                   # 不一樣：第三票決定
    assert votes({0, 1}, {1}, set()) == [1]


def test_loop_rewrites_with_asked_and_stops_at_max_verify():
    # 每一輪都整個答非所問：重寫 MAX_VERIFY-1 次，第 MAX_VERIFY 輪後收尾。
    # 每行都核對得到原文 -> 保留這版，不改說「文件未規定」（審查員看不到原文，可能否定正確答案）
    fake = FakeLLM(["- 由品保部統一報廢 [2]"] * ans.MAX_VERIFY,
                   ["問題要的是：保存期限\n不合格：1"] * (2 * ans.MAX_VERIFY))
    out = _generate(fake)
    answers = [u for judge, u in fake.calls if not judge]
    assert len(answers) == ans.MAX_VERIFY, len(answers)                # 1 次回答 + 2 次重寫
    assert "審查員認為問題要的是：保存期限" in answers[1]
    assert "報廢" in out.text and out.grounded, out.text
    assert any("保留這版" in x for x in out.verify_log), out.verify_log
    assert sum(1 for x in out.verify_log if x.startswith("第")) == ans.MAX_VERIFY


def test_judge_reads_last_verdict_line_after_per_line_reasons():
    # 審查員逐行寫理由時，「第 1 行：不合格，…」不是結論；只看最後的「不合格：」行
    fake = FakeLLM(["- 保存期均為 3 年 [1]\n- 由品保部統一報廢 [2]"],
                   ["問題要的是：保存期限\n第1行：合格\n第2行：不合格，是報廢流程\n不合格：2"] * 2)
    out = _generate(fake)
    assert "3 年" in out.text and "報廢" not in out.text, out.text


def test_unreleased_docs_marked():
    # 同名文件各有規定：摘錄標頭標出未發行；只引用未發行文件的行由程式補標
    d1 = doc("B1", "SOP-X-001", "樣品制作作業辦法")
    d2 = doc("B2", "SOP-Y-002", "樣品制作作業辦法")
    d2["status"] = "新建立"
    assert ans.doc_head(d1) == "SOP-X-001 樣品制作作業辦法"
    assert ans.doc_head(d2) == "SOP-Y-002 樣品制作作業辦法（未發行）"
    assert ans.released({**d1, "status": None})                     # 沒有狀態資料視為已發行
    idx = FakeIdx([d1, d2], [chunk(50, "B1", 3, "5.1.4 樣品需求數量 一般≤100pcs"),
                             chunk(51, "B2", 2, "5.1.4 樣品需求數量: 一般≦10pcs")])
    hits = [Hit(idx.chunks[50], 1, []), Hit(idx.chunks[51], 1, [])]
    got = ans._mark_unreleased("- SOP-X-001：一般≤100pcs [1]\n- SOP-Y-002：一般≦10pcs [2]\n"
                               "- SOP-Y-002（未發行）：≦10pcs [2]\n- 兩份都有規定 [1][2]", hits, idx)
    assert got.splitlines() == ["- SOP-X-001：一般≤100pcs [1]", "- SOP-Y-002：一般≦10pcs（未發行） [2]",
                                "- SOP-Y-002（未發行）：≦10pcs [2]", "- 兩份都有規定 [1][2]"], got
    # 好幾行都只引用同一份未發行文件（問整節）：最前面標一次
    got = ans._mark_unreleased("3.1 VL：檢驗水準 [2]\n3.2 CL：樣本代字 [2]", hits, idx)
    assert got.splitlines() == ["SOP-Y-002（未發行）：", "3.1 VL：檢驗水準 [2]", "3.2 CL：樣本代字 [2]"], got


def test_sister_doc_pulled_in():
    # 同名文件：檢索到其中一份的條文，另一份同一條（內容最像）接在後面；問題點名文件就不補
    from iso_rag.query import search as s
    d1 = doc("B1", "SOP-X-001", "SOP-X-001 樣品制作作業辦法 A5")
    d2 = doc("B2", "SOP-Y-002", "SOP-Y-002-樣品制作作業辦法")
    for d in (d1, d2):
        d["pdf_path"] = "x.pdf"
    idx = FakeIdx([d1, d2], [
        chunk(60, "B1", 6, "5.10.3.5開發階段送樣至下一階段送樣客戶反映沒問題後,前次送樣樣品即可報廢處理\n5.10.3.6至MP後,前段送樣樣品留樣保存兩年後即可報廢"),
        chunk(61, "B1", 3, "5.1.4 樣品需求數量 一般≤100pcs"),
        chunk(62, "B2", 5, "5.10.3.5 開發階段送樣至下一階段送樣客戶反映沒問題後,前次送樣樣品即可報廢處理\n5.10.3.6 至MP後,前段送樣樣品留樣保存一年後即可報廢")])
    hits = [Hit(idx.chunks[62], 1, ["dense#1"])]
    got = s._sisters(Analysis("量產後之前送樣的樣品要留多久才能報廢？", "content"), idx, hits)
    assert [h.chunk.id for h in got] == [62, 60] and got[1].why[-1] == "sister"
    assert s._sisters(Analysis("q", "content", doc_nos=["B2"]), idx, hits) is hits


def test_question_naming_a_table_column_gets_that_table():
    # 問句含指名文件裡某張表的一格（表頭「版次」）：那張表接在屬性卡後面，不管檢索排第幾
    from iso_rag.query import search as s
    idx = FakeIdx([doc("D7", "SOP-QA-009", "品質統計作業程序", "s7")], [
        chunk(70, "D7", 0, "文件編號：D7", kind="meta", sha="s7"),
        chunk(71, "D7", 1, "版次 | 內容 | 日期\n1.1 | 新增 | JUL-01-2006\n3.2 | 更新 | JULY-18-2022", sha="s7"),
        chunk(72, "D7", 2, "6.1 使用數據分析的項目", sha="s7")])
    hits = [Hit(idx.chunks[70], 1, []), Hit(idx.chunks[72], 1, [])]
    a = Analysis("SOP-QA-009 的版次有哪些？", "attribute", doc_nos=["D7"])
    assert [h.chunk.id for h in s._column_tables(a, idx, hits)] == [70, 71, 72]
    a = Analysis("SOP-QA-009 的目的？", "attribute", doc_nos=["D7"])
    assert s._column_tables(a, idx, hits) is hits


def test_echo_lines_have_no_content():
    # 只複述問題或摘錄標頭的行沒有回答任何東西；英數字整個詞比對：「3.2」不是「SOP-QA-032」的一截
    texts = ["SOP-QA-032 品質統計作業程序（未發行）｜PDF\n版次 | 內容\n3.2 | 更新",
             "SOP-QA-032 品質統計作業程序（未發行）｜SOP-QA-032 文件屬性\n系統版本：A"]
    kin = [set(), set()]
    q = "SOP-QA-009 的版次？"
    echo = ["- SOP-QA-009：未發行 [1]", "- SOP-QA-032 品質統計作業程序（未發行） [1]", "SOP-QA-032（未發行）： [1]"]
    real = ["- SOP-QA-009 的版次為 3.2（未發行） [1]", "- 3.2 [1]", "- 版次：A [2]", "- 系統版本：A [2]"]
    assert ans._echo_lines(echo + real, texts, q, kin) == [0, 1, 2]


def test_loop_rewrites_echo_answer():
    # 整個回答只複述標頭：不問審查員，直接附原因重寫
    fake = FakeLLM(["- 樣品保存作業規範 [1]", "- 保存期均為 3 年 [1]", "無"],
                   ["問題要的是：保存期限\n不合格：無"])
    out = _generate(fake)
    answers = [u for judge, u in fake.calls if not judge]
    assert "只寫了文件編號、名稱或發行狀態" in answers[1], answers[1][-200:]
    assert "3 年" in out.text and any("只複述了標頭" in x for x in out.verify_log), (out.text, out.verify_log)


def test_judge_verdict_formats():
    # 結論行的各種寫法：照抄格式說明的「行號（1）」、「第 2 行」、說明裡的數字不算、「無」
    real = ans.llm.complete
    try:
        for text, want in [("問題要的是：x\n不合格：行號（1）", {0}), ("不合格：第 2 行", {1}),
                           ("不合格：2（版次1.2 第 1 行不算）", {1}), ("不合格：無", set()), ("不合格：1, 2", {0, 1})]:
            ans.llm.complete = FakeLLM([], [text])
            got = ans.judge_once("q", ["a", "b"], timeout=10)[0]
            assert got == want, (text, got)
    finally:
        ans.llm.complete = real


def test_judge_sees_digits_masked():
    # 審查員看不到原文，會拿模型自己記得的數字否定正確答案：回答的數值遮成 #，問題不動
    fake = FakeLLM([], ["問題要的是：留樣頻率\n不合格：無"])
    real = ans.llm.complete
    ans.llm.complete = fake
    try:
        got, asked, _ = ans.judge_once("射出課連續生產時多久要留樣一次？",
                                       ["- 不少于1次/96小時 [1]"], timeout=10)
    finally:
        ans.llm.complete = real
    user = fake.calls[0][1]
    assert "96" not in user and "不少于#次/#小時" in user and got == set() and asked == "留樣頻率", user


def test_fill_missing_skips_paraphrase_of_same_source():
    # 補漏補出的一行換了說法，但依據的是同一句原文 -> 重複，不收；字面幾乎一樣、卻是原文另一列的 -> 是漏的，要收
    src = ["5.2 品保部設立留樣櫃(箱)，儲存溫度為 10℃至 35℃，並按生產日期&工單號順序進行歸檔放置。",
           "射出課連續生產時不少於1次/96小時\n組立課連續生產時不少於1次/96小時"]
    kin = [set(), set()]
    real = ans.llm.complete
    for have, reply, want in [
        ("- 品保部設立留樣櫃(箱) [1]\n- 並按生產日期及工單號順序進行歸檔放置 [1]", "- 按生產日期&工單號順序歸檔放置 [1]", 0),
        ("- 品保部設立留樣櫃(箱) [1]", "- 按生產日期&工單號順序進行歸檔放置 [1]", 1),
        ("- 射出課連續生產時不少於1次/96小時 [2]", "- 組立課連續生產時不少於1次/96小時 [2]", 1),
    ]:
        ans.llm.complete = FakeLLM([reply], [])
        try:
            _, added, _ = ans._fill_missing("u", have, src, "q", kin, timeout=10, rows=False)
        finally:
            ans.llm.complete = real
        assert added == want, (have, reply, added)


def test_judge_ignores_digits_in_its_explanation():
    # 審查員在行號後面加說明，說明裡的「版次1.2」不是第 1、2 行
    fake = FakeLLM(["- 保存期均為 3 年 [1]\n- 由品保部統一報廢 [2]", "無"],
                   ["問題要的是：保存期限\n不合格：2（版次1.2 第 1 行不算）"] * 2)
    out = _generate(fake)
    assert "3 年" in out.text and "報廢" not in out.text, out.text


def test_loop_fills_missing_line_then_verifies_again():
    # 第 1 版通過審查但漏了 5.7；補漏補上（原文核對得到）、重寫已有的行不收，下一輪再審查
    fake = FakeLLM(["- 保存期均為 3 年 [1]", "- 保存期均為3年 [1]\n- 由品保部統一報廢 [2]"],
                   ["問題要的是：保存與報廢\n不合格：無"] * 2)
    out = _generate(fake)
    assert "3 年" in out.text and "報廢" in out.text, out.text
    assert out.text.count("保存期") == 1, out.text
    assert any("補上漏掉的 1 行" in x for x in out.verify_log), out.verify_log
    assert sum(1 for judge, _ in fake.calls if judge) == 2               # 補上後又審查了一輪


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


# ---------------------------------------------------------------- 問整個章節
SEC_IDX = FakeIdx(
    [doc("D4", "WI3001", "WI3001-抽樣計劃作業指導書-A2", "s4")],
    [chunk(10, "D4", 2, "1.0 目的：\n為確保各單位能正確執行檢驗轉換之作業。\n3.0 定義：\n3.1 VL: verification level，檢驗水準。",
           sha="s4", clause="1"),
     chunk(11, "D4", 2, "3.3 嚴重缺點(CR): 對使用者有安全顧慮之缺點。\n3.6 AQL: Acceptance Quality Limit，可接受品質水準。",
           sha="s4", clause="3.3"),
     chunk(12, "D4", 2, "3.7 抽樣名詞:Ac：允收數，Re：拒收數\n3.8 抽樣計劃：抽樣計畫所表示產品單位數量從每批或批次均需被檢查\n"
           "4.0 權責：\n4.1 採購、生管：通知供方瞭解該檢驗抽樣計劃。", sha="s4", clause="3.7"),
     chunk(13, "D4", 2, "4.2 QC人員：正確的執行檢驗抽樣計劃作業。\n4.3 品管主管：監督所屬人員正確的執行檢驗抽樣計劃作業。\n"
           "5.0 流程圖\n無", sha="s4", clause="4.2")])


def test_whole_section_fills_missing_chunk():
    from iso_rag.query.search import _whole_sections
    hits = [Hit(SEC_IDX.chunks[i], 1, []) for i in (12, 10, 13)]
    a = Analysis("抽樣計畫作業指導書的定義是?", "content", focused=True)
    got = _whole_sections(a, SEC_IDX, hits)
    assert [h.chunk.id for h in got] == [10, 11, 12, 13]             # 漏掉的 3.3–3.6 補進來，依文件順序
    assert [(s.num, s.title) for s in a.sections] == [("3", "定義")]
    assert [n for n, _ in a.sections[0].items] == ["3.1", "3.3", "3.6", "3.7", "3.8"]
    # 問的是節裡的一條（AQL、允收數的定義）、或沒指向特定文件：不補
    for q in ("抽樣計畫作業指導書中AQL的定義", "WI3001 允收數的定義"):
        a = Analysis(q, "content", focused=True)
        assert [h.chunk.id for h in _whole_sections(a, SEC_IDX, hits)] == [12, 10, 13] and not a.sections, q
    a = Analysis("定義是什麼?", "content")
    assert _whole_sections(a, SEC_IDX, hits) == hits and not a.sections


def test_cite_label_marks_only_clause_used():
    c = SEC_IDX.chunks[12]
    line = "- 抽樣計劃：抽樣計畫所表示產品單位數量從每批或批次均需被檢查 [1]"
    assert ans._evidence_clauses(line, c) == ["3.8"]
    text, _, clauses = ans._replace_cites(line, [Hit(c, 1, [])], SEC_IDX)
    assert clauses == {0: "3.8"} and "【WI3001 p.2 3.8】" in text, text     # 標 3.8，不是段落的 3.7–4.1
    # 4.2、4.3 大半同字：只標解釋得了這一行的那一條
    assert ans._evidence_clauses("- QC人員：正確的執行檢驗抽樣計劃作業 [1]", SEC_IDX.chunks[13]) == ["4.2"]


def test_uncovered_lists_missing_section_items():
    from iso_rag.query.search import Section
    sec = Section("D4", "3", "定義", [10, 11, 12], [("3.1", "VL"), ("3.3", "嚴重缺點(CR)"), ("3.8", "抽樣計劃")])
    assert ans._uncovered("- VL: 檢驗水準 [1]\n- 抽樣計畫：產品單位數量 [3]", [sec]) == ["3.3 嚴重缺點(CR)"]
    assert ans._uncovered("- VL [1]\n- CR：安全顧慮 [2]\n- 抽樣計劃 [3]", [sec]) == []
    assert ans._uncovered("- 任何內容", []) == []


def test_crowded_flags_several_items_on_one_line():
    from iso_rag.query.search import Section
    sec = Section("D4", "3", "定義", [10, 11, 12], [("3.1", "VL"), ("3.3", "嚴重缺點(CR)"), ("3.6", "AQL"),
                                                     ("3.7", "抽樣名詞"), ("3.8", "抽樣計劃")])
    hits = [Hit(SEC_IDX.chunks[i], 1, []) for i in (10, 11, 12)]
    crowded = "- 嚴重缺點(CR)：對使用者有安全顧慮之缺點；AQL: Acceptance Quality Limit，可接受品質水準 [2]"
    got = ans._crowded(crowded, hits, [sec])
    assert len(got) == 1 and "3.3、3.6" in got[0], got
    ok = ("3.3 嚴重缺點(CR)：對使用者有安全顧慮之缺點 [2]\n3.6 AQL: Acceptance Quality Limit，可接受品質水準 [2]\n"
          "3.8 抽樣計劃：抽樣計畫所表示產品單位數量從每批或批次均需被檢查 [3]")
    assert ans._crowded(ok, hits, [sec]) == []


# ---------------------------------------------------------------- 問整張長表
def _tbl_idx():
    rows = "\n".join(f"{n} | Level {'I' if n % 3 else 'III'} | X | Substance {n} | {n}-00-1" for n in range(1, 61))
    parts = rows.splitlines()
    chunks = [chunk(20, "D5", 3, "4.5 REACH\n表 5 高度關注物質(SVHC)清單\n項目 等級/大類 物質名稱 CAS No.\n"
                    + "\n".join(parts[:20]), sha="s5")]
    for k, page in ((1, 4), (2, 5)):
        chunks.append(chunk(20 + k, "D5", page, "項目 | 等級/大類 | 物質名稱 | CAS No.\n" + "\n".join(parts[20 * k:20 * k + 20]),
                            sha="s5"))
    chunks.append(chunk(23, "D5", 6, "4.6 無鹵素產品之規範\n表 6 鹵素限值\n項目 物質 限值\n1 | Cl | 900", sha="s5"))
    return FakeIdx([doc("D5", "HSF-001", "HSF-001 有害物質管理技術標準", "s5")], chunks)


def test_split_rows_pulls_rest_of_cut_table():
    # 修訂履歷表被切成 3 段，只檢索到中間那段：前後同欄數的列接上；後面不是表格的段落不補
    from iso_rag.query import search as s
    idx = FakeIdx([doc("D7", "SOP-QA-009", "品質統計作業程序", "s7")], [
        chunk(30, "D7", 1, "修訂履歷\n版次 | 內容 | 日期\n1.1 | 新增 | 2006\n1.2 | 修正 | 2008", sha="s7"),
        chunk(31, "D7", 1, "1.3 | 增加 | 2009\n3.0 | 改版 | 2011", sha="s7"),
        chunk(32, "D7", 1, "3.2 | 更新 | 2022\n核準 | 審核\n一.目的", sha="s7"),
        chunk(33, "D7", 2, "二、範圍\n適用於廠內各單位", sha="s7")])
    out = s._split_rows(Analysis("q", "content"), idx, [Hit(idx.chunks[31], 1, [])])
    assert [h.chunk.id for h in out] == [30, 31, 32]
    alone = [Hit(idx.chunks[33], 1, [])]
    assert s._split_rows(Analysis("q", "content"), idx, alone) is alone


def test_tables_of_follows_repeated_header():
    from iso_rag.query.search import tables_of
    idx = _tbl_idx()
    t5, t6 = tables_of(idx, "D5", "s5")
    assert (t5.num, t5.title, t5.ids) == ("5", "高度關注物質(SVHC)清單", [20, 21, 22])    # 表頭重複到第 5 頁
    assert t5.head == "項目 等級/大類 物質名稱 CAS No."
    assert (t6.num, t6.ids) == ("6", [23])


def test_whole_table_only_when_question_names_it():
    from iso_rag.query import search as s
    idx = _tbl_idx()
    hits = [Hit(idx.chunks[21], 1, [])]
    old = s.CONTEXT_CHARS
    s.CONTEXT_CHARS = 500                     # 假表很短：調小上限，讓它算長表
    try:
        a = Analysis("有害物質管理技術標準的高度關注物質清單等級一的有什麼", "content", focused=True)
        s._whole_tables(a, idx, hits)
        assert [t.num for t in a.tables] == ["5"]
        a = Analysis("有害物質管理技術標準 表5 有哪些", "content", focused=True)
        s._whole_tables(a, idx, hits)
        assert [t.num for t in a.tables] == ["5"]
        a = Analysis("有害物質管理技術標準 Substance 30 的 CAS No.", "content", focused=True)
        s._whole_tables(a, idx, hits)
        assert a.tables == []                 # 沒提到表名：照一般檢索
    finally:
        s.CONTEXT_CHARS = old


def test_dated_table_rows_are_not_clause_headings():
    # 修訂履歷的列（有日期格）不是條號；被當成條號時，後面真的「一.目的」「二、範圍」反而因為倒退被擋掉。
    # 排成表格的真條文（沒有日期格）照樣是條號
    from iso_rag.ingest import chunker
    text = "\n".join([
        "文件制/修訂變更履歷", "版次 | 制/修訂理由 | 頁次 | 制定單位 | 生效日期 | 制/修訂人",
        "1.1 | 增加作業流程圖 | 1 | 品保部 | JUL-01-2006 | 王小明",
        "1.2 | 新版發行 | 品保部 | 2018/05/05",
        "3.2 | 更新抽样计划 | 4-7页 | 品保部 | JULY-18-2022 | 李志强",
        "一.目的:", "利用統計方式", "二、範圍", "本程序適用於廠內",
        "三、作業內容", "3.2.1.1 | 本項產品應具備自動進入低耗能狀態"])
    assert chunker.headings_in(text) == ["1", "2", "3", "3.2.1.1"], chunker.headings_in(text)
    assert chunker._is_heading("8.3 | 客户退货单 | FM-C099 | 8.88.98.10 | 会议记录单")      # 「8.88.98」不是日期


def test_title_focus_not_locked_when_top_hit_names_it():
    # 問句裡的名稱是一份表單文件的標題，但排第一的是另一份文件、原文也寫了同一個名稱 -> 不鎖定
    from iso_rag.query import search as s
    idx = FakeIdx([doc("F1", "FM301", "FM301-A-ORT 留樣登記表"), doc("F2", "SOP-QA-001", "樣品保存作業規範")],
                  [chunk(40, "F2", 6, "8.使用表單\n8.1 ORT 留樣登記表 SIP-QA101-A1"),
                   chunk(41, "F1", 0, "文件名稱：FM301-A-ORT 留樣登記表", kind="meta")])
    hits = [Hit(idx.chunks[40], 1, []), Hit(idx.chunks[41], 1, [])]
    a = Analysis("ORT 留樣登記表的表單編號？", "content", focus_docs=["F1"], focus_term="留樣登記表")
    assert s._named_elsewhere(a, idx, hits, {"F1"})
    assert not s._named_elsewhere(a, idx, hits[::-1], {"F1"})           # 焦點文件本來就排第一：照樣鎖定
    a.focus_term = "文件代碼"
    assert not s._named_elsewhere(a, idx, hits, {"F1"})                 # 明講文件代碼：照樣鎖定


def test_attribute_question_checks_metadata_against_pdf():
    # 屬性題自我核對：問句的欄位詞是 PDF 裡的欄位標籤、屬性值卻不在那段 -> 兩邊不一致，交給 LLM 看 PDF；
    # PDF 沒有這個欄位、或值對得上、或沒有 PDF -> 直接回屬性
    def d(no, iso, pdf, ver):
        x = doc(no, iso, "品質統計作業程序")
        x.update(pdf_path=pdf, pages=7, doc_version=ver)
        return x
    idx = FakeIdx([d("D9", "SOP-QA-009", "x/SOP-QA-009.pdf", "00"), d("D8", "SOP-QA-008", "x/SOP-QA-008.pdf", "3.2"),
                   d("D10", "SOP-QA-010", None, "00")],
                  [chunk(80, "D9", 1, "版次 | 內容 | 生效日期\n3.1 | 封面 | JUN-15-2012\n3.2 | 更新 | JULY-18-2022"),
                   chunk(81, "D8", 1, "版次 | 內容 | 生效日期\n3.2 | 更新 | JULY-18-2022"),
                   chunk(82, "D9", 2, "6.1 管制圖的版次控制")])
    idx.path = "fake-version"
    router._printed_cache.clear()
    a = router.analyze("SOP-QA-009 的版次？", idx)
    assert a.intent == "attribute" and not a.attr_only                  # PDF 3.1/3.2，屬性 00：不一致
    for q in ("SOP-QA-009 的文件版本？",                                  # PDF 沒有「文件版本」這個欄位
              "SOP-QA-008 的版次？",                                      # 屬性 3.2 在 PDF 修訂履歷裡：一致
              "SOP-QA-010 的版次？"):                                     # 沒有 PDF
        assert router.analyze(q, idx).attr_only, q
    router._printed_cache.clear()


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
