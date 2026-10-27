"""表格結構還原：從頁面影像找出格線圍成的儲存格，把 OCR 字依儲存格組回去。

為什麼要做：OCR 文字是「同一個高度的字串成一行」，多行儲存格（「量测点／位置／Point／(mm)」）會被切成好幾行，
而且跟隔壁欄位的碎片串在一起（「量测点 公 量测 量测 允收值…」），檢索與 LLM 都對不回「欄位 → 數值」。

做法（不針對任何版型寫規則，格線從影像量出來）：
  1. 150 dpi 灰階 → 墨跡 → 水平/垂直長線（形態學開運算），補小缺口
  2. 格線連通區塊夠大、水平與垂直線都有 ≥2 條 → 表格區
  3. 表格區內「不是格線」的連通區塊 = 儲存格（合併儲存格自然是一塊）；碰到表格外框外的區域不算
  4. OCR 字依中心點歸入儲存格；同一列（儲存格上緣對齊）的儲存格以「 | 」串成一行，
     儲存格內換行接起來（中文直接接、英數補空白）
  5. 一列只有一格、或很寬又很多行的格子（整頁外框、本文框）不是資料表：照原本逐行輸出
表格外的字照原本的逐行規則（ocr.to_lines）；全部依高度排序成頁面文字。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ocr

WORK_DPI = 150
LINE_MM = 6.0             # 格線最短長度（字的筆畫通常 < 4 mm）
MIN_TABLE_MM = (25.0, 10.0)
MIN_CELL_MM = (2.5, 2.0)
FRAME_W = 0.45            # 格子寬度 ≥ 頁寬 45% 且 ≥ FRAME_LINES 行 -> 外框/本文框，不當資料格
FRAME_LINES = 4
MAX_SKEW = 5.0            # 只把 ±5° 內的長線當成歪斜的水平線
MIN_DESKEW = 0.3          # 小於 0.3° 不轉正


@dataclass
class Cell:
    x0: float
    y0: float
    x1: float
    y1: float
    words: list = field(default_factory=list)


@dataclass
class Table:
    x0: float
    y0: float
    x1: float
    y1: float
    cells: list[Cell] = field(default_factory=list)
    labels: np.ndarray | None = None      # WORK_DPI 下表格範圍內每個像素屬於哪一格（-1：格線/表外）
    scale: float = WORK_DPI / 72.0
    rot: tuple[float, float, float] | None = None   # 轉正：(角度, 中心 x, 中心 y)（pt）；座標都在轉正後的座標系

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        """頁面座標 (pt) -> 這張表的座標系（偵測時把頁面轉正了多少，字的座標就跟著轉多少）。"""
        if not self.rot:
            return x, y
        a, cx, cy = self.rot
        r = np.deg2rad(a)
        dx, dy = x - cx, y - cy
        # 與 cv2.getRotationMatrix2D(center, a, 1) 相同（影像座標 y 向下）
        return cx + np.cos(r) * dx + np.sin(r) * dy, cy - np.sin(r) * dx + np.cos(r) * dy

    def cell_at(self, x: float, y: float) -> Cell | None:
        """(x, y) pt 落在哪一格。用像素歸屬而不是外接矩形：不規則的空白（外框與資料表之間的走道）
        外接矩形會蓋住整張表，把所有字都搶走。落在格線上就看周圍 2 px。"""
        if self.labels is None:
            return None
        px, py = int(x * self.scale - self.x0 * self.scale), int(y * self.scale - self.y0 * self.scale)
        H, W = self.labels.shape
        for r in (0, 1, 2):
            ys, xs = slice(max(py - r, 0), min(py + r + 1, H)), slice(max(px - r, 0), min(px + r + 1, W))
            ids = self.labels[ys, xs]
            ids = ids[ids >= 0]
            if ids.size:
                return self.cells[int(np.bincount(ids).argmax())]
        return None


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img    # pdfium 渲染是 BGR


def skew_angle(ink: np.ndarray) -> float:
    """長直線（格線、外框）的傾斜角（度，影像座標）。量不到就 0。
    霍夫直線只收接近水平、長度 ≥ 頁寬 1/4 的線，取中位數。"""
    H, W = ink.shape
    lines = cv2.HoughLinesP(ink * 255, 1, np.pi / 1800, threshold=max(W // 6, 80),
                            minLineLength=W // 4, maxLineGap=4)
    if lines is None:
        return 0.0
    angs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) <= MAX_SKEW:
            angs.append(a)
    return float(np.median(angs)) if len(angs) >= 2 else 0.0


def detect(img: np.ndarray, dpi: int = ocr.DPI) -> list[Table]:
    """img：已轉正的頁面影像（pdfium 的 BGR，dpi）。回傳表格（座標為 pt；有轉正時在 Table.rot 的座標系）。"""
    f = WORK_DPI / dpi
    g = cv2.resize(_gray(img), None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if abs(f - 1) > 1e-6 else _gray(img)
    # 紅章、淡底色不當格線：Otsu 與「比背景暗很多」取嚴的那個
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(otsu if otsu >= 1 else 128, float(np.median(g)) - 60)
    ink = (g < thr).astype(np.uint8)
    # 歪斜的頁：細格線在形態學開運算下會斷成階梯、格子分錯（歪 1.5～3° 就明顯變差），
    # 先依長直線的角度把影像轉正再找格線；字的座標用 Table.to_local 轉到同一個座標系
    ang = skew_angle(ink)
    rot = None
    if abs(ang) >= MIN_DESKEW:
        Hh, Ww = ink.shape
        M = cv2.getRotationMatrix2D((Ww / 2, Hh / 2), ang, 1.0)
        ink = cv2.warpAffine(ink, M, (Ww, Hh), flags=cv2.INTER_NEAREST, borderValue=0)
        rot = (ang, Ww / 2 / (WORK_DPI / 72.0), Hh / 2 / (WORK_DPI / 72.0))
    px_mm = WORK_DPI / 25.4
    k = max(int(LINE_MM * px_mm), 10)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    # 掃描的格線常有 1~2 px 斷點：沿線方向閉合，再稍微加粗讓交點相連
    h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    grid = cv2.dilate(h | v, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    s = WORK_DPI / 72.0
    out: list[Table] = []
    for i in range(1, n):
        x, y, w, hh, _ = stats[i]
        if w < MIN_TABLE_MM[0] * px_mm or hh < MIN_TABLE_MM[1] * px_mm:
            continue
        region = (lab[y:y + hh, x:x + w] == i).astype(np.uint8)
        hl = _count_lines(h[y:y + hh, x:x + w] & region, axis=1)
        vl = _count_lines(v[y:y + hh, x:x + w] & region, axis=0)
        if hl < 2 or vl < 2:
            continue
        cells, labels = _cells(region, px_mm)
        if len(cells) < 2:
            continue
        t = Table(x / s, y / s, (x + w) / s, (y + hh) / s, labels=labels, rot=rot)
        t.cells = [Cell((x + cx) / s, (y + cy) / s, (x + cx + cw) / s, (y + cy + ch) / s) for cx, cy, cw, ch in cells]
        out.append(t)
    return out


def _count_lines(m: np.ndarray, axis: int) -> int:
    """m 裡有幾條平行線（axis=1：水平線，看每一列有沒有墨；連續的列算一條）。"""
    prof = m.any(axis=axis)
    return int(np.count_nonzero(np.diff(np.concatenate([[0], prof.astype(np.int8), [0]])) == 1))


def _cells(region: np.ndarray, px_mm: float) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """格線區內被格線圍住的空白區塊 + 每個像素的格子編號。碰到區塊邊緣的不算：那是表格外面。"""
    free = (1 - region).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    H, W = region.shape
    out = []
    remap = np.full(n, -1, np.int32)
    for j in range(1, n):
        x, y, w, h, area = stats[j]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if w < MIN_CELL_MM[0] * px_mm or h < MIN_CELL_MM[1] * px_mm:
            continue
        remap[j] = len(out)
        out.append((int(x), int(y), int(w), int(h)))
    return out, remap[lab]


# ---------------------------------------------------------------- 組字
def _cx(w) -> float:
    return (w.x0 + w.x1) / 2


def _cy(w) -> float:
    return (w.y0 + w.y1) / 2


def _is_cjk(ch: str) -> bool:
    return "㐀" <= ch <= "鿿" or "豈" <= ch <= "﫿" or "　" <= ch <= "〿" or "＀" <= ch <= "￯"


def join_wrapped(parts: list[str]) -> str:
    """儲存格內的換行接起來：兩邊都是中文（含全形標點）直接接，否則補一個空白。"""
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 窄欄裡被折斷的數字（「18~2」+「2」）、符號前後直接接；英文單字之間才補空白
        glue = _is_cjk(out[-1:] or " ") or _is_cjk(p[0]) or (out[-1:].isdigit() and p[0].isdigit()) \
            or out[-1:] in "~-/<>(（=±" or p[0] in ")）<>~-/%=°"
        if out and not glue:
            out += " "
        out += p
    return out


def assemble(words: list, tables: list[Table], page_width: float) -> tuple[str, int]:
    """words：ocr.Word（pt）。回傳 (頁面文字, 還原成表格列的列數)。沒有表格 -> 跟 ocr.page_text 一樣。"""
    if not tables:
        return ocr.page_text(words), 0
    free: list = []
    for w in words:
        cell = None
        for t in tables:
            cx, cy = t.to_local(_cx(w), _cy(w))
            if t.x0 <= cx <= t.x1 and t.y0 <= cy <= t.y1 and (cell := t.cell_at(cx, cy)) is not None:
                break
        if cell is None:
            free.append(w)
        else:
            cell.words.append(w)
    items: list[tuple[float, float, str]] = []        # (y, x, 行文字)
    for line in ocr.to_lines(free):
        items.append((min(_cy(w) for w in line), line[0].x0, " ".join(w.text for w in line)))
    rows = 0
    for t in tables:
        filled = [c for c in t.cells if c.words]
        for row in _rows(filled):
            frames = [c for c in row if _is_frame(c, page_width)]
            if len(row) == 1 or frames:
                # 外框/本文框：格子裡照原本逐行輸出；同列其他小格照列輸出
                for c in frames or row:
                    for line in ocr.to_lines(c.words):
                        items.append((min(_cy(w) for w in line), line[0].x0, " ".join(w.text for w in line)))
                rest = [c for c in row if c not in (frames or row)]
                if len(rest) >= 2:
                    items.append((min(c.y0 for c in rest) + 0.01, rest[0].x0, " | ".join(_cell_text(c) for c in rest)))
                elif rest:
                    items.append((rest[0].y0 + 0.01, rest[0].x0, _cell_text(rest[0])))
                continue
            items.append((min(c.y0 for c in row) + 0.01, row[0].x0, " | ".join(_cell_text(c) for c in row)))
            rows += 1
    items.sort(key=lambda it: (it[0], it[1]))
    return "\n".join(t for _, _, t in items if t.strip()), rows


def _cell_text(c: Cell) -> str:
    return join_wrapped([" ".join(w.text for w in line) for line in ocr.to_lines(c.words)])


def _is_frame(c: Cell, page_width: float) -> bool:
    return (c.x1 - c.x0) >= FRAME_W * page_width and len(ocr.to_lines(c.words)) >= FRAME_LINES


def _rows(cells: list[Cell]) -> list[list[Cell]]:
    """上緣對齊的儲存格成一列（容許半個最小格高的誤差）；列內依 x 排。"""
    if not cells:
        return []
    tol = max(2.0, 0.5 * min(c.y1 - c.y0 for c in cells))
    out: list[list[Cell]] = []
    for c in sorted(cells, key=lambda c: (c.y0, c.x0)):
        if out and abs(c.y0 - out[-1][0].y0) <= tol:
            out[-1].append(c)
        else:
            out.append([c])
    return [sorted(r, key=lambda c: c.x0) for r in out]
