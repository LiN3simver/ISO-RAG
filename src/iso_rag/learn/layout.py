"""版型指紋與線上分群：判斷一頁是不是「沒看過的版型」，並把 OCR 策略記在版型底下。

指紋只用影像（OCR 之前就算得出來，才能決定先用哪個 OCR 策略）。100 dpi 小圖：
  - 文字墨跡密度：頁面切 6 列 × 4 欄，每格墨跡比例（開根號、整組正規化）—— 表頭、本文、表格的位置
  - 格線分布：6 個水平帶各自的水平/垂直長線密度（整組正規化，再乘上格線總量）—— 表單、會審表很高
  - 方向：橫式/直式
版型 = 一群「成員頁」：新的一頁跟各版型成員頁的最高相似度 ≥ MATCH 就歸入，否則開新版型
（不用平均中心：本文頁差異大，平均起來會像所有頁，最後全部併成一群）。
MATCH 以留一份文件的方式校準：標準版型文件的頁幾乎都認得出來，只有真的少見的表單頁會開新版型。
版型頁數 < NEW_PAGES 時視為「新版型」，OCR 會多探索幾種策略。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .quality import ink_layers
from .store import Store, now

GRID_R, GRID_C = 6, 4
W_GRID, W_LINES, W_ORIENT = 1.0, 1.0, 0.6
MATCH = 0.90
NEW_PAGES = 5
MAX_EXEMPLARS = 40


def fingerprint(small: np.ndarray) -> np.ndarray:
    h, w = small.shape[:2]
    text, lh, lv = ink_layers(small)
    grid = np.sqrt([text[i * h // GRID_R:(i + 1) * h // GRID_R, j * w // GRID_C:(j + 1) * w // GRID_C].mean()
                    for i in range(GRID_R) for j in range(GRID_C)])
    grid = grid / (np.linalg.norm(grid) or 1.0)
    lines = np.sqrt([m[i * h // GRID_R:(i + 1) * h // GRID_R].mean() * 10 for m in (lh, lv) for i in range(GRID_R)])
    lines = lines / (np.linalg.norm(lines) or 1.0) * min(1.0, float(np.sqrt((lh.mean() + lv.mean()) * 50)))
    v = np.concatenate([W_GRID * grid, W_LINES * lines, [W_ORIENT * float(w > h)]])
    return (v / (np.linalg.norm(v) or 1.0)).astype(np.float32)


DIM = GRID_R * GRID_C + 2 * GRID_R + 1


@dataclass
class Profile:
    id: str
    exemplars: np.ndarray            # (k, D) 成員頁指紋，最多 MAX_EXEMPLARS 個（第一頁 + 最近的）
    n: int
    docs: list[str] = field(default_factory=list)
    created: str = ""
    origin: str = ""

    @property
    def new(self) -> bool:
        return self.n < NEW_PAGES


class Profiles:
    def __init__(self, store: Store):
        self.store = store
        self.items: list[Profile] = [
            Profile(pid, np.frombuffer(c, dtype=np.float32).reshape(-1, DIM).copy(), n, json.loads(docs), created,
                    origin)
            for pid, c, n, docs, created, _, origin in store.profiles_rows()]

    def match(self, fp: np.ndarray) -> tuple[Profile | None, float]:
        if not self.items:
            return None, 0.0
        sims = [float((p.exemplars @ fp).max()) for p in self.items]
        i = int(np.argmax(sims))
        return self.items[i], sims[i]

    def assign(self, fp: np.ndarray, doc: str, origin: str) -> tuple[Profile, float, bool]:
        """歸入最像的版型（加入成員頁）或開新版型。回傳 (版型, 相似度, 是否新開)。"""
        p, sim = self.match(fp)
        created = p is None or sim < MATCH
        if created:
            p = Profile(f"L{len(self.items) + 1:03d}", fp[None, :].copy(), 0, [], now(), origin)
            self.items.append(p)
        else:
            ex = np.vstack([p.exemplars, fp[None, :]])
            p.exemplars = np.vstack([ex[:1], ex[1:][-(MAX_EXEMPLARS - 1):]])
        p.n += 1
        if doc and doc not in p.docs:
            p.docs.append(doc)
        self.store.profile_put(p.id, p.exemplars.tobytes(), p.n, p.docs, p.created, p.origin)
        return p, sim, created

    def get(self, pid: str) -> Profile | None:
        return next((p for p in self.items if p.id == pid), None)
