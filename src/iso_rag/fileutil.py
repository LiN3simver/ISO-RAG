"""檔案小工具：sha256 快取與原子寫檔。

- sha256(path)：路徑＋大小＋修改時間沒變就沿用上次算的雜湊（data/cache/filehash.sqlite），只重算有變的檔。
- atomic_write(path, data)：先寫 .part 再 replace，中途中斷不會留下寫一半的檔。
- display_name(name)：日誌用的短檔名，去掉檔名開頭的「<十六進位識別碼>_」前綴。
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path

from .config import CACHE_DIR

DB = CACHE_DIR / "filehash.sqlite"
# 有些文件系統匯出的檔名前面帶識別碼（UUID／32 位雜湊，≥16 字）再接「_」，日誌只顯示後面的文件名稱
_ID_PREFIX = re.compile(r"^[0-9A-Fa-f][0-9A-Fa-f-]{15,}_")


def _db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS h (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, sha256 TEXT)")
    return db


def sha256(path: Path) -> str:
    path = Path(path)
    st = path.stat()
    key = str(path.resolve())
    db = _db()
    try:
        r = db.execute("SELECT sha256 FROM h WHERE path = ? AND size = ? AND mtime = ?",
                       (key, st.st_size, st.st_mtime_ns)).fetchone()
        if r:
            return r[0]
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        db.execute("INSERT OR REPLACE INTO h VALUES (?,?,?,?)", (key, st.st_size, st.st_mtime_ns, h.hexdigest()))
        db.commit()
        return h.hexdigest()
    finally:
        db.close()


def atomic_write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")     # 建索引與學習指令可能同時寫同一個檔
    # 文字也照位元組寫：write_text 在 Windows 會把 \n 換成 \r\n，檔案雜湊就跟寫入前算的對不上
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    tmp.replace(path)
    return path


def display_name(name: str) -> str:
    """'<識別碼>_SOP-QA-001 樣品保存作業規範.pdf' -> 'SOP-QA-001 樣品保存作業規範.pdf'；沒有前綴就原樣。"""
    return _ID_PREFIX.sub("", name, count=1)
