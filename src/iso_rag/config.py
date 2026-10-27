"""集中讀設定（專案根目錄的 .env）與資料目錄位置。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"{name} 未設定")
    return val


DATA = ROOT / "data"
PDF_DIR = DATA / "pdf"
CACHE_DIR = DATA / "cache"
INDEX_DIR = DATA / "index"
