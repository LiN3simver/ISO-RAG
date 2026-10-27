"""BAAI/bge-m3 ONNX 編碼器（onnxruntime + tokenizers，免 torch）。

輸出與 sentence-transformers 版本一致：
- BAAI 官方匯出的 onnx/model.onnx，輸入 input_ids/attention_mask，輸出 sentence_embedding
  （CLS pooling 已在圖內）；這裡只做分詞與 L2 正規化。
- 要抓整個 onnx/ 目錄（含 Constant_7_attr__value 外部資料檔）+ 根目錄 tokenizer.json。
- revision 釘死，索引 manifest 會記錄 revision 與 max_seq，載入時比對（ingest 與 query
  的 max_seq 不一致會讓向量對不上）。

模型位置：EMBED_MODEL_DIR（已下載好的 snapshot 目錄）優先；否則下載到 models/hf。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import ROOT, env

MODEL = "BAAI/bge-m3"
REVISION = env("EMBED_REVISION", "5617a9f61b028005a4858fdac845db406aefb181")
MAX_SEQ = int(env("EMBED_MAX_SEQ", "512"))
DIM = 1024


def model_dir() -> Path:
    d = os.environ.get("EMBED_MODEL_DIR")
    if d and (Path(d) / "onnx" / "model.onnx").exists():
        return Path(d)
    from huggingface_hub import snapshot_download

    from ..netfix import use_os_certs
    use_os_certs()
    return Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=["onnx/*", "tokenizer.json"],
                                  cache_dir=str(ROOT / "models" / "hf")))


class Encoder:
    def __init__(self, max_seq: int = MAX_SEQ, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = model_dir()
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_seq)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>"), pad_token="<pad>")
        so = ort.SessionOptions()
        threads = threads or int(env("EMBED_THREADS", "0"))
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(d / "onnx" / "model.onnx"), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.max_seq = max_seq

    def encode(self, texts: list[str] | str, batch_size: int = 16) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        # 依長度排序再分批，padding 少很多；最後還原順序
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tok.encode_batch([texts[j] for j in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            msk = np.array([e.attention_mask for e in enc], dtype=np.int64)
            (vecs,) = self.sess.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": msk})
            out.append((idx, vecs))
        res = np.zeros((len(texts), DIM), dtype=np.float32)
        for idx, vecs in out:
            res[idx] = vecs
        n = np.linalg.norm(res, axis=1, keepdims=True)
        return (res / np.clip(n, 1e-12, None)).astype(np.float32)


@lru_cache(maxsize=1)
def get() -> Encoder:
    return Encoder()
