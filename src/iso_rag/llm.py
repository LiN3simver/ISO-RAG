"""LLM 客戶端（Anthropic Messages API 相容的 gateway）。

處理這類 gateway／推理模型常見的幾個行為：
  1. 認證用 Bearer token -> SDK 要用 auth_token=，不是 api_key=。
  2. 推理模型會先 thinking，而且算進 max_tokens；預算不夠時 text 是空的、
     stop_reason='max_tokens'。所以 max_tokens 是「思考+答案」總預算。
  3. 串流的 text_stream 只吐 text_delta，thinking 自動略過。
  4. 有些 gateway 會快取「完全相同」的請求 -> 在最後一則 user 訊息前面加上
     system/參數的雜湊標記，system 改了就一定是新請求。
  5. httpx 的 read timeout 會被 SSE keep-alive 重置，擋不住慢請求 ->
     在背景執行緒跑，主執行緒自己看時鐘（wall-clock timeout）。
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass

import anthropic

from .config import env

BASE_URL = env("LLM_BASE_URL", "http://127.0.0.1:4000")
MODEL = env("LLM_MODEL", "local-model")   # 實際模型名稱寫在 .env
MAX_TOKENS = int(env("LLM_MAX_TOKENS", "4000"))
TIMEOUT = float(env("LLM_TIMEOUT", "180"))


@dataclass
class LLMResult:
    text: str
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    seconds: float


class EmptyStreamError(RuntimeError):
    pass


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=BASE_URL, auth_token=env("LLM_API_KEY"),
                               timeout=TIMEOUT, max_retries=0)


def _with_policy(system: str, user: str, max_tokens: int) -> str:
    digest = hashlib.sha256(json.dumps(
        {"model": MODEL, "system": system, "max_tokens": max_tokens},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    return f"[request-policy:{digest}]（設定識別碼，回答請勿引用）\n{user}"


def _run(system: str, user: str, max_tokens: int, closer: list) -> LLMResult:
    t0 = time.monotonic()
    with _client().messages.stream(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": _with_policy(system, user, max_tokens)}],
    ) as stream:
        closer.append(stream)
        text = "".join(stream.text_stream)
        try:
            msg = stream.get_final_message()
        except AssertionError:
            # 有些 gateway 把錯誤（例如模型不存在）當成非 SSE 的一行回 200，SDK 收不到 message_start
            raise EmptyStreamError("gateway 回了空串流（沒有 message_start）") from None
    usage = getattr(msg, "usage", None)
    return LLMResult(text.strip(), msg.stop_reason,
                     getattr(usage, "input_tokens", None),
                     getattr(usage, "output_tokens", None),
                     time.monotonic() - t0)


def complete(system: str, user: str, *, max_tokens: int = MAX_TOKENS,
             timeout: float = TIMEOUT, on_wait=None, heartbeat: float = 10.0) -> LLMResult:
    """要一則完整回答；超過 timeout 秒就中止串流並丟 TimeoutError。
    on_wait(已等秒數) 每 heartbeat 秒呼叫一次（給 SSE 送「還在等」的訊號）。"""
    box: queue.Queue = queue.Queue(maxsize=1)
    closer: list = []

    def work():
        try:
            box.put(("ok", _run(system, user, max_tokens, closer)))
        except BaseException as e:  # noqa: BLE001
            box.put(("err", e))

    threading.Thread(target=work, daemon=True, name="llm").start()
    t0 = time.monotonic()
    while True:
        waited = time.monotonic() - t0
        if waited >= timeout:
            for s in closer:
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
            raise TimeoutError(f"LLM 超過 {timeout:.0f} 秒未完成")
        try:
            kind, val = box.get(timeout=min(heartbeat, timeout - waited))
            break
        except queue.Empty:
            if on_wait:
                on_wait(time.monotonic() - t0)
    if kind == "err":
        raise val
    if not val.text and val.stop_reason == "max_tokens":
        raise RuntimeError(f"思考用光了 max_tokens={max_tokens}，沒有產生答案；請調高 LLM_MAX_TOKENS")
    return val


def list_models(timeout: float = 15) -> list[str]:
    import httpx
    r = httpx.get(f"{BASE_URL}/v1/models", timeout=timeout,
                  headers={"Authorization": f"Bearer {env('LLM_API_KEY')}"})
    r.raise_for_status()
    return [m.get("id") for m in r.json().get("data", [])]


def diagnose(e: BaseException) -> str:
    if isinstance(e, TimeoutError):
        return f"逾時：{e}"
    if isinstance(e, anthropic.AuthenticationError):
        return "認證失敗（401）：LLM_API_KEY 錯誤或過期"
    if isinstance(e, anthropic.RateLimitError):
        return "被限流（429）：稍後再試，不需要換 key"
    if isinstance(e, anthropic.NotFoundError):
        return f"找不到模型（404）：LLM_MODEL={MODEL} 不在 gateway 上"
    if isinstance(e, anthropic.BadRequestError):
        return f"請求被拒（400）：{getattr(e, 'message', e)}"
    if isinstance(e, anthropic.APIConnectionError):
        return f"連不到 gateway {BASE_URL}：{e}"
    if isinstance(e, EmptyStreamError):
        try:
            models = list_models(timeout=5)
        except Exception:  # noqa: BLE001
            return f"{e}，且 /v1/models 也查不到"
        if MODEL not in models:
            return (f"gateway 上沒有模型 {MODEL}（目前可用：{', '.join(models) or '無'}），"
                    "請確認模型是否上線，或改 .env 的 LLM_MODEL")
        return str(e)
    return f"{type(e).__name__}: {e}"
