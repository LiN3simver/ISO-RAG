"""在有 TLS 檢查代理（以自簽 CA 重簽 HTTPS）的環境，Python 預設只信 certifi，下載模型會
SSLCertVerificationError。改用作業系統憑證庫（這類 CA 通常已安裝在 OS 的信任清單）。"""
from __future__ import annotations

_done = False


def use_os_certs() -> None:
    global _done
    if _done:
        return
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001  沒裝就照舊
        pass
    _done = True
