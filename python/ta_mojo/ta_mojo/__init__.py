"""ta-mojo: fast technical-analysis indicators matching pandas-ta-classic.

Same warmup/NaN-prefix semantics, same values (within 1e-10) as the
pandas-based oracle — powered by a Mojo kernel where the platform supports it
(macOS arm64, Linux x86_64), with a vendored pure-Python fallback everywhere
else (including Windows).

    import numpy as np
    import ta_mojo

    close = np.array([...], dtype=float)
    ta_mojo.ema(close, length=10)                # EMA_10
    ta_mojo.rsi(close, length=14)                # RSI_14 (Wilder)
    ta_mojo.atr(high, low, close, length=14)     # ATRr_14
    res = ta_mojo.macd(close, 12, 26, 9)         # MACD_12_26_9
    res.macd, res.signal, res.histogram          # float64 ndarrays

Set TA_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.
"""

from ta_mojo._native import backend_info, native_available
from ta_mojo.core import MacdResult, atr, ema, macd, rsi

__version__ = "0.1.1"  # x-release-please-version
__all__ = [
    "ema",
    "rsi",
    "atr",
    "macd",
    "MacdResult",
    "backend_info",
    "native_available",
    "__version__",
]
