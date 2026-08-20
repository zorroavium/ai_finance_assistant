"""
Real-time market data retrieval tool using yfinance with in-memory TTL caching.
Provides current quotes, 52-week ranges, moving averages, and fallback resilience.
"""

import json
import time
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
import yfinance as yf

from src.core.config import CONFIG

CACHE_TTL_SECONDS = CONFIG.get("apis", {}).get("cache_ttl_minutes", 30) * 60

# In-memory cache: { "TICKER": (timestamp, data_dict) }
_MARKET_CACHE: dict[str, tuple[float, dict]] = {}

# Resilient fallback quotes if network or rate limit fails
_FALLBACK_QUOTES = {
    "SPY": {"price": 540.20, "change_pct": 0.35, "name": "SPDR S&P 500 ETF Trust", "pe_ratio": 26.5},
    "VOO": {"price": 495.10, "change_pct": 0.38, "name": "Vanguard S&P 500 ETF", "pe_ratio": 26.4},
    "VTI": {"price": 265.50, "change_pct": 0.28, "name": "Vanguard Total Stock Market ETF", "pe_ratio": 25.1},
    "BND": {"price": 72.80, "change_pct": -0.05, "name": "Vanguard Total Bond Market ETF", "pe_ratio": None},
    "QQQ": {"price": 475.90, "change_pct": 0.82, "name": "Invesco QQQ Trust", "pe_ratio": 32.1},
    "AAPL": {"price": 220.50, "change_pct": 1.15, "name": "Apple Inc.", "pe_ratio": 33.2},
    "MSFT": {"price": 445.00, "change_pct": 0.65, "name": "Microsoft Corporation", "pe_ratio": 35.8},
}


class MarketQuoteInput(BaseModel):
    ticker: str = Field(..., description="Stock or ETF ticker symbol (e.g. 'AAPL', 'VOO', 'SPY').")


def _fetch_from_yfinance(ticker: str) -> dict:
    """Fetch live quote metrics via yfinance."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("navPrice")
    )

    if not current_price:
        # Try fast_info
        fast_info = getattr(t, "fast_info", None)
        if fast_info:
            current_price = getattr(fast_info, "last_price", None)

    if not current_price:
        raise ValueError(f"Could not retrieve valid price for {ticker}")

    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or current_price
    change_pct = ((current_price - prev_close) / prev_close) * 100.0 if prev_close else 0.0

    return {
        "ticker": ticker.upper(),
        "name": info.get("shortName") or info.get("longName") or ticker.upper(),
        "current_price": round(float(current_price), 2),
        "previous_close": round(float(prev_close), 2),
        "change_percent": f"{change_pct:+.2f}%",
        "currency": info.get("currency", "USD"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_day_avg": info.get("fiftyDayAverage"),
        "two_hundred_day_avg": info.get("twoHundredDayAverage"),
        "pe_ratio": info.get("trailingPE"),
        "source": "live_yfinance",
        "cached": False,
    }


@tool(args_schema=MarketQuoteInput)
def get_market_quote(ticker: str) -> str:
    """Retrieve real-time market price, daily change, and key statistics for a stock or ETF ticker."""
    clean_ticker = ticker.upper().strip()
    now = time.time()

    # 1. Check in-memory cache
    if clean_ticker in _MARKET_CACHE:
        cached_time, cached_data = _MARKET_CACHE[clean_ticker]
        if now - cached_time < CACHE_TTL_SECONDS:
            res = dict(cached_data)
            res["cached"] = True
            res["cache_age_seconds"] = int(now - cached_time)
            return json.dumps(res, indent=2)

    # 2. Fetch live data
    try:
        data = _fetch_from_yfinance(clean_ticker)
        _MARKET_CACHE[clean_ticker] = (now, data)
        return json.dumps(data, indent=2)
    except Exception as e:
        # 3. Fallback resilience
        if clean_ticker in _FALLBACK_QUOTES:
            fb = _FALLBACK_QUOTES[clean_ticker]
            return json.dumps({
                "ticker": clean_ticker,
                "name": fb["name"],
                "current_price": fb["price"],
                "change_percent": f"{fb['change_pct']:+.2f}%",
                "source": "fallback_offline_cache",
                "warning": f"Live fetch failed ({str(e)}). Returned fallback quote."
            }, indent=2)

        return json.dumps({
            "error": f"Failed to retrieve market data for {clean_ticker}: {str(e)}",
            "status": "failed"
        })