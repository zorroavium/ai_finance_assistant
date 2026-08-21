"""
Real-time market data retrieval and historical trend tools using yfinance.
Features: In-memory TTL caching, exponential backoff retries, and freshness tracking.
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
import yfinance as yf

from src.core.config import CONFIG

CACHE_TTL_SECONDS = CONFIG.get("apis", {}).get("cache_ttl_minutes", 30) * 60

# In-memory caches
_MARKET_CACHE: dict[str, tuple[float, dict]] = {}
_HISTORY_CACHE: dict[str, tuple[float, dict]] = {}

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


class MarketHistoryInput(BaseModel):
    ticker: str = Field(..., description="Stock or ETF ticker symbol.")
    period: Optional[str] = Field(default="6mo", description="Time period: '1mo', '3mo', '6mo', '1y', '5y'.")


def _fetch_yfinance_with_backoff(ticker: str, max_retries: int = 3) -> dict:
    """Fetches quote data with exponential backoff on failure."""
    last_err = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            # Yahoo's quote-summary endpoint behind ``Ticker.info`` can fail
            # independently of the historical chart endpoint. Use chart data
            # for the price and treat quote metadata as optional.
            history = t.history(period="5d", auto_adjust=False)
            if history.empty or "Close" not in history:
                raise ValueError(f"Could not retrieve valid price for {ticker}")

            close_prices = history["Close"].dropna().tolist()
            current_price = close_prices[-1] if close_prices else None
            previous_close = close_prices[-2] if len(close_prices) > 1 else current_price

            try:
                info = t.info or {}
            except Exception:
                info = {}

            if not current_price:
                current_price = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("navPrice")
                )

            if not current_price:
                fast_info = getattr(t, "fast_info", None)
                if fast_info:
                    current_price = getattr(fast_info, "last_price", None)

            if not current_price:
                raise ValueError(f"Could not retrieve valid price for {ticker}")

            prev_close = previous_close or info.get("regularMarketPreviousClose") or info.get("previousClose") or current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100.0 if prev_close else 0.0

            utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

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
                "timestamp_utc": utc_now,
                "freshness": "LIVE",
                "cached": False,
                "status": "success",
            }
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (2 ** attempt))

    raise last_err or RuntimeError("Max retries exceeded fetching market data")


@tool(args_schema=MarketQuoteInput)
def get_market_quote(ticker: str) -> str:
    """Retrieve real-time market price, daily change, and freshness metadata for a stock or ETF."""
    clean_ticker = ticker.upper().strip()
    now = time.time()

    # 1. Check in-memory TTL cache
    if clean_ticker in _MARKET_CACHE:
        cached_time, cached_data = _MARKET_CACHE[clean_ticker]
        if now - cached_time < CACHE_TTL_SECONDS:
            res = dict(cached_data)
            res["cached"] = True
            res["freshness"] = f"CACHED ({int((now - cached_time) / 60)}m old)"
            res["cache_age_seconds"] = int(now - cached_time)
            return json.dumps(res, indent=2)

    # 2. Live fetch with exponential backoff
    try:
        data = _fetch_yfinance_with_backoff(clean_ticker)
        _MARKET_CACHE[clean_ticker] = (now, data)
        return json.dumps(data, indent=2)
    except Exception as e:
        # 3. Fallback response
        if clean_ticker in _FALLBACK_QUOTES:
            fb = _FALLBACK_QUOTES[clean_ticker]
            return json.dumps({
                "ticker": clean_ticker,
                "name": fb["name"],
                "current_price": fb["price"],
                "change_percent": f"{fb['change_pct']:+.2f}%",
                "freshness": "FALLBACK_OFFLINE",
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "warning": f"Live fetch failed ({str(e)}). Returned fallback dataset.",
                "status": "success",
            }, indent=2)

        return json.dumps({
            "error": f"Failed to retrieve market data for {clean_ticker}: {str(e)}",
            "status": "failed",
        })


@tool(args_schema=MarketHistoryInput)
def get_market_history(ticker: str, period: str = "6mo") -> str:
    """Retrieve historical closing price series for trend analysis and chart visualization."""
    clean_ticker = ticker.upper().strip()
    cache_key = f"{clean_ticker}_{period}"
    now = time.time()

    if cache_key in _HISTORY_CACHE:
        cached_time, cached_data = _HISTORY_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return json.dumps(cached_data, indent=2)

    try:
        t = yf.Ticker(clean_ticker)
        df = t.history(period=period)
        if df.empty:
            return json.dumps({"error": f"No historical data for {clean_ticker}", "status": "failed"})

        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        closes = [round(float(c), 2) for c in df["Close"].tolist()]

        data = {
            "ticker": clean_ticker,
            "period": period,
            "dates": dates,
            "close_prices": closes,
            "start_price": closes[0] if closes else None,
            "end_price": closes[-1] if closes else None,
            "period_return_pct": round(((closes[-1] - closes[0]) / closes[0]) * 100.0, 2) if closes else 0.0,
            "status": "success",
        }
        _HISTORY_CACHE[cache_key] = (now, data)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve historical trend: {str(e)}", "status": "failed"})