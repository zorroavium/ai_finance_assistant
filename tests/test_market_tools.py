import json
import pandas as pd

import src.tools.market_tools as market_tools
from src.tools.market_tools import get_market_quote, get_market_history, normalize_ticker


def test_company_name_normalizes_to_ticker(monkeypatch):
    class SearchResult:
        quotes = [{"quoteType": "EQUITY", "symbol": "MSFT"}]

    monkeypatch.setattr(market_tools.yf, "Search", lambda query: SearchResult())
    assert normalize_ticker("Microsoft") == "MSFT"
    assert normalize_ticker("Microsoft Corporation") == "MSFT"


def test_get_market_quote_freshness_and_metadata():
    output = get_market_quote.invoke({"ticker": "VOO"})
    data = json.loads(output)
    assert data["ticker"] == "VOO"
    assert "current_price" in data
    assert "freshness" in data
    assert "timestamp_utc" in data


def test_get_market_history_trend():
    output = get_market_history.invoke({"ticker": "SPY", "period": "1mo"})
    data = json.loads(output)
    assert data["ticker"] == "SPY"
    assert "close_prices" in data
    assert len(data["close_prices"]) > 5
    assert "period_return_pct" in data


def test_quote_uses_history_when_quote_metadata_fails(monkeypatch):
    class MetadataFailureTicker:
        @property
        def info(self):
            raise RuntimeError("quote summary unavailable")

        def history(self, period, auto_adjust):
            return pd.DataFrame({"Close": [100.0, 102.5]})

    monkeypatch.setattr(market_tools.yf, "Ticker", lambda ticker: MetadataFailureTicker())

    data = market_tools._fetch_yfinance_with_backoff("TEST")

    assert data["current_price"] == 102.5
    assert data["previous_close"] == 100.0
    assert data["status"] == "success"


def test_invalid_or_mistyped_ticker_returns_actionable_error(monkeypatch):
    class EmptyTicker:
        def history(self, period, auto_adjust=False):
            return pd.DataFrame()

    monkeypatch.setattr(market_tools.yf, "Ticker", lambda ticker: EmptyTicker())
    data = json.loads(get_market_quote.invoke({"ticker": "MFST"}))

    assert data["status"] == "failed"
    assert data["suggested_ticker"] == "MSFT"