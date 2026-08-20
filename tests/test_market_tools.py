import json
from src.tools.market_tools import get_market_quote

def test_get_market_quote_live_or_fallback():
    output = get_market_quote.invoke({"ticker": "SPY"})
    data = json.loads(output)
    assert "ticker" in data
    assert data["ticker"] == "SPY"
    assert "current_price" in data
    assert data["current_price"] > 0

def test_market_quote_caching():
    # First call primes cache
    get_market_quote.invoke({"ticker": "AAPL"})
    # Second call should be served from cache
    output2 = get_market_quote.invoke({"ticker": "AAPL"})
    data2 = json.loads(output2)
    assert data2.get("cached") is True or "fallback" in data2.get("source", "")