# AI Finance Assistant — Technical Design Document (TDD)

## 1. Executive Summary & Design Principles
The **AI Finance Assistant** is a multi-agent conversational AI system designed to democratize financial education, provide portfolio analysis, contextualize live market data, and model financial goals.

### Guiding Principles
- **Separation of Concerns:** Distinct specialist nodes for domain-isolated execution.
- **Determinism First:** Mathematical calculations (growth formulas, HHI diversification, asset allocation) are executed in deterministic Python tools rather than left to LLM probability.
- **Context Isolation:** Per-agent message scratchpads use state reducers to reduce cross-agent state contamination.
- **Defense in Depth:** RAG confidence thresholds, citation validation, in-memory caching, exponential backoff, and offline fallbacks are used where external services can fail.

---

## 2. System Architecture & Topology

The system implements the **Router–Worker–Synthesizer** pattern via LangGraph's dynamic `Send` fan-out API:


```

```
                          [ User Query ]
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Orchestrator Node   │
                    │ (Intent Decomposition)│
                    └───────────┬───────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │      Parallel Dynamic Fan-Out     │
              │        (LangGraph Send API)       │
              ▼                                   ▼
  ┌───────────────────────┐           ┌───────────────────────┐
  │    RAG Specialists    │           │  Analytical Specialists│
  │ --------------------- │           │ --------------------- │
  │ • Finance Q&A Agent   │           │ • Portfolio Analysis  │
  │ • Tax Education Agent │           │ • Market Intelligence │
  │ • Glossary Retriever  │           │ • Goal Planning Agent │
  └───────────┬───────────┘           └───────────┬───────────┘
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Synthesizer Node    │
                    │  (Merge & Citations)  │
                    └───────────┬───────────┘
                                │
                                ▼
                    [ User Interface Output ]

```

```

---

## 3. State Schema & Reducer Protocols

```python
class FinanceAssistantState(TypedDict):
    user_query: str
    messages: Annotated[list[AnyMessage], operator.add]
    user_profile: dict
    tasks: list[AgentTask]
    requires_synthesis: bool
    agent_results: Annotated[list[dict], reset_or_add]
    final_answer: str

    # Agent scratchpads (isolated per turn)
    qa_messages: Annotated[list[AnyMessage], reset_or_add]
    portfolio_messages: Annotated[list[AnyMessage], reset_or_add]
    market_messages: Annotated[list[AnyMessage], reset_or_add]
    goal_messages: Annotated[list[AnyMessage], reset_or_add]
    news_messages: Annotated[list[AnyMessage], reset_or_add]
    tax_messages: Annotated[list[AnyMessage], reset_or_add]

```

### State Isolation Mechanism (`reset_or_add`)

Standard `operator.add` accumulates conversation history monotonically. To prevent internal tool messages and chain-of-thought scratchpads from accumulating across multi-turn sessions, worker-specific channels use `reset_or_add`:

* When a new turn initiates, `None` is passed to clear scratchpads.


* When nodes yield findings, results are appended only within the current execution turn.



---

## 4. Vector Store & RAG Retrieval Design

### Semantic Indexing Pipeline

* **Embedding Model:** `text-embedding-3-small`.


* **Knowledge Base:** 54+ curated articles in JSON format covering Investing Basics, Portfolio Management, Tax Accounts, Risk Planning, and Glossary.


* **Similarity Metric:** Cosine similarity calculated over NumPy vectors.


* **Confidence Calibration:**
* `score >= 0.35`: High confidence.


* `score < 0.35`: Low confidence response requiring conservative wording.

The active store is an in-memory implementation, not FAISS. The retriever supports category filters and returns document IDs, titles, content, and relevance scores. Finance Q&A and Tax agent responses validate generated `[ref: ID]` markers against retrieved documents.





---

## 5. Market Data Integration & Resilience

* **Primary Provider:** `yfinance`.


* **Caching Layer:** 30-minute in-memory TTL dictionary (`_MARKET_CACHE` and `_HISTORY_CACHE`).


* **Rate-Limit Resilience:** Exponential backoff with 3 retry attempts (`0.5s * 2^attempt`).


* **Name Resolution:** Full company/fund names are resolved with Yahoo Finance search; common benchmark fallback data remains available when live retrieval fails.
* **Offline Fallback:** Static baseline data for high-liquidity benchmark tickers (`SPY`, `VOO`, `VTI`, `BND`, `QQQ`, `AAPL`, `MSFT`).



---

## 6. Performance Benchmarks & Edge Cases

| Scenario | Execution Route | Latency Benchmark |
| --- | --- | --- |
| Single Intent Q&A | `Orchestrator` → `finance_qa` → `Synthesizer` | Environment-dependent

 |
| Parallel Dual Intent | `Orchestrator` → [`market_agent`, `tax_agent`] → `Synthesizer` | Environment-dependent

 |
| Market Quote (Cached) | `market_agent` → In-Memory Cache | Environment-dependent

 |
| Market Quote (Live) | `market_agent` → `yfinance` | Environment-dependent

 |
| Malformed JSON Input | Portfolio tool validation | Environment-dependent

 |

---

## 7. Deployment & Verification Runbook

```bash
# 1. Environment Activation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Automated Test Suite Execution
python -m pytest -q --cov=src --cov-report=term-missing

# 3. Launching Streamlit Web App
streamlit run src/web_app/app.py

```

The latest verified baseline is 38 passing tests and 81% total coverage. Coverage and latency depend on the selected Python environment, installed provider versions, network access, and API response times.

## Optional MCP Interface

`mcp_server.py` uses FastMCP and exposes `get_quote` and `evaluate_portfolio`. It reuses the production market and portfolio tools. MCP-specific automated tests and Claude Desktop configuration are not currently included.

```

```