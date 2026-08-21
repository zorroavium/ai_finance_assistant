# AI Finance Assistant — Technical Design Document (TDD)

## 1. Executive Summary & Design Principles
The **AI Finance Assistant** is a multi-agent conversational AI system designed to democratize financial education, provide portfolio analysis, contextualize live market data, and model financial goals[cite: 27].

### Guiding Principles
- **Separation of Concerns:** Distinct specialist nodes for domain-isolated execution[cite: 27].
- **Determinism First:** Mathematical calculations (growth formulas, HHI diversification, asset allocation) are executed in pure Python deterministic tools rather than left to LLM probability[cite: 27].
- **Context Isolation:** Per-agent message scratchpads using state reducers prevent token bloat and cross-agent hallucination[cite: 27].
- **Defense in Depth:** Semantic thresholding for RAG, in-memory caching with exponential backoff for APIs, and mock fallbacks for offline resilience[cite: 27].

---

## 2. System Architecture & Topology

The system implements the **Router–Worker–Synthesizer** pattern via LangGraph's dynamic `Send` fan-out API[cite: 27]:


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

* **Embedding Model:** `text-embedding-3-small` (1536 dimensions).


* **Knowledge Base:** 54+ curated articles in JSON format covering Investing Basics, Portfolio Management, Tax Accounts, Risk Planning, and Glossary.


* **Similarity Metric:** Cosine similarity over normalized vectors.


* **Confidence Calibration:**
* `score >= 0.35`: High confidence (direct attribution, references attached).


* `score < 0.35`: Low confidence fallback (conservative response with citation disclaimer).





---

## 5. Market Data Integration & Resilience

* **Primary Provider:** `yfinance` with Alpha Vantage configuration fallback.


* **Caching Layer:** 30-minute in-memory TTL dictionary (`_MARKET_CACHE` and `_HISTORY_CACHE`).


* **Rate-Limit Resilience:** Exponential backoff with 3 retry attempts (`0.5s * 2^attempt`).


* **Offline Fallback:** Static baseline dictionary for high-liquidity benchmark tickers (`SPY`, `VOO`, `VTI`, `BND`, `QQQ`, `AAPL`, `MSFT`).



---

## 6. Performance Benchmarks & Edge Cases

| Scenario | Execution Route | Latency Benchmark |
| --- | --- | --- |
| Single Intent Q&A | `Orchestrator` → `finance_qa` → `Synthesizer` | ~1.1s

 |
| Parallel Dual Intent | `Orchestrator` → [`market_agent`, `tax_agent`] → `Synthesizer` | ~1.8s

 |
| Market Quote (Cached) | `market_agent` → In-Memory Cache | < 15ms

 |
| Market Quote (Live) | `market_agent` → `yfinance` | ~450ms

 |
| Malformed JSON Input | `portfolio_agent` / `goal_agent` → Error Handler | < 5ms

 |

---

## 7. Deployment & Verification Runbook

```bash
# 1. Environment Activation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Automated Test Suite Execution
pytest tests/ -v --cov=src

# 3. Launching Streamlit Web App
streamlit run src/web_app/app.py

```

```

```