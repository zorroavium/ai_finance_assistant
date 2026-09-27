# Finnie - Personal Finance Agent — Multi-Agent Financial Education App

> **Democratizing Financial Literacy Through Intelligent Conversational AI**  
> *Applied Agentic AI Capstone Project*

---

## 🏛️ System Architecture


```

```
                       [ User Query ]
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Orchestrator Node   │
                 │  (Intent & Task Pydantic)
                 └───────────┬───────────┘
                             │
             ┌───────────────┴───────────────┐
             │  Parallel Dynamic Fan-Out    │
             │      (LangGraph Send API)     │
             ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│     RAG Specialists      │    │    Analytical Workers    │
│ ------------------------ │    │ ------------------------ │
│ • Finance Q&A Agent      │    │ • Portfolio Analysis     │
│ • Tax Education Agent    │    │ • Market Data (yfinance) │
│ • Financial Glossary     │    │ • Goal Planning & Proj   │
│ (Cosine Sim + Confidence)│    │ • News Synthesizer       │
└────────────┬─────────────┘    └────────────┬─────────────┘
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Synthesizer Node    │
                 │ (Merge, Citations,    │
                 │ Compliance Disclaimer)│
                 └───────────┬───────────┘
                             │
                             ▼
                 [ Structured User Output ]

```

```

---

## 🚀 Key Features

1. **6 Specialized Domain Agents**:
    - `finance_qa`: Answers investing concepts grounded in the curated knowledge base.
    - `portfolio_agent`: Calculates allocation, diversification, concentration, risk alignment, and expense-ratio metrics.
    - `market_agent`: Retrieves live stock/ETF data through `yfinance`, including dynamic company-name lookup, caching, retries, and fallbacks.
    - `goal_agent`: Models compound-growth scenarios and financial milestones.
    - `tax_agent`: Explains tax-advantaged accounts and capital-gains concepts using the tax knowledge category.
    - `news_agent`: Adds optional Tavily-backed recent news context and explains market implications.
2. **Dynamic Task Decomposition**: Composite questions are split into specialist tasks and dispatched in parallel through LangGraph `Send`.
3. **Grounded Retrieval**: The RAG layer returns top-K documents, confidence scores, categories, and reference IDs. Finance Q&A and Tax outputs validate generated reference IDs against retrieved documents.
4. **Context Preservation**: User profile, portfolio holdings, and recent conversation history are passed into specialist prompts.
5. **Interactive Streamlit Dashboard**: The app contains Chat, Portfolio, Markets, Goals, and Knowledge tabs, plus sidebar API configuration, investor profile, portfolio management, chat sessions, and quick actions.

---

## 📂 Project Structure

```text
ai_finance_assistant/
├── config.yaml                    # System configuration, models, thresholds, APIs
├── requirements.txt               # Locked production dependencies
├── README.md                      # Architecture, setup, API & troubleshooting guide
├── TECHNICAL_DESIGN.md            # In-depth architectural & state specifications
├── src/
│   ├── core/
│   │   ├── config.py              # YAML and environment variable loader
│   │   └── logger.py              # Color-coded ANSI terminal tracing
│   ├── data/
│   │   ├── kb/                    # 54+ Curated financial knowledge base articles
│   │   │   ├── investing_basics.json
│   │   │   ├── portfolio_mgmt.json
│   │   │   ├── tax_accounts.json
│   │   │   ├── risk_planning.json
│   │   │   └── financial_glossary.json
│   │   └── sample_portfolios.json # Starter templates (Balanced, Conservative)
│   ├── rag/
│   │   ├── vector_store.py        # OpenAI embeddings, in-memory store & cosine similarity
│   │   └── retriever.py           # RAG search tool with confidence scoring
│   ├── tools/
│   │   ├── market_tools.py        # Real-time yfinance quotes with 30-min TTL caching
│   │   ├── portfolio_tools.py     # Deterministic asset allocation & fee calculators
│   │   └── goal_tools.py          # Compound interest & growth projection math
│   ├── agents/
│   │   ├── orchestrator.py        # Structured classification & Send API dispatcher
│   │   ├── finance_qa.py          # Grounded financial concepts specialist
│   │   ├── portfolio_agent.py     # Portfolio health & diversification analyzer
│   │   ├── market_agent.py        # Live market intelligence specialist
│   │   ├── goal_agent.py          # Financial milestone projection specialist
│   │   ├── tax_agent.py           # Tax account rules & education specialist
│   │   ├── news_agent.py          # Macroeconomic sentiment synthesizer
│   │   ├── general_chat.py        # Conversational greetings & help concierge
│   │   └── synthesizer.py         # Multi-agent aggregator & compliance reviewer
│   ├── workflow/
│   │   ├── state.py               # Central LangGraph state, Pydantic task schemas
│   │   └── graph.py               # StateGraph compilation & checkpointer
│   ├── utils/
│   │   └── news_search.py         # Optional Tavily news client with caching
│   └── web_app/
│       └── app.py                 # Five-tab Streamlit web application
├── mcp_server.py                  # Optional FastMCP quote/portfolio tools
└── tests/
    ├── test_tools.py              # Unit tests for portfolio and goal calculators
    ├── test_rag.py                # Unit tests for vector search & confidence scoring
    ├── test_market_tools.py       # Unit tests for live pricing & cache TTL
    ├── test_agents.py             # Agent execution & tool calling tests
    ├── test_router_agent.py       # Router fallback and empty-router tests
    ├── test_edge_cases.py         # Edge cases & malformed input handling
    └── test_workflow.py           # End-to-end multi-agent graph integration tests

```

---

## 🛠️ Setup & Installation

### 1. Prerequisites

* Python 3.10+ (tested on Python 3.11 / 3.12 / 3.14)


* OpenAI API Key



### 2. Environment Setup

```bash
# 1. Clone or navigate to the repository
cd ai_finance_assistant

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Environment configuration
cp .env.example .env
# Add your OPENAI_API_KEY inside the .env file.
# Optional: TAVILY_API_KEY enables recent news search.

```

---

## 🧪 Running the Test Suite

Execute the full test suite with coverage reporting:

```bash
python -m pytest -q --cov=src --cov-report=term-missing

# Latest verified baseline: 38 tests passing, 81% coverage.

```

---

## 🖥️ Running the Web Application

Launch the Streamlit multi-tab interface:

```bash
streamlit run src/web_app/app.py

```

---

## ⚖️ Regulatory & Educational Disclaimer

The Finnie - Personal Finance Agent is designed strictly for educational and informational purposes. It does not offer personalized investment, financial, legal, or tax advice. Market quotes are retrieved with caching and may be delayed.

## Optional MCP Server

The optional FastMCP server exposes `get_quote(ticker)` and `evaluate_portfolio(holdings_json, user_risk_appetite)`. Both reuse the production market and portfolio tools.

Run it locally with:

```bash
python mcp_server.py
```

## Known Limitations

- The active RAG implementation uses an in-memory cosine-similarity store; it is not FAISS-backed.
- Recent news search requires `TAVILY_API_KEY`; without it, the news agent uses model knowledge only.
- Market data depends on external Yahoo Finance availability and may use fallback benchmark data.
- MCP currently exposes quote and portfolio tools only and has limited automated coverage.
