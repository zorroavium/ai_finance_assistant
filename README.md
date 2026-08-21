# AI Finance Assistant — Production Multi-Agent RAG System

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
   - `finance_qa`: Answers core investing concepts grounded in 54+ curated KB articles with citation IDs (`[ref: INV-001]`)[cite: 26].
   - `portfolio_agent`: Deterministically calculates portfolio total values, asset allocation percentages, and weighted expense ratios[cite: 26].
   - `market_agent`: Retrieves real-time stock/ETF metrics via `yfinance` with a 30-minute in-memory TTL cache and fallback resilience[cite: 26].
   - `goal_agent`: Models compound interest future values and milestones across custom time horizons[cite: 26].
   - `tax_agent`: Explains rules for tax-advantaged accounts (Roth IRA, Traditional IRA, 401(k), HSA) and capital gains[cite: 26].
   - `news_agent`: Synthesizes macroeconomic shifts and sentiment for beginner investors[cite: 26].
2. **Dynamic Task Decomposition**: Composite user queries (e.g., *"What is VOO price and how is it taxed in a Roth IRA?"*) are broken into distinct tasks executed in parallel[cite: 26].
3. **Calibrated Confidence Scoring**: RAG retrieval evaluates cosine similarity against a calibrated threshold (`0.35` on `text-embedding-3-small`), preventing hallucinations when topics fall outside the knowledge base[cite: 26].
4. **State Isolation**: Agent scratchpads use a custom reducer (`reset_or_add`) to prevent cross-talk and reset temporary state between user turns[cite: 26].
5. **Interactive Multi-Tab Dashboard**: Built with Streamlit, providing Chat, Portfolio Analysis, Live Market Lookup, and Compound Growth visualizers[cite: 26].

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
│   │   ├── vector_store.py        # In-memory vector store & cosine similarity
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
│   └── web_app/
│       └── app.py                 # Multi-tab Streamlit web application
└── tests/
    ├── test_tools.py              # Unit tests for portfolio and goal calculators
    ├── test_rag.py                # Unit tests for vector search & confidence scoring
    ├── test_market_tools.py       # Unit tests for live pricing & cache TTL
    ├── test_agents.py             # Agent execution & tool calling tests
    ├── test_routing.py            # Orchestrator classification & decomposition tests
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
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Environment configuration
cp .env.example .env
# Add your OPENAI_API_KEY inside the .env file

```

---

## 🧪 Running the Test Suite

Execute the full test suite with coverage reporting:

```bash
pytest tests/ -v

```

---

## 🖥️ Running the Web Application

Launch the Streamlit multi-tab interface:

```bash
streamlit run src/web_app/app.py

```

---

## ⚖️ Regulatory & Educational Disclaimer

The AI Finance Assistant is designed strictly for educational and informational purposes. It does not offer personalized investment, financial, legal, or tax advice. Market quotes are retrieved with caching and may be delayed.
