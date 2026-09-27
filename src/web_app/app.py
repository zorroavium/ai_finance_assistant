"""
Finnie - Personal Finance Agent — Interactive Multi-Tab Streamlit Dashboard.
Features: Multi-agent Chat, Portfolio Analytics, Live Market Tickers & Trends, and Goal Projections.
"""

import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.core.config import BASE_DIR, set_runtime_api_keys, get_openai_api_key
from src.rag.retriever import search_financial_kb
from src.tools.goal_tools import calculate_savings_plan, project_goal_growth
from src.tools.market_tools import clear_market_cache, get_market_history, get_market_quote
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.workflow.graph import build_finance_graph

st.set_page_config(
    page_title="Finnie - Personal Finance Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 1100px;
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
      }
      .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 500;
        border-radius: 6px 6px 0 0;
      }
      [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="⏳ Initializing Knowledge Base and Agents…")
def get_graph():
    return build_finance_graph()


graph = get_graph()


def _new_chat() -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.conversations[chat_id] = {
        "title": "New Session",
        "messages": [],
        "thread_id": chat_id,
        "turns": [],
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    _new_chat()
if "active_portfolio" not in st.session_state:
    st.session_state.active_portfolio = {}
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "experience_level": "Beginner",
        "goals": [],
    }


def _current() -> dict:
    return st.session_state.conversations[st.session_state.current_chat_id]


def _invoke_graph(prompt: str) -> dict:
    """Invoke the same contextual workflow used by the Chat tab."""
    current = _current()
    history = list(current["messages"]) + [{"role": "user", "content": prompt}]
    return graph.invoke(
        {
            "user_query": prompt,
            "messages": [
                HumanMessage(content=item["content"])
                if item["role"] == "user"
                else AIMessage(content=item["content"])
                for item in history
            ],
            "user_profile": st.session_state.user_profile,
            "portfolio": st.session_state.active_portfolio,
            "context": {
                "user_profile": st.session_state.user_profile,
                "portfolio": st.session_state.active_portfolio,
                "conversation_history": [
                    f"{item['role']}: {item['content']}" for item in history
                ],
            },
            "conversation_history": [
                f"{item['role']}: {item['content']}" for item in history
            ],
            "tasks": [],
            "requires_synthesis": False,
            "agent_results": None,
            "qa_messages": None,
            "portfolio_messages": None,
            "market_messages": None,
            "goal_messages": None,
            "news_messages": None,
            "tax_messages": None,
        },
        config={"configurable": {"thread_id": current["thread_id"]}},
    )


@st.cache_data(ttl=300)
def _get_market_indices() -> list[dict]:
    indices = [("SPY", "S&P 500"), ("QQQ", "NASDAQ 100"), ("DIA", "Dow Jones"), ("VTI", "Total Market")]
    data = []
    for ticker, label in indices:
        try:
            quote = json.loads(get_market_quote.invoke({"ticker": ticker}))
            data.append({"ticker": ticker, "label": label, **quote})
        except Exception as exc:
            data.append({"ticker": ticker, "label": label, "error": str(exc)})
    return data


def _clear_current_chat() -> None:
    current = _current()
    current["messages"] = []
    current["turns"] = []
    current["title"] = "New Session"


def _render_api_configuration() -> None:
    with st.expander("🔑 API Configuration", expanded=False):
        st.caption("Keys are kept in runtime session state and are not written to project files.")
        openai_key = st.text_input(
            "OpenAI API key",
            value=st.session_state.get("openai_api_key_input", ""),
            type="password",
            key="openai_api_key_input",
        )
        tavily_key = st.text_input(
            "Tavily API key (optional news search)",
            value=st.session_state.get("tavily_api_key_input", ""),
            type="password",
            key="tavily_api_key_input",
        )
        if st.button("Update API keys", use_container_width=True):
            set_runtime_api_keys(openai_key, tavily_key)
            get_graph.clear()
            st.success("Runtime API configuration updated.")
            st.rerun()

        st.caption(f"OpenAI: {'configured' if get_openai_api_key() else 'missing'}")
        tavily_configured = bool(os.getenv("TAVILY_API_KEY"))
        st.caption(f"Tavily news search: {'configured' if tavily_configured else 'not configured'}")
        st.caption("Market data: yfinance with cached/offline fallback")


def _render_sidebar_portfolio() -> None:
    """Manage the same portfolio object consumed by portfolio and chat agents."""
    st.markdown("##### Portfolio")
    option = st.radio(
        "Portfolio options",
        ["Load sample portfolio", "Enter custom portfolio"],
        key="sidebar_portfolio_option",
        label_visibility="collapsed",
    )

    if option == "Load sample portfolio":
        sample_file = BASE_DIR / "src/data/sample_portfolios.json"
        sample_data = json.loads(sample_file.read_text()) if sample_file.exists() else {}
        portfolio_type = st.selectbox(
            "Sample portfolio type",
            ["Balanced", "Conservative"],
            key="sidebar_portfolio_type",
        )
        if st.button("📊 Load sample portfolio", use_container_width=True):
            sample_key = "balanced" if portfolio_type == "Balanced" else "conservative"
            st.session_state.active_portfolio = {
                "holdings": sample_data.get(sample_key, {}).get("holdings", []),
                "risk_appetite": "Moderate" if sample_key == "balanced" else "Conservative",
            }
            st.success(f"{portfolio_type} portfolio loaded.")
            st.rerun()
    else:
        if "sidebar_custom_holdings" not in st.session_state:
            st.session_state.sidebar_custom_holdings = []

        with st.expander("📝 Enter portfolio details", expanded=True):
            input_columns = st.columns(2)
            with input_columns[0]:
                symbol = st.text_input("Symbol", key="sidebar_new_symbol", placeholder="AAPL")
                quantity = st.number_input("Quantity", min_value=0.01, value=10.0, step=0.01, key="sidebar_new_quantity")
            with input_columns[1]:
                price = st.number_input("Current price", min_value=0.01, value=100.0, step=0.01, key="sidebar_new_price")
                category = st.text_input("Category", value="US Equities", key="sidebar_new_category")

            if st.button("➕ Add holding", use_container_width=True):
                if not symbol.strip():
                    st.error("Symbol is required.")
                else:
                    st.session_state.sidebar_custom_holdings.append({
                        "symbol": symbol.upper().strip(),
                        "shares": float(quantity),
                        "price": float(price),
                        "category": category.strip() or "Other",
                        "expense_ratio": 0.0,
                    })
                    st.rerun()

            custom_holdings = st.session_state.sidebar_custom_holdings
            if custom_holdings:
                st.dataframe(pd.DataFrame(custom_holdings), hide_index=True, use_container_width=True)
                custom_risk = st.select_slider(
                    "Portfolio risk level",
                    ["Conservative", "Moderate", "Aggressive"],
                    value="Moderate",
                    key="sidebar_custom_risk",
                )
                save_columns = st.columns(2)
                with save_columns[0]:
                    if st.button("💾 Save portfolio", type="primary", use_container_width=True):
                        st.session_state.active_portfolio = {
                            "holdings": list(custom_holdings),
                            "risk_appetite": custom_risk,
                        }
                        st.session_state.sidebar_custom_holdings = []
                        st.success("Portfolio saved.")
                        st.rerun()
                with save_columns[1]:
                    if st.button("🔄 Clear holdings", use_container_width=True):
                        st.session_state.sidebar_custom_holdings = []
                        st.rerun()

    if st.session_state.active_portfolio:
        holdings = st.session_state.active_portfolio.get("holdings", [])
        st.caption(f"Active portfolio: {len(holdings)} holding(s)")
        if st.button("🗑️ Clear active portfolio", use_container_width=True):
            st.session_state.active_portfolio = {}
            st.session_state.portfolio_analysis = None
            st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Finnie - Personal Finance Agent")
    st.caption("Democratizing Financial Literacy via Multi-Agent AI")
    st.divider()

    _render_api_configuration()
    st.divider()

    if st.button("➕ New Chat Session", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    st.markdown("##### Chat History")
    for chat_id, chat in reversed(list(st.session_state.conversations.items())):
        is_current = chat_id == st.session_state.current_chat_id
        cols = st.columns([5, 1])
        with cols[0]:
            label = chat["title"] or "New Session"
            prefix = "🟢 " if is_current else "💬 "
            if st.button(f"{prefix}{label[:25]}", key=f"btn-{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with cols[1]:
            if len(st.session_state.conversations) > 1:
                if st.button("✕", key=f"del-{chat_id}", help="Delete chat"):
                    del st.session_state.conversations[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = next(iter(st.session_state.conversations))
                    st.rerun()

    st.divider()
    _render_sidebar_portfolio()
    st.divider()
    st.markdown("##### Investor Profile")
    st.session_state.user_profile["experience_level"] = st.selectbox(
        "Experience level",
        ["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(
            st.session_state.user_profile["experience_level"]
        ),
    )
    st.session_state.user_profile["goals"] = st.multiselect(
        "Financial goals",
        ["Retirement", "Emergency fund", "Home purchase", "Education", "General wealth"],
        default=st.session_state.user_profile["goals"],
    )
    st.divider()
    st.markdown("##### Quick Actions")
    if st.button("🔄 Clear current conversation", use_container_width=True):
        _clear_current_chat()
        st.rerun()
    if st.button("📈 Refresh market data", use_container_width=True):
        clear_market_cache()
        st.rerun()
    st.session_state.show_confidence = st.checkbox(
        "Show routing details", value=st.session_state.get("show_confidence", True)
    )
    st.session_state.show_sources = st.checkbox(
        "Show knowledge sources", value=st.session_state.get("show_sources", False)
    )
    st.session_state.show_suggestions = st.checkbox(
        "Show suggestions", value=st.session_state.get("show_suggestions", True)
    )
    st.divider()
    st.markdown("##### Quick Tickers")
    st.caption("SPY • VOO • VTI • BND • QQQ • AAPL • MSFT")
    st.divider()
    st.caption("⚡ Powered by LangGraph, OpenAI & yfinance")


st.title("Finnie - Personal Finance Agent")
st.caption("Multi-agent financial guidance, real-time analytics, and goal modeling.")

tabs = st.tabs([
    "💬 Chat Assistant",
    "📊 Portfolio Analysis",
    "📈 Market Overview",
    "🎯 Goal Planning",
    "📚 Knowledge",
])

# ── TAB 1: CHAT ASSISTANT ─────────────────────────────────────────────
with tabs[0]:
    current = _current()

    def _render_routing_meta(meta: dict):
        if not st.session_state.get("show_confidence", True) or not meta or not meta.get("agents"):
            return
        with st.expander("🔍 Routing & Execution Details", expanded=False):
            st.markdown(f"**Agents Activated:** `{', '.join(meta['agents'])}`")
            for t in meta.get("tasks", []):
                st.markdown(f"- **Agent:** `{t['agent']}` | **Focus:** {t['focus']} | **Query:** *{t['query']}*")

    turn_idx = 0
    for msg in current["messages"]:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if turn_idx < len(current["turns"]):
                    _render_routing_meta(current["turns"][turn_idx])
                turn_idx += 1

    pending_prompt = st.session_state.pop("pending_chat_question", None)
    prompt = pending_prompt or st.chat_input("Ask a financial question (e.g., 'What is DCA and what is the price of VTI?')...")
    if prompt:
        current["messages"].append({"role": "user", "content": prompt})
        if current["title"] == "New Session":
            current["title"] = prompt[:35]

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Specialist agents collaborating..."):
                try:
                    result = graph.invoke(
                        {
                            "user_query": prompt,
                            "messages": [
                                HumanMessage(content=message["content"])
                                if message["role"] == "user"
                                else AIMessage(content=message["content"])
                                for message in current["messages"]
                            ],
                            "user_profile": st.session_state.user_profile,
                            "portfolio": st.session_state.active_portfolio,
                            "context": {
                                "user_profile": st.session_state.user_profile,
                                "portfolio": st.session_state.active_portfolio,
                                "conversation_history": [
                                    f"{message['role']}: {message['content']}"
                                    for message in current["messages"]
                                ],
                            },
                            "conversation_history": [
                                f"{message['role']}: {message['content']}"
                                for message in current["messages"]
                            ],
                            "tasks": [],
                            "requires_synthesis": False,
                            "agent_results": None,
                            "qa_messages": None,
                            "portfolio_messages": None,
                            "market_messages": None,
                            "goal_messages": None,
                            "news_messages": None,
                            "tax_messages": None,
                        },
                        config={"configurable": {"thread_id": current["thread_id"]}},
                    )
                    answer = result.get("final_answer", "Could not generate response.")
                    tasks = result.get("tasks", []) or []
                    meta = {
                        "agents": sorted({
                            (t.agent if hasattr(t, "agent") else t.get("agent", "?"))
                            for t in tasks
                        }),
                        "tasks": [
                            {
                                "agent": t.agent if hasattr(t, "agent") else t.get("agent", "?"),
                                "focus": t.focus if hasattr(t, "focus") else t.get("focus", ""),
                                "query": t.query if hasattr(t, "query") else t.get("query", ""),
                            }
                            for t in tasks
                        ],
                    }
                except Exception as e:
                    answer = f"⚠️ System Error: `{e}`"
                    meta = {"agents": [], "tasks": []}

            st.markdown(answer)
            _render_routing_meta(meta)

        current["messages"].append({"role": "assistant", "content": answer})
        current["turns"].append(meta)
        st.rerun()

    st.markdown("**Try these questions:**")
    example_columns = st.columns(3)
    examples = [
        "What are ETFs?",
        "How do I diversify my portfolio?",
        "Explain compound interest",
        "Should I invest in stocks or bonds?",
        "How large should an emergency fund be?",
        "How do 401(k)s work?",
    ]
    for index, example in enumerate(examples):
        with example_columns[index % 3]:
            if st.button(example, key=f"example_question_{index}", use_container_width=True):
                st.session_state.pending_chat_question = example
                st.rerun()


# ── TAB 2: PORTFOLIO ANALYSIS ─────────────────────────────────────────
with tabs[1]:
    st.subheader("Portfolio Analysis")

    sample_file = BASE_DIR / "src/data/sample_portfolios.json"
    sample_data = json.loads(sample_file.read_text()) if sample_file.exists() else {}

    if not st.session_state.active_portfolio:
        st.info("No portfolio loaded. Load a sample, enter holdings manually, or import a CSV.")
        setup_cols = st.columns(3)
        with setup_cols[0]:
            template = st.selectbox("Sample portfolio", ["Balanced", "Conservative"])
            if st.button("📊 Load sample portfolio", use_container_width=True):
                key = "balanced" if template == "Balanced" else "conservative"
                st.session_state.active_portfolio = {
                    "holdings": sample_data.get(key, {}).get("holdings", []),
                    "risk_appetite": "Moderate" if key == "balanced" else "Conservative",
                }
                st.rerun()
        with setup_cols[1]:
            st.markdown("**Import CSV**")
            uploaded = st.file_uploader("CSV holdings", type=["csv"], label_visibility="collapsed")
            if uploaded is not None and st.button("📤 Import uploaded CSV", use_container_width=True):
                imported = pd.read_csv(uploaded).to_dict("records")
                st.session_state.active_portfolio = {"holdings": imported, "risk_appetite": "Moderate"}
                st.rerun()
        with setup_cols[2]:
            st.markdown("**Manual entry**")
            manual_json = st.text_area("Holdings JSON", value="[]", height=120)
            if st.button("✏️ Use manual holdings", use_container_width=True):
                try:
                    imported = json.loads(manual_json)
                    if not isinstance(imported, list):
                        raise ValueError("Holdings must be a JSON list")
                    st.session_state.active_portfolio = {"holdings": imported, "risk_appetite": "Moderate"}
                    st.rerun()
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(f"Invalid holdings: {exc}")
    else:
        portfolio = st.session_state.active_portfolio
        holdings = portfolio.get("holdings", [])
        risk_appetite = st.selectbox(
            "Target risk appetite",
            ["Conservative", "Moderate", "Aggressive"],
            index=["Conservative", "Moderate", "Aggressive"].index(portfolio.get("risk_appetite", "Moderate")),
        )
        portfolio["risk_appetite"] = risk_appetite

        if st.button("🗑️ Clear portfolio"):
            st.session_state.active_portfolio = {}
            st.session_state.portfolio_analysis = None
            st.rerun()

        st.markdown("##### Holdings breakdown")
        st.dataframe(pd.DataFrame(holdings), use_container_width=True, hide_index=True)
        metrics = json.loads(calculate_portfolio_metrics.invoke({
            "holdings_json": json.dumps(holdings),
            "user_risk_appetite": risk_appetite,
        }))
        if metrics.get("status") == "success" or "total_value" in metrics:
            columns = st.columns(4)
            columns[0].metric("Total value", f"${metrics['total_value']:,.2f}")
            columns[1].metric("Holdings", len(holdings))
            columns[2].metric("Diversification", f"{metrics['diversification_score']} / 100")
            columns[3].metric("Risk status", metrics["risk_alignment"])
            st.info(metrics.get("recommendation", ""))
            if metrics.get("concentrated_positions"):
                st.warning("Concentration risk: " + ", ".join(metrics["concentrated_positions"]))

            allocation = pd.DataFrame(
                list(metrics["allocation_percentages"].items()),
                columns=["Category", "Percentage"],
            )
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.plotly_chart(px.pie(allocation, names="Category", values="Percentage", hole=0.45), use_container_width=True)
            with chart_cols[1]:
                st.plotly_chart(px.bar(allocation, x="Category", y="Percentage", color="Category"), use_container_width=True)

        if st.button("🔍 Analyze portfolio with specialist agent", type="primary", use_container_width=True):
            try:
                result = _invoke_graph(
                    "Analyze my portfolio using the loaded holdings and explain allocation, diversification, fees, concentration risks, and risk alignment."
                )
                st.session_state.portfolio_analysis = result.get("final_answer", "No analysis returned.")
            except Exception as exc:
                st.error(f"Portfolio analysis failed: {exc}")
        if st.session_state.get("portfolio_analysis"):
            st.markdown("##### Specialist analysis")
            st.markdown(st.session_state.portfolio_analysis)


# ── TAB 3: MARKET OVERVIEW ────────────────────────────────────────────
with tabs[2]:
    st.subheader("Market Overview")
    st.markdown("##### Major indices")
    index_columns = st.columns(4)
    for index, data in enumerate(_get_market_indices()):
        with index_columns[index]:
            if data.get("current_price") is not None:
                st.metric(
                    data["label"],
                    f"${data['current_price']:,.2f}",
                    data.get("change_percent", "N/A"),
                    delta_color="normal" if not str(data.get("change_percent", "-")).startswith("-") else "inverse",
                )
            else:
                st.metric(data["label"], "N/A", "Data unavailable")

    st.markdown("##### Stock / ETF lookup")
    with st.form("market_lookup_form", clear_on_submit=False):
        col_search, col_button = st.columns([3, 1])
        with col_search:
            ticker_input = st.text_input(
                "Enter ticker or company name",
                value="VOO",
                placeholder="AAPL, MSFT, Microsoft, or VOO",
            ).strip()
        with col_button:
            lookup = st.form_submit_button("Look up", type="primary", use_container_width=True)

    if lookup:
        st.session_state.last_ticker = ticker_input

    if st.session_state.get("last_ticker") == ticker_input:
        quote_str = get_market_quote.invoke({"ticker": ticker_input})
        quote_data = json.loads(quote_str)

        if "current_price" in quote_data:
            fresh_badge = quote_data.get("freshness", "LIVE")
            ts = quote_data.get("timestamp_utc", "")
            st.caption(f"Status: `{fresh_badge}` • Timestamp: `{ts}`")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${quote_data['current_price']:.2f}")
            m2.metric("Daily Change", quote_data.get("change_percent", "N/A"))
            m3.metric("52-Week High", f"${quote_data.get('fifty_two_week_high', 0) or 0:.2f}")
            m4.metric("52-Week Low", f"${quote_data.get('fifty_two_week_low', 0) or 0:.2f}")

            display_ticker = quote_data.get("ticker", ticker_input)
            st.caption(f"{quote_data.get('name', display_ticker)} • {quote_data.get('currency', 'USD')}")
            hist_str = get_market_history.invoke({"ticker": display_ticker, "period": "6mo"})
            hist_data = json.loads(hist_str)

            if "close_prices" in hist_data:
                st.markdown(f"##### 6-Month Price Trend ({hist_data.get('period_return_pct', 0):+.2f}% Return)")
                df_hist = pd.DataFrame({
                    "Date": pd.to_datetime(hist_data["dates"]),
                    "Close": hist_data["close_prices"],
                })
                fig_trend = px.line(
                    df_hist,
                    x="Date",
                    y="Close",
                    title=f"{display_ticker} 6-Month Historical Performance",
                    template="plotly_white",
                )
                fig_trend.update_traces(line_color="#2E7D32" if hist_data.get("period_return_pct", 0) >= 0 else "#C62828")
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.error(quote_data.get("error", "Could not fetch ticker details."))
            suggested_ticker = quote_data.get("suggested_ticker")
            if suggested_ticker:
                st.info(f"Did you mean **{suggested_ticker}**?")
                if st.button(f"Look up {suggested_ticker}", key="suggested_ticker_lookup"):
                    st.session_state.last_ticker = suggested_ticker
                    st.rerun()
            else:
                st.caption("Check the symbol and try again. Examples: AAPL, MSFT, SPY, VOO, QQQ.")


# ── TAB 4: GOAL PLANNING ──────────────────────────────────────────────
with tabs[3]:
    st.subheader("Financial Goal Planning")
    goal_tab, projection_tab = st.tabs(["Savings goal calculator", "Wealth projection"])

    with goal_tab:
        goal_cols = st.columns(2)
        with goal_cols[0]:
            goal_amount = st.number_input("Goal amount ($)", min_value=1000.0, value=100000.0, step=1000.0)
            goal_years = st.number_input("Time horizon (years)", min_value=1, max_value=50, value=10, step=1)
        with goal_cols[1]:
            current_savings = st.number_input("Current savings ($)", min_value=0.0, value=10000.0, step=1000.0)
            annual_return = st.slider("Expected annual return (%)", 0.0, 15.0, 7.0, 0.5) / 100

        if st.button("Calculate savings plan", type="primary", use_container_width=True):
            plan = calculate_savings_plan(goal_amount, int(goal_years), current_savings, annual_return)
            st.session_state.goal_plan = plan

        if st.session_state.get("goal_plan"):
            plan = st.session_state.goal_plan
            metrics = st.columns(3)
            metrics[0].metric("Monthly savings required", f"${plan['monthly_contribution_required']:,.2f}")
            metrics[1].metric("Interest earned", f"${plan['interest_earned']:,.2f}")
            metrics[2].metric("Final balance", f"${plan['final_balance']:,.2f}")
            months = list(range(len(plan["monthly_balances"])))
            st.plotly_chart(
                go.Figure(go.Scatter(x=months, y=plan["monthly_balances"], name="Projected balance")),
                use_container_width=True,
            )
            if plan["feasible"]:
                st.success("This target is within the calculator's contribution feasibility threshold.")
            else:
                st.warning("The required monthly contribution is high. Consider a longer timeline or smaller target.")

    with projection_tab:
        projection_cols = st.columns(3)
        with projection_cols[0]:
            init_deposit = st.number_input("Initial investment ($)", value=5000.0, step=500.0)
            monthly_contrib = st.number_input("Monthly contribution ($)", value=500.0, step=50.0)
        with projection_cols[1]:
            risk_profile = st.selectbox("Risk profile", ["Conservative", "Moderate", "Aggressive"], index=1)
            years = st.slider("Investment horizon (years)", 1, 40, 20, 1)
        with projection_cols[2]:
            custom_rate = st.checkbox("Custom return override")
            override_rate = st.number_input("Custom return rate (%)", value=7.5, step=0.5) if custom_rate else None

        proj_data = json.loads(project_goal_growth.invoke({
            "initial_amount": init_deposit,
            "monthly_contribution": monthly_contrib,
            "years": years,
            "risk_profile": risk_profile,
            "annual_return_pct": override_rate,
        }))
        projection_metrics = st.columns(4)
        projection_metrics[0].metric("Expected wealth", f"${proj_data['projected_median_value']:,.2f}")
        projection_metrics[1].metric("Principal contributed", f"${proj_data['total_contributed']:,.2f}")
        projection_metrics[2].metric("Pessimistic", f"${proj_data['pessimistic_band_value']:,.2f}")
        projection_metrics[3].metric("Optimistic", f"${proj_data['optimistic_band_value']:,.2f}")
        st.info(f"Suggested asset mix: {proj_data['recommended_asset_mix']} at approximately {proj_data['assumed_annual_return']}.")


# ── TAB 5: KNOWLEDGE BASE ─────────────────────────────────────────────
with tabs[4]:
    st.subheader("Financial Knowledge Base")
    st.caption("Search the curated investing and tax education articles used by the finance agents.")

    knowledge_query = st.text_input(
        "Search financial topics",
        placeholder="e.g. diversification, compound interest, Roth IRA",
    )
    knowledge_category = st.selectbox(
        "Category",
        ["All", "Investing Basics", "Tax Accounts", "Portfolio Management", "Risk & Planning"],
    )

    if st.button("Search knowledge base", type="primary") and knowledge_query:
        try:
            result = json.loads(search_financial_kb.invoke({
                "query": knowledge_query,
                "category": None if knowledge_category == "All" else knowledge_category,
            }))
            if result.get("results"):
                st.success(f"Found {len(result['results'])} relevant articles.")
                for index, document in enumerate(result["results"], start=1):
                    with st.expander(f"{index}. {document.get('title', 'Untitled')}"):
                        st.write(document.get("content", ""))
                        st.caption(
                            f"{document.get('category', 'General')} • "
                            f"Reference: {document.get('id', 'N/A')} • "
                            f"Relevance: {document.get('relevance_score', 0):.3f}"
                        )
            else:
                st.warning(result.get("message", "No matching articles found."))
                if st.button("Ask this question in Chat", key="knowledge_to_chat"):
                    st.session_state.pending_chat_question = knowledge_query
                    st.rerun()
        except Exception as exc:
            st.error(f"Knowledge search unavailable: {exc}")

    st.markdown("##### Popular Topics")
    topic_columns = st.columns(3)
    for index, topic in enumerate([
        "Investing basics",
        "Diversification",
        "Tax-advantaged accounts",
    ]):
        with topic_columns[index]:
            st.info(topic)