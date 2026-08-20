"""
AI Finance Assistant — Interactive Multi-Tab Streamlit Dashboard.
Features: Multi-agent Chat, Portfolio Analytics, Live Market Tickers, and Goal Projections.
"""

import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.messages import HumanMessage

# Ensure project root is in sys.path
_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.core.config import BASE_DIR
from src.tools.goal_tools import project_goal_growth
from src.tools.market_tools import get_market_quote
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.workflow.graph import build_finance_graph

# ── Page Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1100px; }
      [data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 0.5rem; }
      .metric-card {
        background-color: rgba(127,127,127,0.06);
        border: 1px solid rgba(127,127,127,0.15);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
      }
      .disclaimer-box {
        font-size: 0.8rem;
        color: #888;
        border-top: 1px solid #ddd;
        padding-top: 0.5rem;
        margin-top: 5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached Graph Singleton ───────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Initializing Financial Knowledge Bases and Agents…")
def get_graph():
    return build_finance_graph()


graph = get_graph()


# ── Session State Management ──────────────────────────────────────────
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


def _current() -> dict:
    return st.session_state.conversations[st.session_state.current_chat_id]


# ── Sidebar Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 AI Finance Assistant")
    st.caption("Democratizing Financial Literacy via Multi-Agent AI")
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
    st.markdown("##### Quick Tickers")
    st.caption("SPY • VOO • VTI • BND • QQQ • AAPL • MSFT")
    st.divider()
    st.caption("⚡ Powered by LangGraph, OpenAI & yfinance")


# ── Top Level Tabs ────────────────────────────────────────────────────
tabs = st.tabs([
    "💬 Chat Assistant",
    "📊 Portfolio Analysis",
    "📈 Market Overview",
    "🎯 Goal Planning",
])

# ──────────────────────────────────────────────────────────────────────
# TAB 1: CHAT ASSISTANT
# ──────────────────────────────────────────────────────────────────────
with tabs[0]:
    current = _current()
    st.subheader("Conversational Financial Guide")
    st.caption("Ask about investment basics, retirement accounts, portfolio strategies, or live stock prices.")

    def _render_routing_meta(meta: dict):
        if not meta or not meta.get("agents"):
            return
        with st.expander("🔍 Routing & Execution Details", expanded=False):
            st.markdown(f"**Agents Activated:** `{', '.join(meta['agents'])}`")
            for t in meta.get("tasks", []):
                st.markdown(f"- **Agent:** `{t['agent']}` | **Focus:** {t['focus']} | **Query:** *{t['query']}*")

    # Render Conversation History
    turn_idx = 0
    for msg in current["messages"]:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if turn_idx < len(current["turns"]):
                    _render_routing_meta(current["turns"][turn_idx])
                turn_idx += 1

    prompt = st.chat_input("Ask a financial question (e.g., 'What is DCA and what is the price of VTI?')...")
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
                            "messages": [HumanMessage(content=prompt)],
                            "user_profile": {},
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


# ──────────────────────────────────────────────────────────────────────
# TAB 2: PORTFOLIO ANALYSIS
# ──────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Portfolio Allocation & Health Check")

    # Load Sample Portfolios
    sample_file = BASE_DIR / "src/data/sample_portfolios.json"
    sample_data = {}
    if sample_file.exists():
        with open(sample_file, "r") as f:
            sample_data = json.load(f)

    portfolio_mode = st.selectbox(
        "Choose Portfolio Template or Custom Input:",
        ["Moderate Growth (60/40)", "Conservative Income", "Custom JSON Input"],
    )

    if portfolio_mode == "Moderate Growth (60/40)":
        holdings = sample_data.get("balanced", {}).get("holdings", [])
    elif portfolio_mode == "Conservative Income":
        holdings = sample_data.get("conservative", {}).get("holdings", [])
    else:
        raw_json = st.text_area(
            "Enter Holdings JSON:",
            value=json.dumps([
                {"symbol": "VOO", "shares": 20, "price": 495.0, "category": "US Equities", "expense_ratio": 0.03},
                {"symbol": "VXUS", "shares": 30, "price": 60.0, "category": "Intl Equities", "expense_ratio": 0.08},
                {"symbol": "BND", "shares": 25, "price": 72.0, "category": "Bonds", "expense_ratio": 0.03},
            ], indent=2),
            height=150,
        )
        try:
            holdings = json.loads(raw_json)
        except Exception:
            st.error("Invalid JSON format.")
            holdings = []

    if holdings:
        df_holdings = pd.DataFrame(holdings)
        st.markdown("##### Current Holdings")
        st.dataframe(df_holdings, use_container_width=True)

        res_str = calculate_portfolio_metrics.invoke({"holdings_json": json.dumps(holdings)})
        res_data = json.loads(res_str)

        if "total_value" in res_data:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Portfolio Value", f"${res_data['total_value']:,.2f}")
            c2.metric("Weighted Expense Ratio", f"{res_data['weighted_expense_ratio_pct']:.3f}%")
            c3.metric("Asset Classes", f"{len(res_data['allocation_percentages'])}")

            # Visualizations
            st.markdown("##### Asset Allocation Breakdown")
            alloc_df = pd.DataFrame(
                list(res_data["allocation_percentages"].items()),
                columns=["Category", "Percentage"],
            )

            fig = px.pie(
                alloc_df,
                names="Category",
                values="Percentage",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# TAB 3: MARKET OVERVIEW
# ──────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Live Market Quotes & Ticker Analytics")
    col_search, _ = st.columns([2, 2])
    with col_search:
        ticker_input = st.text_input("Enter Stock / ETF Symbol:", value="VOO").upper()

    if ticker_input:
        quote_str = get_market_quote.invoke({"ticker": ticker_input})
        quote_data = json.loads(quote_str)

        if "current_price" in quote_data:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${quote_data['current_price']:.2f}")
            m2.metric("Daily Change", quote_data.get("change_percent", "N/A"))
            m3.metric("52-Week High", f"${quote_data.get('fifty_two_week_high', 0) or 0:.2f}")
            m4.metric("52-Week Low", f"${quote_data.get('fifty_two_week_low', 0) or 0:.2f}")

            st.json(quote_data)
        else:
            st.error(quote_data.get("error", "Could not fetch ticker details."))


# ──────────────────────────────────────────────────────────────────────
# TAB 4: GOAL PLANNING
# ──────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Financial Goal & Compound Growth Projections")

    g1, g2 = st.columns(2)
    with g1:
        init_deposit = st.number_input("Initial Investment ($):", value=5000, step=500)
        monthly_contrib = st.number_input("Monthly Contribution ($):", value=500, step=50)
    with g2:
        return_rate = st.slider("Expected Annual Return (%):", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
        years = st.slider("Investment Horizon (Years):", min_value=1, max_value=40, value=20, step=1)

    proj_str = project_goal_growth.invoke({
        "initial_amount": float(init_deposit),
        "monthly_contribution": float(monthly_contrib),
        "annual_return_pct": float(return_rate),
        "years": int(years),
    })
    proj_data = json.loads(proj_str)

    p1, p2, p3 = st.columns(3)
    p1.metric("Projected Total Wealth", f"${proj_data['projected_value']:,.2f}")
    p2.metric("Total Contributed", f"${proj_data['total_contributed']:,.2f}")
    p3.metric("Total Growth / Interest", f"${proj_data['total_growth_interest']:,.2f}")

    # Generate growth curve
    year_points = list(range(0, years + 1))
    contrib_points = []
    total_points = []

    r = (return_rate / 100.0) / 12.0
    for y in year_points:
        n = y * 12
        if r == 0:
            fv = init_deposit + (monthly_contrib * n)
        else:
            fv = (init_deposit * ((1 + r) ** n)) + (monthly_contrib * (((1 + r) ** n - 1) / r))
        contrib_points.append(init_deposit + (monthly_contrib * n))
        total_points.append(fv)

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=year_points, y=contrib_points, name="Principal Contributed", fill="tozeroy"))
    fig_growth.add_trace(go.Scatter(x=year_points, y=total_points, name="Projected Balance (Compound Growth)", fill="tonexty"))
    fig_growth.update_layout(
        title="Wealth Accumulation Over Time",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
    )
    st.plotly_chart(fig_growth, use_container_width=True)