"""
AI Finance Assistant — Interactive Multi-Tab Streamlit Dashboard.
Features: Multi-agent Chat, Portfolio Analytics, Live Market Tickers & Trends, and Goal Projections.
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

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.core.config import BASE_DIR
from src.tools.goal_tools import project_goal_growth
from src.tools.market_tools import get_market_history, get_market_quote
from src.tools.portfolio_tools import calculate_portfolio_metrics
from src.workflow.graph import build_finance_graph

st.set_page_config(
    page_title="AI Finance Assistant",
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


def _current() -> dict:
    return st.session_state.conversations[st.session_state.current_chat_id]


# ── Sidebar ───────────────────────────────────────────────────────────
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


st.title("AI Finance Assistant")
st.caption("Multi-agent financial guidance, real-time analytics, and goal modeling.")

tabs = st.tabs([
    "💬 Chat Assistant",
    "📊 Portfolio Analysis",
    "📈 Market Overview",
    "🎯 Goal Planning",
])

# ── TAB 1: CHAT ASSISTANT ─────────────────────────────────────────────
with tabs[0]:
    current = _current()

    def _render_routing_meta(meta: dict):
        if not meta or not meta.get("agents"):
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


# ── TAB 2: PORTFOLIO ANALYSIS ─────────────────────────────────────────
with tabs[1]:
    st.subheader("Portfolio Allocation & Health Check")

    sample_file = BASE_DIR / "src/data/sample_portfolios.json"
    sample_data = {}
    if sample_file.exists():
        with open(sample_file, "r") as f:
            sample_data = json.load(f)

    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        portfolio_mode = st.selectbox(
            "Choose Portfolio Template or Custom Input:",
            ["Moderate Growth (60/40)", "Conservative Income", "Custom JSON Input"],
        )
    with col_p2:
        risk_appetite = st.selectbox(
            "Target Risk Appetite:",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
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
            height=130,
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

        res_str = calculate_portfolio_metrics.invoke({
            "holdings_json": json.dumps(holdings),
            "user_risk_appetite": risk_appetite,
        })
        res_data = json.loads(res_str)

        if "total_value" in res_data:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Value", f"${res_data['total_value']:,.2f}")
            c2.metric("Weighted Exp Ratio", f"{res_data['weighted_expense_ratio_pct']:.3f}%")
            c3.metric("Diversification Score", f"{res_data.get('diversification_score', 0)} / 100")
            c4.metric("Risk Status", res_data.get("risk_alignment", "Aligned"))

            st.info(f"💡 **Assessment & Recommendation:** {res_data.get('recommendation', '')}")

            if res_data.get("concentrated_positions"):
                st.warning(f"⚠️ **Concentration Risk Flag (>25% position):** {', '.join(res_data['concentrated_positions'])}")

            st.markdown("##### Asset Allocation Breakdown")
            alloc_df = pd.DataFrame(
                list(res_data["allocation_percentages"].items()),
                columns=["Category", "Percentage"],
            )

            col_pie, col_bar = st.columns(2)
            with col_pie:
                fig_pie = px.pie(
                    alloc_df,
                    names="Category",
                    values="Percentage",
                    hole=0.45,
                    title="Allocation Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_bar:
                fig_bar = px.bar(
                    alloc_df,
                    x="Category",
                    y="Percentage",
                    title="Allocation Weights (%)",
                    color="Category",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig_bar, use_container_width=True)


# ── TAB 3: MARKET OVERVIEW ────────────────────────────────────────────
with tabs[2]:
    st.subheader("Live Market Quotes & Trend Analytics")
    col_search, _ = st.columns([2, 2])
    with col_search:
        ticker_input = st.text_input("Enter Stock / ETF Symbol:", value="VOO").upper()

    if ticker_input:
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

            # Historical Trend Chart
            hist_str = get_market_history.invoke({"ticker": ticker_input, "period": "6mo"})
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
                    title=f"{ticker_input} 6-Month Historical Performance",
                    template="plotly_white",
                )
                fig_trend.update_traces(line_color="#2E7D32" if hist_data.get("period_return_pct", 0) >= 0 else "#C62828")
                st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.error(quote_data.get("error", "Could not fetch ticker details."))


# ── TAB 4: GOAL PLANNING ──────────────────────────────────────────────
with tabs[3]:
    st.subheader("Risk-Aware Financial Goal & Wealth Projections")

    g1, g2, g3 = st.columns(3)
    with g1:
        init_deposit = st.number_input("Initial Investment ($):", value=5000, step=500)
        monthly_contrib = st.number_input("Monthly Contribution ($):", value=500, step=50)
    with g2:
        risk_profile = st.selectbox("Investor Risk Profile:", ["Conservative", "Moderate", "Aggressive"], index=1)
        years = st.slider("Investment Horizon (Years):", min_value=1, max_value=40, value=20, step=1)
    with g3:
        custom_rate = st.checkbox("Custom Return Override")
        override_rate = st.number_input("Custom Return Rate (%):", value=7.5, step=0.5) if custom_rate else None

    proj_str = project_goal_growth.invoke({
        "initial_amount": float(init_deposit),
        "monthly_contribution": float(monthly_contrib),
        "years": int(years),
        "risk_profile": risk_profile,
        "annual_return_pct": float(override_rate) if override_rate else None,
    })
    proj_data = json.loads(proj_str)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Projected Median Wealth", f"${proj_data['projected_median_value']:,.2f}")
    p2.metric("Total Principal Contributed", f"${proj_data['total_contributed']:,.2f}")
    p3.metric("Pessimistic Scenario", f"${proj_data['pessimistic_band_value']:,.2f}")
    p4.metric("Optimistic Scenario", f"${proj_data['optimistic_band_value']:,.2f}")

    st.info(f"🎯 **Suggested Strategy:** {proj_data.get('recommended_asset_mix', '')} at ~{proj_data.get('assumed_annual_return', '')} expected return.")

    # Multi-scenario growth curve
    year_points = list(range(0, years + 1))
    contrib_points = []
    median_points = []
    low_points = []
    high_points = []

    profile_rates = {"Conservative": (3.0, 4.5, 6.0), "Moderate": (5.0, 7.5, 10.0), "Aggressive": (6.5, 10.0, 13.5)}
    r_low, r_med, r_high = profile_rates[risk_profile]
    if override_rate:
        r_med = override_rate

    def _fv(r_pct, y):
        r = (r_pct / 100.0) / 12.0
        n = y * 12
        return init_deposit + (monthly_contrib * n) if r == 0 else (init_deposit * ((1 + r) ** n)) + (monthly_contrib * (((1 + r) ** n - 1) / r))

    for y in year_points:
        contrib_points.append(init_deposit + (monthly_contrib * y * 12))
        median_points.append(_fv(r_med, y))
        low_points.append(_fv(r_low, y))
        high_points.append(_fv(r_high, y))

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=year_points, y=high_points, name="Optimistic Market Band", line=dict(dash="dot", color="#4CAF50")))
    fig_growth.add_trace(go.Scatter(x=year_points, y=median_points, name="Expected Median Growth", line=dict(color="#1976D2", width=3)))
    fig_growth.add_trace(go.Scatter(x=year_points, y=low_points, name="Conservative Market Band", line=dict(dash="dot", color="#FF9800")))
    fig_growth.add_trace(go.Scatter(x=year_points, y=contrib_points, name="Principal Contributed", line=dict(dash="dash", color="#757575")))

    fig_growth.update_layout(
        title=f"Multi-Scenario Wealth Accumulation ({risk_profile} Profile)",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
    )
    st.plotly_chart(fig_growth, use_container_width=True)