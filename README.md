# AI Finance Assistant

Production-grade multi-agent financial literacy and guidance platform built with **LangGraph**, **OpenAI/Gemini**, and **Streamlit**.

## Project Architecture

1. **Orchestrator Node**: Intent routing and task decomposition via structured Pydantic schemas.
2. **Specialized Worker Agents**:
   - `finance_qa`: RAG-backed foundational investment literacy.
   - `portfolio_agent`: Portfolio concentration and Sharpe ratio analysis.
   - `market_agent`: Real-time quotes and moving averages.
   - `goal_agent`: Deterministic milestone and growth projections.
   - `news_agent`: Macro and ticker sentiment synthesis.
   - `tax_agent`: Account rules (Roth IRA, 401k, capital gains).
3. **Response Synthesizer**: Unifies parallel responses and attaches compliance disclaimers.

## Setup Instructions

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Environment configuration
cp .env.example .env
# Add OPENAI_API_KEY in .env

# 4. Launch web application
streamlit run src/web_app/app.py
