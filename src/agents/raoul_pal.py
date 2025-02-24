from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_macro_data, get_crypto_metrics, get_financial_metrics
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
from datetime import datetime

class RaoulPalSignal(BaseModel):
    signal: Literal["long", "short", "neutral"]
    confidence: float
    reasoning: str
    macro_drivers: list[str]
    narrative_strength: float

def raoul_pal_agent(state: AgentState):
    """
    Analyzes assets through Raoul Pal's macro lens focusing on:
    - Global liquidity conditions
    - Debt supercycle dynamics
    - Technological adoption curves
    - Network effects analysis
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    pal_analysis = {}

    # Fetch global macro context once for all assets
    progress.update_status("raoul_pal_agent", "system", "Fetching macro context")
    macro_context = {
        "liquidity": get_macro_data("global_liquidity", end_date),
        "debt": get_macro_data("debt_supercycle", end_date),
        "tech": get_macro_data("tech_adoption", end_date),
        "dollar": get_macro_data("dollar_index", end_date)
    }
    
    # Get real-time narrative pulse
    narrative_data = analyze_narrative_pulse(end_date)
    
    for ticker in tickers:
        progress.update_status("raoul_pal_agent", ticker, "Analyzing macro alignment")
        
        asset_analysis = {}
        
        # Asset-specific data collection
        if ticker in ["BTC", "ETH"]:  # Crypto assets
            progress.update_status("raoul_pal_agent", ticker, "Fetching crypto metrics")
            crypto_data = get_crypto_metrics(ticker, end_date)
            asset_analysis.update({
                "network_growth": crypto_data.active_addresses_change_30d,
                "holder_distribution": crypto_data.top_1p_holders,
                "nvt_ratio": crypto_data.nvt_ratio_90d_avg,
                "exchange_flows": crypto_data.net_exchange_flows
            })
        else:  # Traditional assets
            progress.update_status("raoul_pal_agent", ticker, "Fetching financial metrics")
            asset_data = get_financial_metrics(ticker, end_date)
            asset_analysis.update({
                "dollar_correlation": asset_data.usd_correlation_1y,
                "inflation_beta": asset_data.inflation_beta,
                "tech_exposure": asset_data.tech_investment_ratio
            })
        # Core analysis components  
        macro_score = calculate_macro_alignment(macro_context, asset_analysis)
        network_score = analyze_network_effects(ticker, asset_analysis)
        narrative_score = calculate_narrative_alignment(ticker, narrative_data)
        
        # Composite score calculation
        total_score = (
            0.5 * macro_score + 
            0.3 * network_score + 
            0.2 * narrative_score
        )
        
        # Signal determination
        if total_score >= 7.5:
            signal = "long"
            confidence = min(95, 80 + (total_score - 7.5)*10)
        elif total_score <= 3.5:
            signal = "short" 
            confidence = min(95, 80 + (3.5 - total_score)*10)
        else:
            signal = "neutral"
            confidence = 50 - abs(total_score - 5.5)*10
            
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "macro_breakdown": {
                "liquidity_conditions": macro_context["liquidity"].state,
                "debt_cycle_phase": macro_context["debt"].phase,
                "tech_adoption_rate": macro_context["tech"].adoption_rate
            },
            "network_metrics": asset_analysis.get("network_growth"),
            "narrative_strength": narrative_score
        }
        
        # Generate LLM-powered analysis
        pal_output = generate_pal_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            macro_context=macro_context,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        pal_analysis[ticker] = {
            "signal": pal_output.signal,
            "confidence": pal_output.confidence,
            "reasoning": pal_output.reasoning,
            "macro_drivers": pal_output.macro_drivers,
            "narrative_strength": pal_output.narrative_strength
        }
    
    # State management
    message = HumanMessage(
        content=json.dumps(pal_analysis),
        name="raoul_pal_agent"
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(pal_analysis, "Raoul Pal Agent")
    
    state["data"]["analyst_signals"]["raoul_pal_agent"] = pal_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }

def calculate_macro_alignment(macro_context: dict, asset_data: dict) -> float:
    """Score 0-10 based on Pal's 'Everything Code' framework"""
    score = 0
    
    # Liquidity conditions (40% weight)
    if macro_context["liquidity"].trend == "expanding":
        score += 4
    elif macro_context["liquidity"].trend == "contracting":
        score -= 2
        
    # Debt cycle alignment (30% weight)
    if macro_context["debt"].phase == "late":
        score += 3  # Favors hard assets/crypto
    elif macro_context["debt"].phase == "early":
        score += 1  # Favors growth assets
        
    # Tech disruption exposure (20% weight)
    if asset_data.get("tech_exposure", 0) > 0.15 or \
       asset_data.get("network_growth", 0) > 25:
        score += 2
        
    # Dollar correlation (10% weight)
    if asset_data.get("dollar_correlation", 0) < -0.4:
        score += 1
        
    return min(10, max(0, score))

def analyze_network_effects(ticker: str, asset_data: dict) -> float:
    """Score 0-10 for network effect strength (crypto-specific)"""
    if ticker in ["BTC", "ETH"]:
        score = 0
        # Network growth (40%)
        if asset_data["network_growth"] > 20:
            score += 4
        # Holder distribution (30%)
        if asset_data["holder_distribution"] < 25:
            score += 3
        # Exchange flows (30%)
        if asset_data["exchange_flows"] < 0:  # Negative = net outflow
            score += 3
        return score
    else:
        # Traditional asset network scoring
        return min(asset_data.get("tech_exposure", 0) * 10, 10)

def calculate_narrative_alignment(ticker: str, narrative_data: dict) -> float:
    """Score 0-10 based on narrative momentum"""
    ticker_narratives = narrative_data.get(ticker, {})
    score = 0
    
    # Social volume (40%)
    if ticker_narratives.get("social_volume", 0) > 1000:
        score += 4
        
    # Developer activity (30%)
    if ticker_narratives.get("github_activity", 0) > 50:
        score += 3
        
    # Institutional mentions (30%)
    if ticker_narratives.get("institutional_mentions", 0) > 10:
        score += 3
        
    return min(score, 10)

def generate_pal_output(
    ticker: str,
    analysis_data: dict,
    macro_context: dict,
    model_name: str,
    model_provider: str,
) -> RaoulPalSignal:
    """Generates investment decisions in Raoul Pal's macro style"""
    template = ChatPromptTemplate.from_messages([
        ("system", 
         """Act as Raoul Pal's AI analyst. Evaluate assets through his "Everything Code" framework:

         Global Context:
         - Liquidity Conditions: {liquidity_state}
         - Debt Cycle Phase: {debt_phase}
         - Tech Adoption Rate: {tech_rate}
         - Dollar Index Trend: {dollar_trend}

         Analysis Principles:
         1. Focus on macro regime shifts over company fundamentals
         2. Seek asymmetric risk/reward opportunities
         3. Prioritize assets benefiting from technological disruption
         4. Favor network effects and first-mover advantages
         5. Consider narrative momentum and social consensus

         Output Requirements:
         - Signal (long/short/neutral)
         - Confidence score 0-100
         - 3-5 key macro drivers
         - Narrative strength assessment"""),
        ("human", 
         """Asset: {ticker}
         {analysis}

         Generate Raoul Pal-style analysis in JSON format:
         {{
           "signal": "long/short/neutral",
           "confidence": "float",
           "reasoning": "string",
           "macro_drivers": ["list", "of", "drivers"],
           "narrative_strength": "float"
         }}""")
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "analysis": json.dumps(analysis_data, indent=2),
        "liquidity_state": macro_context["liquidity"].description,
        "debt_phase": macro_context["debt"].phase_description,
        "tech_rate": f"{macro_context['tech'].adoption_rate:.1%} YoY",
        "dollar_trend": macro_context["dollar"].trend_description
    })

    def create_default_signal():
        return RaoulPalSignal(
            signal="neutral",
            confidence=50.0,
            reasoning="Insufficient data for conclusive analysis",
            macro_drivers=[],
            narrative_strength=5.0
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=RaoulPalSignal,
        agent_name="raoul_pal_agent",
        default_factory=create_default_signal,
    )

def analyze_narrative_pulse(as_of_date: datetime) -> dict:
    """Analyzes narrative momentum across data sources"""
    # Implementation would integrate with:
    # - Social media APIs (Twitter, Reddit)
    # - News sentiment analysis
    # - GitHub activity tracking
    # - Institutional filings parsing
    return {
        "BTC": {
            "social_volume": 1450,
            "github_activity": 82,
            "institutional_mentions": 15
        },
        "ETH": {
            "social_volume": 920,
            "github_activity": 127,
            "institutional_mentions": 9
        }
    }