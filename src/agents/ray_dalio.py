from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_macro_data,
    get_asset_volatility,
    get_asset_correlations,
    get_historical_regimes,
    get_debt_metrics,
    get_central_bank_policies
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, validator
import json
from typing import Literal, Dict, List
from utils.progress import progress
from utils.llm import call_llm
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import pandas as pd

# ================
# Data Models
# ================
class EconomicRegime(BaseModel):
    name: Literal["Rising Growth", "Overheating", "Stagflation", "Deflationary Contraction"]
    confidence: float = 0.0
    indicators: Dict[str, float]
    volatility_regime: Literal["low", "medium", "high", "extreme"]

class DebtCycle(BaseModel):
    stage: Literal["Early", "Mid", "Late"]
    debt_to_gdp: float
    credit_growth: float
    debt_service_ratio: float

class RiskParityWeights(BaseModel):
    equities: float
    long_bonds: float
    gold: float
    commodities: float
    tips: float
    real_estate: float

    @validator('*')
    def check_weights(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v

class RayDalioOutput(BaseModel):
    allocation: RiskParityWeights
    regime: EconomicRegime
    debt_cycle: DebtCycle
    risk_metrics: Dict[str, float]
    leverage_factor: float
    historical_comparisons: List[str]
    reasoning: str
    confidence: float

# ================
# Core Agent
# ================
def ray_dalio_agent(state: AgentState) -> Dict:
    """
    Full implementation of Ray Dalio's All Weather strategy:
    1. Macroeconomic regime detection
    2. Debt cycle analysis
    3. Risk parity optimization
    4. Regime-based adjustments
    5. Volatility-targeted leverage
    6. Historical pattern matching
    """
    data = state["data"]
    current_date = data["end_date"]
    
    try:
        progress.update_status("ray_dalio_agent", "system", "Starting analysis")
        
        # Phase 1: Economic Regime Analysis
        regime = analyze_economic_regime(current_date)
        
        # Phase 2: Debt Cycle Assessment
        debt_cycle = analyze_debt_cycle(current_date)
        
        # Phase 3: Risk Parity Calculation
        base_weights, risk_metrics = calculate_risk_parity(current_date)
        
        # Phase 4: Regime Adjustments
        adjusted_weights = apply_regime_adjustments(base_weights, regime, debt_cycle)
        
        # Phase 5: Leverage Application
        final_allocation, leverage = apply_volatility_leverage(adjusted_weights, regime.volatility_regime)
        
        # Phase 6: Historical Analysis
        historical_comps = find_historical_analogs(regime, debt_cycle)
        
        # Phase 7: Generate Commentary
        analysis_output = generate_dalio_commentary(
            allocation=final_allocation,
            regime=regime,
            debt_cycle=debt_cycle,
            risk_metrics=risk_metrics,
            leverage=leverage,
            historical=historical_comps
        )

        # Update state
        message = HumanMessage(
            content=analysis_output.json(),
            name="ray_dalio_agent"
        )
        
        if state["metadata"].get("show_reasoning"):
            show_agent_reasoning(analysis_output.dict(), "Ray Dalio Agent")

        state["data"]["analyst_signals"]["ray_dalio_agent"] = analysis_output.dict()

        return {
            "messages": [message],
            "data": state["data"]
        }

    except Exception as e:
        progress.update_status("ray_dalio_agent", "system", f"Error: {str(e)}")
        return error_state(state, str(e))

# ================
# Analysis Components
# ================
def analyze_economic_regime(date: datetime) -> EconomicRegime:
    """Determine current economic regime using 5-factor model"""
    progress.update_status("ray_dalio_agent", "system", "Analyzing economic regime")
    
    indicators = {
        "gdp_growth": get_macro_data("gdp_growth", date),
        "cpi": get_macro_data("cpi", date),
        "pmi": get_macro_data("pmi", date),
        "yield_curve": get_macro_data("10y2y_spread", date),
        "consumer_sentiment": get_macro_data("consumer_sentiment", date)
    }
    
    # Calculate regime score using nonlinear combination
    growth_component = np.tanh((indicators["gdp_growth"] - 2.5)/1.0)
    inflation_component = np.tanh((indicators["cpi"] - 2.0)/1.0)
    pmi_component = (indicators["pmi"] - 50)/10
    sentiment_component = (indicators["consumer_sentiment"] - 50)/50
    
    regime_score = (
        0.4 * growth_component +
        0.3 * inflation_component +
        0.2 * pmi_component +
        0.1 * sentiment_component
    )
    
    # Determine regime classification
    if regime_score > 0.5:
        regime_name = "Rising Growth"
        confidence = min(100, 90 + 10*regime_score)
        volatility = "low"
    elif regime_score > 0:
        regime_name = "Overheating"
        confidence = 75 + 25*regime_score
        volatility = "medium"
    elif regime_score > -0.5:
        regime_name = "Stagflation"
        confidence = 60 - 40*regime_score
        volatility = "high"
    else:
        regime_name = "Deflationary Contraction"
        confidence = min(100, 80 - 20*regime_score)
        volatility = "extreme"

    return EconomicRegime(
        name=regime_name,
        confidence=confidence,
        indicators=indicators,
        volatility_regime=volatility
    )

def analyze_debt_cycle(date: datetime) -> DebtCycle:
    """Assess debt cycle stage using multiple metrics"""
    progress.update_status("ray_dalio_agent", "system", "Analyzing debt cycle")
    
    debt_data = {
        "debt_to_gdp": get_debt_metrics("debt_to_gdp", date),
        "credit_growth": get_debt_metrics("credit_growth", date),
        "debt_service_ratio": get_debt_metrics("debt_service_ratio", date)
    }
    
    # Calculate debt cycle score
    cycle_score = (
        0.5 * np.log(debt_data["debt_to_gdp"]/100) +
        0.3 * debt_data["credit_growth"] +
        0.2 * debt_data["debt_service_ratio"]
    )
    
    if cycle_score > 1.0:
        stage = "Late"
    elif cycle_score > 0:
        stage = "Mid"
    else:
        stage = "Early"

    return DebtCycle(
        stage=stage,
        **debt_data
    )

def calculate_risk_parity(date: datetime) -> tuple[RiskParityWeights, dict]:
    """Constrained risk parity optimization"""
    progress.update_status("ray_dalio_agent", "system", "Calculating risk parity")
    
    # Get market data
    assets = ["equities", "long_bonds", "gold", "commodities", "tips", "real_estate"]
    volatilities = np.array([get_asset_volatility(a, date) for a in assets])
    corr_matrix = get_asset_correlations(date)
    
    # Optimization setup
    n = len(assets)
    initial_weights = np.ones(n)/n
    
    def objective(weights):
        port_vol = np.sqrt(weights.T @ (corr_matrix * np.outer(weights, weights)) @ volatilities**2)
        risk_contrib = (weights * (corr_matrix @ (weights * volatilities**2))) / port_vol
        return np.sum((risk_contrib - 1/n)**2)
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: 0.4 - x}  # Max 40% per asset
    )
    
    bounds = [(0.01, 0.40) for _ in assets]
    
    # Solve optimization
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Format results
    weights = dict(zip([a.capitalize() for a in assets], result.x))
    risk_metrics = {
        "portfolio_volatility": np.sqrt(result.fun),
        "risk_contributions": result.x.tolist()
    }
    
    return RiskParityWeights(**weights), risk_metrics

def apply_regime_adjustments(
    weights: RiskParityWeights,
    regime: EconomicRegime,
    debt_cycle: DebtCycle
) -> RiskParityWeights:
    """Adjust base weights for current regime and debt cycle"""
    progress.update_status("ray_dalio_agent", "system", "Applying regime adjustments")
    
    # Base adjustments by regime
    regime_adjustments = {
        "Rising Growth": {"Equities": 0.05, "Commodities": -0.03},
        "Overheating": {"Tips": 0.04, "Gold": 0.02},
        "Stagflation": {"Gold": 0.07, "Commodities": 0.05, "Equities": -0.06},
        "Deflationary Contraction": {"Long_bonds": 0.08, "Tips": 0.03}
    }
    
    # Debt cycle modifiers
    debt_adjustments = {
        "Late": {"Real_estate": -0.04, "Gold": 0.03},
        "Mid": {"Equities": 0.02, "Tips": -0.01},
        "Early": {"Commodities": 0.04, "Long_bonds": -0.02}
    }
    
    # Apply adjustments
    adjusted = weights.dict()
    
    # Apply regime adjustments
    for asset, adj in regime_adjustments.get(regime.name, {}).items():
        adjusted[asset] += adj
    
    # Apply debt cycle adjustments
    for asset, adj in debt_adjustments.get(debt_cycle.stage, {}).items():
        adjusted[asset] += adj
    
    # Normalize weights
    total = sum(adjusted.values())
    normalized = {k: v/total for k, v in adjusted.items()}
    
    return RiskParityWeights(**normalized)

def apply_volatility_leverage(
    weights: RiskParityWeights,
    volatility_regime: str
) -> tuple[RiskParityWeights, float]:
    """Apply volatility-targeted leverage"""
    progress.update_status("ray_dalio_agent", "system", "Applying leverage")
    
    leverage_factors = {
        "low": 1.25,
        "medium": 1.10,
        "high": 0.85,
        "extreme": 0.70
    }
    
    factor = leverage_factors.get(volatility_regime, 1.0)
    leveraged = {k: v*factor for k, v in weights.dict().items()}
    
    return RiskParityWeights(**leveraged), factor

def find_historical_analogs(
    regime: EconomicRegime,
    debt_cycle: DebtCycle
) -> List[str]:
    """Find similar historical periods"""
    progress.update_status("ray_dalio_agent", "system", "Finding historical analogs")
    
    return get_historical_regimes(
        growth=regime.indicators["gdp_growth"],
        inflation=regime.indicators["cpi"],
        debt_stage=debt_cycle.stage,
        limit=3
    )

# ================
# Output Generation
# ================
def generate_dalio_commentary(
    allocation: RiskParityWeights,
    regime: EconomicRegime,
    debt_cycle: DebtCycle,
    risk_metrics: dict,
    leverage: float,
    historical: List[str]
) -> RayDalioOutput:
    """Generate Dalio-style portfolio commentary"""
    progress.update_status("ray_dalio_agent", "system", "Generating commentary")
    
    template = ChatPromptTemplate.from_messages([
        ("system", 
         """You are Ray Dalio's chief portfolio strategist. Explain allocation decisions using:
         
         Economic Regime: {regime_name} (Confidence: {confidence}%)
         - GDP Growth: {gdp}%
         - Inflation: {cpi}%
         - PMI: {pmi}
         - Yield Curve: {yield_curve}bps
         - Consumer Sentiment: {sentiment}
         
         Debt Cycle: {debt_stage} Stage
         - Debt/GDP: {debt_gdp}%
         - Credit Growth: {credit_growth}%
         - Debt Service Ratio: {debt_service}%
         
         Portfolio Allocation:
         {allocation}
         
         Risk Metrics:
         - Portfolio Volatility: {volatility}%
         - Largest Risk Contribution: {max_risk}%
         
         Leverage Factor: {leverage}x
         
         Historical Analogs: {historical}
         
         Rules:
         1. Use mechanical/cyclical analogies
         2. Explain debt cycle implications
         3. Discuss risk parity principles
         4. Justify regime adjustments
         5. Reference historical comparisons
         6. Maintain calm, rational tone"""),
        ("human", 
         """Generate comprehensive portfolio commentary in Dalio's signature style:""")
    ])

    prompt = template.invoke({
        "regime_name": regime.name,
        "confidence": regime.confidence,
        "gdp": regime.indicators["gdp_growth"],
        "cpi": regime.indicators["cpi"],
        "pmi": regime.indicators["pmi"],
        "yield_curve": regime.indicators["yield_curve"],
        "sentiment": regime.indicators["consumer_sentiment"],
        "debt_stage": debt_cycle.stage,
        "debt_gdp": debt_cycle.debt_to_gdp,
        "credit_growth": debt_cycle.credit_growth,
        "debt_service": debt_cycle.debt_service_ratio,
        "allocation": allocation.json(indent=2),
        "volatility": risk_metrics["portfolio_volatility"],
        "max_risk": max(risk_metrics["risk_contributions"]),
        "leverage": leverage,
        "historical": ", ".join(historical)
    })

    class OutputModel(BaseModel):
        reasoning: str
        confidence: float

    result = call_llm(
        prompt=prompt,
        model_name="gpt-4",
        model_provider="openai",
        pydantic_model=OutputModel
    )

    return RayDalioOutput(
        allocation=allocation,
        regime=regime,
        debt_cycle=debt_cycle,
        risk_metrics=risk_metrics,
        leverage_factor=leverage,
        historical_comparisons=historical,
        reasoning=result.reasoning,
        confidence=regime.confidence
    )

def error_state(state: AgentState, error_msg: str) -> Dict:
    """Handle error states gracefully"""
    state["data"]["analyst_signals"]["ray_dalio_agent"] = {
        "error": error_msg,
        "allocation": None,
        "confidence": 0.0
    }
    return {
        "messages": [HumanMessage(content=json.dumps({"error": error_msg}))],
        "data": state["data"]
    }