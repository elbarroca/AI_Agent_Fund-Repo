from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_media_sentiment, get_short_interest, get_prices
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import pandas as pd
import numpy as np

class JimCramerSignal(BaseModel):
    signal: Literal["buy", "sell", "hold", "speculative buy"]
    confidence: float
    reasoning: str
    catchphrase: str

def jim_cramer_agent(state: AgentState):
    """
    Enhanced Jim Cramer agent with quant-grade technical analysis and signature entertainment factor
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    cramer_analysis = {}
    
    # Get market-wide sentiment
    progress.update_status("jim_cramer_agent", "system", "Checking market mood")
    market_sentiment = get_media_sentiment("SPY", end_date, sources=["cnbc", "twitter", "reddit"])
    
    for ticker in tickers:
        progress.update_status("jim_cramer_agent", ticker, "Collecting data")
        
        try:
            # Get full price history for advanced technicals
            prices = get_prices(ticker, start_date, end_date)
            prices_df = prices_to_df(prices)
            
            # Calculate advanced technical indicators
            technicals = {
                "RSI": calculate_rsi(prices_df, 14).iloc[-1],
                "50D_MA": calculate_ema(prices_df, 50).iloc[-1],
                "200D_MA": calculate_ema(prices_df, 200).iloc[-1],
                "volume_spike": prices_df["volume"].iloc[-1] / prices_df["volume"].rolling(30).mean().iloc[-1],
                "adx": calculate_adx(prices_df, 14)["adx"].iloc[-1],
                "atr": calculate_atr(prices_df).iloc[-1],
                "hurst": calculate_hurst_exponent(prices_df["close"]),
                "bb_position": get_bollinger_position(prices_df),
                "close": prices_df["close"].iloc[-1]
            }
            
            sentiment_data = get_media_sentiment(ticker, end_date)
            short_data = get_short_interest(ticker, end_date)
            
            progress.update_status("jim_cramer_agent", ticker, "Crunching numbers")
            technical_score = analyze_technicals(technicals, prices_df)
            news_score = analyze_news_flow(sentiment_data)
            meme_score = analyze_meme_potential(ticker, sentiment_data)
            
            # BOOYAH Score v2 with volatility adjustment
            total_score = (
                0.35 * technical_score +
                0.25 * news_score +
                0.25 * meme_score +
                0.15 * (1 - short_data["days_to_cover"]/10)
            ) * volatility_adjustment(technicals["atr"], technicals["close"])
            
            # Generate high-conviction signal
            if total_score >= 8.0:
                signal = "buy"
                confidence = min(95, 80 + (total_score - 8.0)*3)
            elif total_score <= 2.0:
                signal = "sell"
                confidence = min(95, 80 + (2.0 - total_score)*3)
            elif meme_score > 7.5 and technicals["volume_spike"] > 2.0:
                signal = "speculative buy"
                confidence = 65 + (meme_score * 3)
            else:
                signal = "hold"
                confidence = 50 - abs(total_score - 5.0)*10
            
            analysis_data[ticker] = {
                "signal": signal,
                "score": total_score,
                "technicals": technicals,
                "news_sentiment": sentiment_data,
                "short_interest": short_data,
                "market_mood": market_sentiment["composite_score"]
            }
            
            progress.update_status("jim_cramer_agent", ticker, "Generating analysis")
            cramer_output = generate_cramer_output(
                ticker=ticker, 
                analysis_data=analysis_data[ticker],
                model_name=state["metadata"]["model_name"],
                model_provider=state["metadata"]["model_provider"],
            )
            
            cramer_analysis[ticker] = {
                "signal": cramer_output.signal,
                "confidence": cramer_output.confidence,
                "reasoning": cramer_output.reasoning,
                "catchphrase": cramer_output.catchphrase
            }
            
        except Exception as e:
            progress.update_status("jim_cramer_agent", ticker, f"Error: {str(e)}")
            continue
        
        progress.update_status("jim_cramer_agent", ticker, "Done")
    
    message = HumanMessage(
        content=json.dumps(cramer_analysis),
        name="jim_cramer_agent"
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(cramer_analysis, "Jim Cramer Agent")
    
    state["data"]["analyst_signals"]["jim_cramer_agent"] = cramer_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }

def analyze_technicals(technicals: dict, prices_df: pd.DataFrame) -> float:
    """Quant-powered technical analysis scoring (0-10 scale)"""
    score = 0
    
    # 1. Trend Power (30%)
    if technicals["adx"] > 40:
        score += 3.0
        if technicals["50D_MA"] > technicals["200D_MA"]:
            score += 1.5  # Golden cross bonus
    elif technicals["adx"] < 25:
        score -= 2.0
    
    # 2. Mean Reversion Potential (25%)
    bb_score = {
        "below_lower": 4.0,
        "between_bands": 0.0,
        "above_upper": -3.0
    }[technicals["bb_position"]]
    score += bb_score
    
    # 3. Momentum Juice (25%)
    rsi = technicals["RSI"]
    if rsi < 30:
        score += 2.5
    elif rsi > 70:
        score -= 2.0
    score += min(2.0, (technicals["volume_spike"] - 1.0))  # Volume bonus
    
    # 4. Volatility Edge (20%)
    atr_ratio = technicals["atr"] / technicals["close"]
    if atr_ratio > 0.04:
        score += 2.0  # High volatility opportunity
    
    # Normalize to 10-point scale
    return min(10.0, max(0.0, score * 1.2))

def analyze_news_flow(sentiment_data: dict) -> float:
    """Media-driven momentum scoring"""
    score = 0
    
    # Breaking news impact
    breaking_news = [n for n in sentiment_data["recent_news"] if n["impact_score"] > 0.7]
    score += min(3.0, len(breaking_news) * 1.5)
    
    # Earnings surprise multiplier
    surprise = sentiment_data.get("earnings_surprise", 0)
    score += abs(surprise) * 20  # 1% surprise = 0.2 points
    
    # Insider activity
    if sentiment_data["insider_buy_ratio"] > 0.6:
        score += 2.0
    elif sentiment_data["insider_sell_ratio"] > 0.6:
        score -= 1.5
    
    return min(10.0, max(0.0, score))

def analyze_meme_potential(ticker: str, sentiment_data: dict) -> float:
    """Meteoric rise probability score"""
    score = 0
    
    # Social media frenzy
    if sentiment_data["social_volume"] > 5000:
        score += 4.0
    elif sentiment_data["social_volume"] > 2000:
        score += 2.0
    
    # Short squeeze potential
    if sentiment_data["short_interest_ratio"] > 30:
        score += 3.0
    elif sentiment_data["short_interest_ratio"] > 20:
        score += 1.5
    
    # Retail frenzy indicator
    if sentiment_data["retail_volume_pct"] > 0.5:
        score += 2.0
    
    return min(10.0, score)

def volatility_adjustment(atr: float, price: float) -> float:
    """Dynamic confidence scaling based on volatility"""
    atr_ratio = atr / price
    if atr_ratio > 0.05:  # High volatility
        return 1.3
    elif atr_ratio < 0.02:  # Low volatility
        return 0.8
    return 1.0

def generate_cramer_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> JimCramerSignal:
    """Generates high-octane Cramer-style recommendations"""
    template = ChatPromptTemplate.from_messages([
        ("system", 
         f"""You are Jim Cramer's hyperactive AI twin! Analyze {ticker} with:
         
         TECHNICAL WEAPONS:
         - ADX Trend Strength: {{adx}} (40+ = STRONG TREND)
         - Bollinger Band Position: {{bb_pos}} 
         - RSI: {{rsi}} (UNDER 30 = OVERSOLD)
         - 50D/200D MA: {{ma50}} vs {{ma200}}
         - Volume Spike: {{volume_spike}}x
         - ATR Volatility: {{atr}} (${'{0:.2f}'.format(analysis_data['technicals']['atr'])})
         
         RULES:
         1. USE ALL CAPS FOR KEY POINTS
         2. MENTION 3+ TECHNICAL FACTORS
         3. INCLUDE 2 MEMORABLE CATCHPHRASES
         4. REFERENCE VOLATILITY FOR DRAMA
         5. NEVER DOUBT YOURSELF
         
         Respond in JSON:
         {{
           "signal": "buy/sell/hold/speculative buy",
           "confidence": 0-100,
           "reasoning": "string",
           "catchphrase": "string"
         }}"""),
        ("human", 
         f"""TICKER: {ticker}
         PRICE: ${analysis_data['technicals']['close']}
         NEWS: {analysis_data['news_sentiment']['top_headline']}
         SHORT INTEREST: {analysis_data['short_interest']['days_to_cover']} days
         MEME SCORE: {analysis_data['score']}/10
         MARKET MOOD: {analysis_data['market_mood']}
         GIVE ME THE CRAMER CALL!""")
    ])

    prompt = template.invoke({
        "adx": round(analysis_data["technicals"]["adx"], 1),
        "bb_pos": analysis_data["technicals"]["bb_position"].upper(),
        "rsi": round(analysis_data["technicals"]["RSI"], 1),
        "ma50": round(analysis_data["technicals"]["50D_MA"], 2),
        "ma200": round(analysis_data["technicals"]["200D_MA"], 2),
        "volume_spike": round(analysis_data["technicals"]["volume_spike"], 1),
        "atr": round(analysis_data["technicals"]["atr"], 2)
    })

    def create_default_signal():
        return JimCramerSignal(
            signal="hold",
            confidence=50.0,
            reasoning="Needs more research!",
            catchphrase="STAY ON THE SIDELINES!"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=JimCramerSignal,
        agent_name="jim_cramer_agent",
        default_factory=create_default_signal,
    )

# Technical analysis utilities
def prices_to_df(prices):
    return pd.DataFrame(prices).set_index("date")

def calculate_ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()

def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period=14):
    df = df.copy()
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]
    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()
    return df

def calculate_atr(df, period=14):
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )
    return tr.rolling(period).mean()

def get_bollinger_position(df, window=20):
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper = ma + 2*std
    lower = ma - 2*std
    close = df["close"].iloc[-1]
    if close > upper.iloc[-1]:
        return "above_upper"
    elif close < lower.iloc[-1]:
        return "below_lower"
    return "between_bands"

def calculate_hurst_exponent(series, max_lag=50):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    try:
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except:
        return 0.5