import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from data.models import (
    Price,
    FinancialMetrics,
    CompanyNews
)

from data.cache import get_cache

# Global cache instance
_cache = get_cache()

class AlphaVantageClient:
    """Alpha Vantage API client that matches the expected interface for data sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key. If not provided, will check ALPHA_VANTAGE_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """
        Fetch daily price data from Alpha Vantage.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of Price objects
        """
        # Check cache first
        if cached_data := _cache.get_prices(ticker):
            # Filter cached data by date range
            filtered_data = [Price(**price) for price in cached_data 
                            if start_date <= price["time"] <= end_date]
            if filtered_data:
                return filtered_data
                
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "full",  # Get full data to capture the date range
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                return []
                
            time_series = data["Time Series (Daily)"]
            prices = []
            
            for date, values in time_series.items():
                if start_date <= date <= end_date:
                    price = Price(
                        time=date,
                        open=float(values["1. open"]),
                        high=float(values["2. high"]),
                        low=float(values["3. low"]),
                        close=float(values["4. close"]),
                        volume=int(values["5. volume"])
                    )
                    prices.append(price)
            
            # Sort by date in descending order (newest first)
            prices.sort(key=lambda x: x.time, reverse=True)
            
            # Cache the results
            _cache.set_prices(ticker, [p.model_dump() for p in prices])
            
            return prices
            
        except Exception as e:
            print(f"Alpha Vantage error getting prices for {ticker}: {str(e)}")
            return []
    
    def get_intraday_prices(
        self, 
        ticker: str, 
        interval: str = "5min",
        output_size: str = "full"
    ) -> List[Price]:
        """
        Fetch intraday price data from Alpha Vantage.
        
        Args:
            ticker: Stock ticker symbol
            interval: Time interval between data points (e.g., "1min", "5min", "15min", "30min", "60min")
            output_size: Amount of data to return ("compact" or "full")
            
        Returns:
            List of Price objects with intraday data
        """
        # For intraday data, we don't use the standard price cache
        # as it would conflict with daily prices
        cache_key = f"{ticker}_intraday_{interval}"
        
        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": ticker,
                "interval": interval,
                "outputsize": output_size,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                return []
                
            time_series = data[time_series_key]
            prices = []
            
            for timestamp, values in time_series.items():
                price = Price(
                    time=timestamp,
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    volume=int(values["5. volume"])
                )
                prices.append(price)
            
            # Sort by timestamp in descending order (newest first)
            prices.sort(key=lambda x: x.time, reverse=True)
            
            return prices
            
        except Exception as e:
            print(f"Alpha Vantage error getting intraday prices for {ticker}: {str(e)}")
            return []
    
    def get_technical_indicators(
        self,
        ticker: str,
        indicator: str,
        time_period: int = 14,
        series_type: str = "close"
    ) -> Dict[str, Any]:
        """
        Fetch technical indicator values from Alpha Vantage.
        
        Args:
            ticker: Stock ticker symbol
            indicator: Technical indicator type (e.g., "RSI", "MACD", "SMA", etc.)
            time_period: Time period to consider for the indicator
            series_type: Price series to use (e.g., "close", "open", "high", "low")
            
        Returns:
            Dictionary with technical indicator data
        """
        try:
            # Handle different indicator types
            if indicator.upper() == "SMA":
                function = "SMA"
            elif indicator.upper() == "EMA":
                function = "EMA"
            elif indicator.upper() == "MACD":
                function = "MACD"
            elif indicator.upper() == "RSI":
                function = "RSI"
            elif indicator.upper() == "ADX":
                function = "ADX"
            elif indicator.upper() == "CCI":
                function = "CCI"
            elif indicator.upper() == "BBANDS":
                function = "BBANDS"
            elif indicator.upper() == "STOCH":
                function = "STOCH"
            else:
                function = indicator.upper()
            
            params = {
                "function": function,
                "symbol": ticker,
                "interval": "daily",
                "time_period": time_period,
                "series_type": series_type,
                "apikey": self.api_key
            }
            
            # MACD has different parameters
            if function == "MACD":
                params.pop("time_period", None)
                params.update({
                    "fastperiod": "12",
                    "slowperiod": "26",
                    "signalperiod": "9"
                })
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Remove metadata and return the indicator values
            if "Meta Data" in data:
                data.pop("Meta Data")
            
            # Convert the nested structure to a simpler format
            result = {}
            
            # Handle different response structures based on the indicator
            for key, values in data.items():
                if key.startswith("Technical Analysis:"):
                    indicator_data = {}
                    
                    for date, indicators in values.items():
                        indicator_data[date] = {}
                        for indicator_key, value in indicators.items():
                            # Remove indicator prefix for cleaner keys
                            clean_key = indicator_key.split(". ")[-1] if ". " in indicator_key else indicator_key
                            indicator_data[date][clean_key] = float(value)
                    
                    result["data"] = indicator_data
                    result["indicator"] = function
                    result["time_period"] = time_period
            
            return result
            
        except Exception as e:
            print(f"Alpha Vantage error getting {indicator} for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    def get_forex_data(
        self,
        from_currency: str,
        to_currency: str,
        output_size: str = "full"
    ) -> List[Dict[str, Any]]:
        """
        Fetch forex data from Alpha Vantage.
        
        Args:
            from_currency: From currency code (e.g., "USD")
            to_currency: To currency code (e.g., "EUR")
            output_size: Amount of data to return ("compact" or "full")
            
        Returns:
            List of dictionaries with forex data
        """
        try:
            params = {
                "function": "FX_DAILY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "outputsize": output_size,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series FX (Daily)" not in data:
                return []
                
            time_series = data["Time Series FX (Daily)"]
            result = []
            
            for date, values in time_series.items():
                forex_data = {
                    "date": date,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"])
                }
                result.append(forex_data)
            
            # Sort by date in descending order (newest first)
            result.sort(key=lambda x: x["date"], reverse=True)
            
            return result
            
        except Exception as e:
            print(f"Alpha Vantage error getting forex data for {from_currency}/{to_currency}: {str(e)}")
            return []
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """
        Fetch sector performance data from Alpha Vantage.
        
        Returns:
            Dictionary with sector performance data
        """
        try:
            params = {
                "function": "SECTOR",
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Remove metadata
            if "Meta Data" in data:
                data.pop("Meta Data")
            
            return data
            
        except Exception as e:
            print(f"Alpha Vantage error getting sector performance: {str(e)}")
            return {"error": str(e)}
    
    def search_ticker(self, keywords: str) -> List[Dict[str, str]]:
        """
        Search for ticker symbols by keywords using Alpha Vantage.
        
        Args:
            keywords: Search keywords
            
        Returns:
            List of dictionaries with search results
        """
        try:
            params = {
                "function": "SYMBOL_SEARCH",
                "keywords": keywords,
                "apikey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "bestMatches" not in data:
                return []
                
            return data["bestMatches"]
            
        except Exception as e:
            print(f"Alpha Vantage error searching for {keywords}: {str(e)}")
            return []

def alpha_vantage_to_df(data: List[Price]) -> pd.DataFrame:
    """
    Convert Alpha Vantage price data to a DataFrame.
    
    Args:
        data: List of Price objects
    
    Returns:
        DataFrame with price data
    """
    df = pd.DataFrame([p.model_dump() for p in data])
    
    if not df.empty:
        df["Date"] = pd.to_datetime(df["time"])
        df.set_index("Date", inplace=True)
        
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        df.sort_index(inplace=True)
    
    return df 