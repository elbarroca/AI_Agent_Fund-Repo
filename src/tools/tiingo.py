import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from data.models import (
    Price,
    FinancialMetrics,
    InsiderTrade,
    CompanyNews
)
from data.cache import get_cache

# Global cache instance
_cache = get_cache()


class TiingoClient:
    """Tiingo API client for accessing financial data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tiingo client.
        
        Args:
            api_key: Tiingo API key. If not provided, will check TIINGO_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY", "")
        self.base_url = "https://api.tiingo.com"
        
    def get_prices(self, ticker: str, start_date: str, end_date: str, frequency: str = "daily") -> List[Price]:
        """
        Fetch historical price data from Tiingo.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (e.g., "daily", "hourly", "minute")
            
        Returns:
            List of Price objects
        """
        # Check cache first
        if cached_data := _cache.get_prices(ticker):
            # Filter cached data by date range
            filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
            if filtered_data:
                return filtered_data
                
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            
            # Format endpoint based on frequency
            if frequency == "daily":
                endpoint = f"/tiingo/daily/{ticker}/prices"
            else:
                # For intraday data
                endpoint = f"/iex/{ticker}/prices"
                
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "format": "json"
            }
            
            # Add frequency-specific parameters for intraday data
            if frequency != "daily":
                params["resampleFreq"] = frequency
                
            response = requests.get(f"{self.base_url}{endpoint}", headers=headers, params=params)
            data = response.json()
            
            if not data:
                return []
                
            prices = []
            for item in data:
                # Format datetime differently based on whether it's daily or intraday
                if frequency == "daily":
                    date_str = item.get("date").split("T")[0]  # Remove the time component
                else:
                    date_str = item.get("date")
                    
                price = Price(
                    time=date_str,
                    open=float(item.get("open", 0)),
                    close=float(item.get("close", 0)),
                    high=float(item.get("high", 0)),
                    low=float(item.get("low", 0)),
                    volume=int(item.get("volume", 0))
                )
                prices.append(price)
            
            # Sort by date in descending order (newest first)
            prices.sort(key=lambda x: x.time, reverse=True)
            
            # Cache the results
            _cache.set_prices(ticker, [p.model_dump() for p in prices])
            
            return prices
            
        except Exception as e:
            print(f"Tiingo error getting prices for {ticker}: {str(e)}")
            return []
    
    def get_intraday_prices(self, ticker: str, start_date: str, end_date: str, interval: str = "1hour") -> List[Price]:
        """
        Fetch intraday price data from Tiingo.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (e.g., "1min", "5min", "1hour")
            
        Returns:
            List of Price objects
        """
        # Map interval to Tiingo's resampleFreq parameter
        if interval == "1min":
            frequency = "1min"
        elif interval == "5min":
            frequency = "5min"
        elif interval == "15min":
            frequency = "15min"
        elif interval == "30min":
            frequency = "30min"
        elif interval == "1hour":
            frequency = "1hour"
        else:
            frequency = "1hour"  # Default
            
        return self.get_prices(ticker, start_date, end_date, frequency=frequency)
    
    def get_ticker_metadata(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch ticker metadata from Tiingo.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with ticker metadata
        """
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            
            response = requests.get(f"{self.base_url}/tiingo/daily/{ticker}", headers=headers)
            data = response.json()
            
            return data
            
        except Exception as e:
            print(f"Tiingo error getting metadata for {ticker}: {str(e)}")
            return {}
    
    def get_news(self, tickers: List[str] = None, tags: List[str] = None, limit: int = 10) -> List[CompanyNews]:
        """
        Fetch company news from Tiingo.
        
        Args:
            tickers: List of stock ticker symbols
            tags: List of news tags
            limit: Maximum number of news items to retrieve
            
        Returns:
            List of CompanyNews objects
        """
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            
            params = {"limit": limit}
            
            if tickers:
                params["tickers"] = ",".join(tickers)
                
            if tags:
                params["tags"] = ",".join(tags)
                
            response = requests.get(f"{self.base_url}/tiingo/news", headers=headers, params=params)
            data = response.json()
            
            if not data:
                return []
                
            news_items = []
            for item in data:
                news = CompanyNews(
                    id=item.get("id", ""),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    url=item.get("url", ""),
                    tickers=item.get("tickers", []),
                    tags=item.get("tags", []),
                    published_at=item.get("publishedDate", ""),
                    source=item.get("source", "Tiingo")
                )
                news_items.append(news)
            
            return news_items
            
        except Exception as e:
            print(f"Tiingo error getting news: {str(e)}")
            return []
    
    def get_bulk_metrics(self, tickers: List[str], metric_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetch key metrics for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            metric_names: List of metric names to retrieve
            
        Returns:
            Dictionary mapping tickers to metric values
        """
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            
            ticker_string = ",".join(tickers)
            
            response = requests.get(f"{self.base_url}/tiingo/fundamentals/definitions", headers=headers)
            available_metrics = response.json()
            
            results = {}
            
            # Filter valid metrics based on available metrics from the API
            valid_metrics = [metric for metric in metric_names if any(m.get("metric") == metric for m in available_metrics)]
            
            for ticker in tickers:
                try:
                    ticker_data = {}
                    
                    # Get the most recent metrics for each ticker
                    response = requests.get(f"{self.base_url}/tiingo/fundamentals/{ticker}/statements", headers=headers)
                    statements = response.json()
                    
                    if statements and "statementData" in statements[0]:
                        latest_data = statements[0]["statementData"]
                        
                        for metric in valid_metrics:
                            if metric in latest_data:
                                ticker_data[metric] = latest_data[metric]
                    
                    results[ticker] = ticker_data
                    
                except Exception as inner_e:
                    print(f"Tiingo error getting metrics for {ticker}: {str(inner_e)}")
                    results[ticker] = {}
            
            return results
            
        except Exception as e:
            print(f"Tiingo error getting bulk metrics: {str(e)}")
            return {ticker: {} for ticker in tickers}
    
    def get_crypto_prices(self, crypto_ticker: str, start_date: str, end_date: str, frequency: str = "daily") -> List[Price]:
        """
        Fetch cryptocurrency price data from Tiingo.
        
        Args:
            crypto_ticker: Cryptocurrency ticker (e.g., "btcusd")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (e.g., "daily", "hourly", "1min")
            
        Returns:
            List of Price objects
        """
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "format": "json"
            }
            
            # Add frequency-specific parameters
            if frequency != "daily":
                params["resampleFreq"] = frequency
                
            response = requests.get(f"{self.base_url}/tiingo/crypto/prices", headers=headers, params=params)
            data = response.json()
            
            if not data:
                return []
                
            prices = []
            for item in data:
                price = Price(
                    time=item.get("date"),
                    open=float(item.get("open", 0)),
                    close=float(item.get("close", 0)),
                    high=float(item.get("high", 0)),
                    low=float(item.get("low", 0)),
                    volume=float(item.get("volume", 0))
                )
                prices.append(price)
            
            # Sort by date in descending order (newest first)
            prices.sort(key=lambda x: x.time, reverse=True)
            
            return prices
            
        except Exception as e:
            print(f"Tiingo error getting crypto prices for {crypto_ticker}: {str(e)}")
            return []
    
    def get_forex_prices(self, forex_pair: str, start_date: str, end_date: str, frequency: str = "daily") -> List[Price]:
        """
        Fetch forex price data from Tiingo.
        
        Args:
            forex_pair: Forex pair (e.g., "eurusd")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (e.g., "daily", "hourly", "1min")
            
        Returns:
            List of Price objects
        """
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "format": "json"
            }
            
            # Add frequency-specific parameters
            if frequency != "daily":
                params["resampleFreq"] = frequency
                
            response = requests.get(f"{self.base_url}/tiingo/fx/{forex_pair}/prices", headers=headers, params=params)
            data = response.json()
            
            if not data:
                return []
                
            prices = []
            for item in data:
                price = Price(
                    time=item.get("date"),
                    open=float(item.get("open", 0)),
                    close=float(item.get("close", 0)),
                    high=float(item.get("high", 0)),
                    low=float(item.get("low", 0)),
                    volume=0  # Forex typically doesn't have volume from Tiingo
                )
                prices.append(price)
            
            # Sort by date in descending order (newest first)
            prices.sort(key=lambda x: x.time, reverse=True)
            
            return prices
            
        except Exception as e:
            print(f"Tiingo error getting forex prices for {forex_pair}: {str(e)}")
            return []


def tiingo_to_df(prices: List[Price]) -> pd.DataFrame:
    """
    Convert Tiingo price data to a DataFrame.
    
    Args:
        prices: List of Price objects
        
    Returns:
        DataFrame with price data
    """
    df = pd.DataFrame([p.model_dump() for p in prices])
    
    if not df.empty:
        df["Date"] = pd.to_datetime(df["time"])
        df.set_index("Date", inplace=True)
        
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        df.sort_index(inplace=True)
    
    return df 