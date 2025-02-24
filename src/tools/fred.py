import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from data.cache import get_cache

# Global cache instance
_cache = get_cache()


class FredClient:
    """Federal Reserve Economic Data (FRED) API client for accessing economic data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FRED client.
        
        Args:
            api_key: FRED API key. If not provided, will check FRED_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.base_url = "https://api.stlouisfed.org/fred"
        
    def get_economic_data(self, indicator: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch economic data from FRED.
        
        Args:
            indicator: Economic indicator (e.g., "GDP", "UNRATE", "CPI")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with economic data
        """
        # Check cache first
        cache_key = f"economic_data_{indicator}_{start_date}_{end_date}"
        if cached_data := _cache.get_custom(cache_key):
            return cached_data
                
        try:
            # Map common indicator names to FRED series IDs
            series_mapping = {
                "GDP": "GDP",  # Gross Domestic Product
                "RGDP": "GDPC1",  # Real Gross Domestic Product
                "CPI": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
                "CPIYOY": "CPIAUCSL",  # CPI Year-over-Year (calculated)
                "UNRATE": "UNRATE",  # Unemployment Rate
                "FEDFUNDS": "FEDFUNDS",  # Federal Funds Effective Rate
                "M2": "M2SL",  # M2 Money Stock
                "GS10": "GS10",  # 10-Year Treasury Constant Maturity Rate
                "GS2": "GS2",  # 2-Year Treasury Constant Maturity Rate
                "T10Y2Y": "T10Y2Y",  # 10-Year Treasury Minus 2-Year Treasury
                "T10Y3M": "T10Y3M",  # 10-Year Treasury Minus 3-Month Treasury
                "INDPRO": "INDPRO",  # Industrial Production Index
                "PCE": "PCEPI",  # Personal Consumption Expenditures Price Index
                "PCEYOY": "PCEPI",  # PCE Year-over-Year (calculated)
                "PAYEMS": "PAYEMS",  # All Employees, Total Nonfarm
                "HOUST": "HOUST",  # Housing Starts: Total: New Privately Owned Housing Units Started
                "RSAFS": "RSAFS",  # Retail Sales: Total
                "TOTCI": "TOTCI",  # Capacity Utilization: Total Industry
                "PERMIT": "PERMIT",  # New Private Housing Units Authorized by Building Permit
                "DEXUSEU": "DEXUSEU",  # USD/EUR Exchange Rate
                "DTWEXB": "DTWEXB",  # Trade Weighted U.S. Dollar Index
                "USREC": "USREC",  # NBER Recession Indicator
                "GFDEBTN": "GFDEBTN",  # Federal Debt: Total Public Debt
                "DEBT_TO_GDP": "GFDEGDQ188S",  # Federal Debt: Total Public Debt as Percent of GDP
                "MORTGAGE30US": "MORTGAGE30US",  # 30-Year Fixed Rate Mortgage Average
                "CSUSHPISA": "CSUSHPISA",  # S&P/Case-Shiller U.S. National Home Price Index
                "MICH": "UMCSENT",  # University of Michigan: Consumer Sentiment
                "ISM_MAN": "NAPM",  # ISM Manufacturing PMI
                "ISM_NONMAN": "NMFCI",  # ISM Non-Manufacturing Index
                "NFCI": "NFCI",  # Chicago Fed National Financial Conditions Index
                "VIXCLS": "VIXCLS",  # CBOE Volatility Index
                "SP500": "SP500",  # S&P 500
                "WILL5000PR": "WILL5000PR",  # Wilshire 5000 Total Market Index
                "AHETPI": "AHETPI",  # Average Hourly Earnings of All Employees
                "PCE_YOY": "PCEPI"  # Personal Consumption Expenditures Price Index YoY
            }
            
            # Get the FRED series ID
            series_id = series_mapping.get(indicator.upper(), indicator)
            
            # Format dates for the API
            try:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                
                # Format dates as YYYY-MM-DD
                observation_start = start_date_obj.strftime("%Y-%m-%d")
                observation_end = end_date_obj.strftime("%Y-%m-%d")
            except ValueError:
                # If date parsing fails, use the original strings
                observation_start = start_date
                observation_end = end_date
            
            params = {
                "api_key": self.api_key,
                "file_type": "json",
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "frequency": "m"  # Monthly data
            }
            
            # Make the API request
            response = requests.get(f"{self.base_url}/series/observations", params=params)
            
            if response.status_code != 200:
                print(f"FRED API error: {response.status_code} - {response.text}")
                return {}
                
            data = response.json()
            
            if "observations" not in data:
                return {}
                
            observations = data["observations"]
            
            # Process the data into a more usable format
            result = {
                "indicator": indicator,
                "series_id": series_id,
                "name": self.get_series_info(series_id).get("title", indicator),
                "data": []
            }
            
            for obs in observations:
                value = obs.get("value")
                if value and value != ".":
                    try:
                        value = float(value)
                    except ValueError:
                        value = None
                else:
                    value = None
                
                result["data"].append({
                    "date": obs.get("date"),
                    "value": value
                })
            
            # If it's a year-over-year indicator, calculate the values
            if indicator.upper() in ["CPIYOY", "PCEYOY"]:
                base_data = result["data"].copy()
                yoy_data = []
                
                for i, current in enumerate(base_data):
                    if i < 12 or current["value"] is None:  # Need at least 12 months of data
                        yoy_data.append({
                            "date": current["date"],
                            "value": None
                        })
                        continue
                    
                    year_ago = base_data[i - 12]
                    if year_ago["value"] is None or year_ago["value"] == 0:
                        yoy_value = None
                    else:
                        yoy_value = ((current["value"] / year_ago["value"]) - 1) * 100
                    
                    yoy_data.append({
                        "date": current["date"],
                        "value": yoy_value
                    })
                
                result["data"] = yoy_data
            
            # Cache the result
            _cache.set_custom(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"FRED error getting economic data for {indicator}: {str(e)}")
            return {}
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get information about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series information
        """
        try:
            params = {
                "api_key": self.api_key,
                "file_type": "json",
                "series_id": series_id
            }
            
            response = requests.get(f"{self.base_url}/series", params=params)
            
            if response.status_code != 200:
                return {}
                
            data = response.json()
            
            if "seriess" not in data or not data["seriess"]:
                return {}
                
            return data["seriess"][0]
            
        except Exception as e:
            print(f"FRED error getting series info for {series_id}: {str(e)}")
            return {}
    
    def get_category_children(self, category_id: int = 0) -> List[Dict[str, Any]]:
        """
        Get children categories for a FRED category.
        
        Args:
            category_id: FRED category ID (0 for the root category)
            
        Returns:
            List of dictionaries with category information
        """
        try:
            params = {
                "api_key": self.api_key,
                "file_type": "json",
                "category_id": category_id
            }
            
            response = requests.get(f"{self.base_url}/category/children", params=params)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            
            if "categories" not in data:
                return []
                
            return data["categories"]
            
        except Exception as e:
            print(f"FRED error getting category children for {category_id}: {str(e)}")
            return []
    
    def get_category_series(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Get series for a FRED category.
        
        Args:
            category_id: FRED category ID
            
        Returns:
            List of dictionaries with series information
        """
        try:
            params = {
                "api_key": self.api_key,
                "file_type": "json",
                "category_id": category_id
            }
            
            response = requests.get(f"{self.base_url}/category/series", params=params)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            
            if "seriess" not in data:
                return []
                
            return data["seriess"]
            
        except Exception as e:
            print(f"FRED error getting category series for {category_id}: {str(e)}")
            return []
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for FRED series.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with series information
        """
        try:
            params = {
                "api_key": self.api_key,
                "file_type": "json",
                "search_text": search_text,
                "limit": limit
            }
            
            response = requests.get(f"{self.base_url}/series/search", params=params)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            
            if "seriess" not in data:
                return []
                
            return data["seriess"]
            
        except Exception as e:
            print(f"FRED error searching series for {search_text}: {str(e)}")
            return []
    
    def get_macro_data(self, indicators: List[str], start_date: str, end_date: str) -> Dict[str, Dict[str, Any]]:
        """
        Fetch macroeconomic data for multiple indicators.
        
        Args:
            indicators: List of economic indicators
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping indicators to their data
        """
        results = {}
        
        for indicator in indicators:
            results[indicator] = self.get_economic_data(indicator, start_date, end_date)
        
        return results


def economic_data_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert economic data to a DataFrame.
    
    Args:
        data: Economic data from get_economic_data
        
    Returns:
        DataFrame with economic data
    """
    if not data or "data" not in data or not data["data"]:
        return pd.DataFrame()
    
    df = pd.DataFrame(data["data"])
    
    if not df.empty:
        df["Date"] = pd.to_datetime(df["date"])
        df.rename(columns={"value": data.get("indicator", "value")}, inplace=True)
        df.set_index("Date", inplace=True)
        df.drop(columns=["date"], inplace=True, errors="ignore")
        
        # Convert values to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df.sort_index(inplace=True)
    
    return df 