from enum import Enum
from typing import List, Optional, Dict, Any, Union

import pandas as pd

from data.models import (
    Price,
    FinancialMetrics,
    InsiderTrade,
    CompanyNews,
    LineItem
)

# Import data source clients
from tools.api import (
    get_prices as api_get_prices,
    get_financial_metrics as api_get_financial_metrics,
    search_line_items as api_search_line_items,
    get_insider_trades as api_get_insider_trades,
    get_company_news as api_get_company_news,
    get_market_cap as api_get_market_cap,
    prices_to_df as api_prices_to_df,
    get_intraday_prices as api_get_intraday_prices,
    get_technical_indicators as api_get_technical_indicators,
    get_economic_data as api_get_economic_data,
    get_forex_data as api_get_forex_data,
    get_short_interest as api_get_short_interest,
    get_sec_filings as api_get_sec_filings
)

from tools.yahoo_finance import YahooFinanceClient
# Import additional data source clients
from tools.alpha_vantage import AlphaVantageClient
from tools.financial_modeling_prep import FMPClient
from tools.fred import FredClient
from tools.tiingo import TiingoClient

# Create global instances of clients
_yahoo_client = YahooFinanceClient()
_alpha_vantage_client = None  # Will be initialized on demand
_fmp_client = None  # Will be initialized on demand
_fred_client = None  # Will be initialized on demand
_tiingo_client = None  # Will be initialized on demand

class DataSource(str, Enum):
    """Enum for supported data sources"""
    API = "api"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINANCIAL_MODELING_PREP = "fmp"
    FRED = "fred"
    TIINGO = "tiingo"


# Global setting for the default data source
_default_data_source = DataSource.API

def set_default_data_source(source: DataSource):
    """Set the default data source for all data fetching operations."""
    global _default_data_source
    _default_data_source = source

# Function to initialize the Alpha Vantage client on demand
def get_alpha_vantage_client():
    global _alpha_vantage_client
    if _alpha_vantage_client is None:
        _alpha_vantage_client = AlphaVantageClient()
    return _alpha_vantage_client

# Function to initialize the FMP client on demand
def get_fmp_client():
    global _fmp_client
    if _fmp_client is None:
        # Import here to avoid circular imports
        _fmp_client = FMPClient()
    return _fmp_client

# Function to initialize the FRED client on demand
def get_fred_client():
    global _fred_client
    if _fred_client is None:
        # Import here to avoid circular imports
        _fred_client = FredClient()
    return _fred_client

# Function to initialize the Tiingo client on demand
def get_tiingo_client():
    global _tiingo_client
    if _tiingo_client is None:
        # Import here to avoid circular imports
        _tiingo_client = TiingoClient()
    return _tiingo_client

# Function to determine best data source for a specific data type
def get_optimal_source_for_data_type(data_type: str, user_specified_source: Optional[DataSource] = None) -> DataSource:
    """
    Determine the optimal data source for a given data type based on availability and quality.
    User-specified source takes precedence if provided.
    
    Args:
        data_type: Type of data being requested
        user_specified_source: Optionally specified data source to use
        
    Returns:
        DataSource: The optimal data source to use
    """
    # User-specified source takes precedence
    if user_specified_source is not None:
        return user_specified_source
    
    # Default mappings based on data type and source quality
    data_type_to_source = {
        # Price data - Yahoo Finance is good for historical EOD, Alpha Vantage for intraday
        "historical_prices": DataSource.YAHOO_FINANCE,
        "intraday_prices": DataSource.ALPHA_VANTAGE,
        
        # Fundamental data - Yahoo for basic, FMP for detailed (when available)
        "financial_metrics": DataSource.YAHOO_FINANCE,
        "line_items": DataSource.YAHOO_FINANCE,
        "detailed_financials": DataSource.FINANCIAL_MODELING_PREP,
        "analyst_estimates": DataSource.FINANCIAL_MODELING_PREP,
        
        # Technical indicators - Alpha Vantage specializes in these
        "technical_indicators": DataSource.ALPHA_VANTAGE,
        
        # Market data
        "market_cap": DataSource.YAHOO_FINANCE,
        "short_interest": DataSource.FINANCIAL_MODELING_PREP,
        "institutional_ownership": DataSource.FINANCIAL_MODELING_PREP,
        
        # Alternative data
        "insider_trades": DataSource.YAHOO_FINANCE,
        "company_news": DataSource.TIINGO,  # Tiingo for news/sentiment
        "forex": DataSource.ALPHA_VANTAGE,
        "crypto": DataSource.YAHOO_FINANCE,
        
        # Economic data - FRED specializes in this
        "economic_data": DataSource.FRED,
        "recession_indicators": DataSource.FRED,
        "gdp": DataSource.FRED,
        "inflation": DataSource.FRED,
        "unemployment": DataSource.FRED,
        "interest_rates": DataSource.FRED,
        "yield_curve": DataSource.FRED,
    }
    
    # Get the optimal source for the data type
    if data_type in data_type_to_source:
        source = data_type_to_source[data_type]
        
        # Check if we need to fall back to API
        if source == DataSource.FINANCIAL_MODELING_PREP and get_fmp_client() is None:
            return DataSource.API
        elif source == DataSource.ALPHA_VANTAGE and get_alpha_vantage_client() is None:
            return DataSource.API
        elif source == DataSource.FRED and get_fred_client() is None:
            return DataSource.API
        elif source == DataSource.TIINGO and get_tiingo_client() is None:
            return DataSource.API
            
        return source
    
    # Default to API for anything not explicitly mapped
    return DataSource.API


def get_prices(
    ticker: str, 
    start_date: str, 
    end_date: str,
    data_source: Optional[DataSource] = None
) -> List[Price]:
    """
    Fetch price data from the selected data source.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_source: Optional data source override
        
    Returns:
        List of Price objects
    """
    source = data_source or _default_data_source
    
    if source == DataSource.API:
        return api_get_prices(ticker, start_date, end_date)
    elif source == DataSource.YAHOO_FINANCE:
        return _yahoo_client.get_prices(ticker, start_date, end_date)
    elif source == DataSource.ALPHA_VANTAGE:
        return get_alpha_vantage_client().get_prices(ticker, start_date, end_date)
    elif source == DataSource.FINANCIAL_MODELING_PREP:
        return get_fmp_client().get_prices(ticker, start_date, end_date)
    elif source == DataSource.TIINGO:
        return get_tiingo_client().get_prices(ticker, start_date, end_date)
    else:
        raise ValueError(f"Unsupported data source for prices: {source}")


def get_intraday_prices(
    ticker: str,
    interval: str = "5min",
    output_size: str = "full",
    data_source: Optional[DataSource] = None
) -> List[Price]:
    """
    Get intraday price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval between data points (e.g., "1min", "5min", "15min", "30min", "60min")
        output_size: Amount of data to return ("compact" or "full")
        data_source: Data source to use (optional)
        
    Returns:
        List of Price objects with intraday data
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("intraday_prices")
        
    source = data_source
    
    if source == DataSource.ALPHA_VANTAGE:
        av_client = get_alpha_vantage_client()
        if av_client:
            return av_client.get_intraday_prices(ticker, interval, output_size)
    elif source == DataSource.API:
        return api_get_intraday_prices(ticker, interval, output_size)
    
    # Fall back to API if source not implemented or failed
    from tools.api import get_intraday_prices as api_get_intraday_prices
    return api_get_intraday_prices(ticker, interval, output_size)


def get_technical_indicators(
    ticker: str,
    indicator: str,
    time_period: int = 14,
    series_type: str = "close",
    data_source: Optional[DataSource] = None
) -> Dict[str, Any]:
    """
    Get technical indicator data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        indicator: Technical indicator to calculate (e.g., "RSI", "MACD", "SMA", etc.)
        time_period: Time period to consider for the indicator (e.g., 14 days for RSI)
        series_type: Price series to use (e.g., "close", "open", "high", "low")
        data_source: Data source to use (optional)
        
    Returns:
        Dictionary with technical indicator data
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("technical_indicators")
        
    source = data_source
    
    if source == DataSource.ALPHA_VANTAGE:
        av_client = get_alpha_vantage_client()
        if av_client:
            return av_client.get_technical_indicators(ticker, indicator, time_period, series_type)
    elif source == DataSource.API:
        return api_get_technical_indicators(ticker, indicator, time_period, series_type)
    
    # Fall back to API if source not implemented or failed
    from tools.api import get_technical_indicators as api_get_technical_indicators
    return api_get_technical_indicators(ticker, indicator, time_period, series_type)


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    data_source: Optional[DataSource] = None
) -> List[FinancialMetrics]:
    """
    Fetch financial metrics from the selected data source.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        period: Period type (e.g., "ttm" for trailing twelve months)
        limit: Maximum number of results
        data_source: Optional data source override
        
    Returns:
        List of FinancialMetrics objects
    """
    # Decide source based on whether we need detailed fundamentals
    if data_source is None:
        data_source = get_optimal_source_for_data_type("detailed_financials")
    
    source = data_source
    
    if source == DataSource.API:
        return api_get_financial_metrics(ticker, end_date, period, limit)
    elif source == DataSource.YAHOO_FINANCE:
        return _yahoo_client.get_financial_metrics(ticker, end_date, period, limit)
    elif source == DataSource.FINANCIAL_MODELING_PREP:
        return get_fmp_client().get_financial_metrics(ticker, end_date, period, limit)
    else:
        raise ValueError(f"Unsupported data source for financial metrics: {source}")


def search_line_items(
    ticker: str,
    line_items: List[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    data_source: Optional[DataSource] = None
) -> List[LineItem]:
    """
    Search for financial line items from the selected data source.
    
    Args:
        ticker: Stock ticker symbol
        line_items: List of line item identifiers
        end_date: End date in YYYY-MM-DD format
        period: Period type
        limit: Maximum number of results
        data_source: Optional data source override
        
    Returns:
        List of LineItem objects
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("line_items")
    
    source = data_source
    
    if source == DataSource.API:
        return api_search_line_items(ticker, line_items, end_date, period, limit)
    elif source == DataSource.YAHOO_FINANCE:
        return _yahoo_client.search_line_items(ticker, line_items, end_date, period, limit)
    elif source == DataSource.FINANCIAL_MODELING_PREP:
        return get_fmp_client().search_line_items(ticker, line_items, end_date, period, limit)
    else:
        raise ValueError(f"Unsupported data source for line items: {source}")


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
    data_source: Optional[DataSource] = None
) -> List[InsiderTrade]:
    """
    Fetch insider trades from the selected data source.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        start_date: Optional start date in YYYY-MM-DD format
        limit: Maximum number of results
        data_source: Optional data source override
        
    Returns:
        List of InsiderTrade objects
    """
    source = data_source or _default_data_source
    
    if source == DataSource.API:
        return api_get_insider_trades(ticker, end_date, start_date, limit)
    elif source == DataSource.YAHOO_FINANCE:
        return _yahoo_client.get_insider_trades(ticker, end_date, start_date, limit)
    elif source == DataSource.FINANCIAL_MODELING_PREP:
        return get_fmp_client().get_insider_trades(ticker, end_date, start_date, limit)
    else:
        raise ValueError(f"Unsupported data source for insider trades: {source}")


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
    data_source: Optional[DataSource] = None
) -> List[CompanyNews]:
    """
    Fetch company news from the selected data source.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        start_date: Optional start date in YYYY-MM-DD format
        limit: Maximum number of results
        data_source: Optional data source override
        
    Returns:
        List of CompanyNews objects
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("company_news")
    
    source = data_source
    
    if source == DataSource.API:
        return api_get_company_news(ticker, end_date, start_date, limit)
    elif source == DataSource.YAHOO_FINANCE:
        return _yahoo_client.get_company_news(ticker, end_date, start_date, limit)
    elif source == DataSource.TIINGO:
        return get_tiingo_client().get_company_news(ticker, end_date, start_date, limit)
    else:
        raise ValueError(f"Unsupported data source for company news: {source}")


def get_market_cap(
    ticker: str,
    end_date: str,
    data_source: Optional[DataSource] = None
) -> float:
    """
    Fetch market cap from the selected data source.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        data_source: Optional data source override
        
    Returns:
        Market capitalization value or None
    """
    source = data_source or _default_data_source
    
    if source == DataSource.API:
        return api_get_market_cap(ticker, end_date)
    elif source == DataSource.YAHOO_FINANCE:
        return _yahoo_client.get_market_cap(ticker, end_date)
    elif source == DataSource.FINANCIAL_MODELING_PREP:
        return get_fmp_client().get_market_cap(ticker, end_date)
    else:
        raise ValueError(f"Unsupported data source for market cap: {source}")


def get_analyst_estimates(
    ticker: str,
    end_date: str,
    data_source: Optional[DataSource] = None
) -> Dict[str, Any]:
    """
    Get analyst estimates for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        data_source: Data source to use (optional)
        
    Returns:
        Dictionary with analyst estimates
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("analyst_estimates")
        
    source = data_source
    
    if source == DataSource.FINANCIAL_MODELING_PREP:
        fmp_client = get_fmp_client()
        if fmp_client:
            return fmp_client.get_analyst_estimates(ticker, end_date)
    elif source == DataSource.API:
        from tools.api import get_analyst_estimates as api_get_analyst_estimates
        return api_get_analyst_estimates(ticker, end_date)
    
    # Fall back to API if source not implemented or failed
    from tools.api import get_analyst_estimates as api_get_analyst_estimates
    return api_get_analyst_estimates(ticker, end_date)


def get_economic_data(
    indicator: str,
    start_date: str,
    end_date: str,
    data_source: Optional[DataSource] = None
) -> Dict[str, Any]:
    """
    Get economic indicator data.
    
    Args:
        indicator: Economic indicator code (e.g., "GDP", "UNRATE" for unemployment rate)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_source: Data source to use (optional)
        
    Returns:
        Dictionary with economic data
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("economic_data")
        
    source = data_source
    
    if source == DataSource.FRED:
        fred_client = get_fred_client()
        if fred_client:
            return fred_client.get_economic_data(indicator, start_date, end_date)
    elif source == DataSource.API:
        return api_get_economic_data(indicator, start_date, end_date)
    
    # Fall back to API if source not implemented or failed
    from tools.api import get_economic_data as api_get_economic_data
    return api_get_economic_data(indicator, start_date, end_date)


def get_short_interest(
    ticker: str,
    end_date: str,
    data_source: Optional[DataSource] = None
) -> Dict[str, Any]:
    """
    Get short interest data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        data_source: Data source to use (optional)
        
    Returns:
        Dictionary with short interest data
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("short_interest")
        
    source = data_source
    
    if source == DataSource.FINANCIAL_MODELING_PREP:
        fmp_client = get_fmp_client()
        if fmp_client:
            return fmp_client.get_short_interest(ticker, end_date)
    elif source == DataSource.YAHOO_FINANCE:
        # Yahoo Finance has limited short interest data in the info dictionary
        # but we can extract what's available
        yahoo_client = YahooFinanceClient()
        metrics = yahoo_client.get_financial_metrics(ticker, end_date)
        if metrics and metrics[0].short_percent_of_float is not None:
            return {
                "short_percent_of_float": metrics[0].short_percent_of_float,
                "short_ratio": metrics[0].short_ratio,
                "date": end_date
            }
    elif source == DataSource.API:
        return api_get_short_interest(ticker, end_date)
    
    # Fall back to API if source not implemented or failed
    from tools.api import get_short_interest as api_get_short_interest
    return api_get_short_interest(ticker, end_date)


def get_sec_filings(
    ticker: str,
    filing_type: str,
    limit: int = 10,
    data_source: Optional[DataSource] = None
) -> List[Dict[str, Any]]:
    """
    Get SEC filings for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        filing_type: Type of SEC filing (e.g., "10-K", "10-Q", "8-K", etc.)
        limit: Maximum number of filings to retrieve
        data_source: Data source to use (optional)
        
    Returns:
        List of dictionaries with SEC filings
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("sec_filings")
        
    source = data_source
    
    if source == DataSource.FINANCIAL_MODELING_PREP:
        fmp_client = get_fmp_client()
        if fmp_client:
            try:
                return fmp_client.get_sec_filings(ticker, filing_type, limit)
            except AttributeError:
                # Fall back to API if method not implemented
                pass
    elif source == DataSource.API:
        return api_get_sec_filings(ticker, filing_type, limit)
    
    # Fall back to API if source not implemented or failed
    from tools.api import get_sec_filings as api_get_sec_filings
    return api_get_sec_filings(ticker, filing_type, limit)


def prices_to_df(prices: List[Price]) -> pd.DataFrame:
    """
    Convert a list of Price objects to a pandas DataFrame.
    This function is the same regardless of the data source.
    
    Args:
        prices: List of Price objects
        
    Returns:
        DataFrame with price data
    """
    return api_prices_to_df(prices)


def get_price_data(
    ticker: str, 
    start_date: str, 
    end_date: str,
    data_source: Optional[DataSource] = None
) -> pd.DataFrame:
    """
    Convenience function to get price data as a DataFrame.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_source: Optional data source override
        
    Returns:
        DataFrame with price data
    """
    prices = get_prices(ticker, start_date, end_date, data_source)
    return prices_to_df(prices)


def get_forex_data(
    from_currency: str,
    to_currency: str,
    start_date: str,
    end_date: str,
    data_source: Optional[DataSource] = None
) -> List[Dict[str, Any]]:
    """
    Get foreign exchange (forex) data.
    
    Args:
        from_currency: From currency code (e.g., "USD")
        to_currency: To currency code (e.g., "EUR")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_source: Data source to use (optional)
        
    Returns:
        List of dictionaries with forex data
    """
    if data_source is None:
        data_source = get_optimal_source_for_data_type("forex")
        
    source = data_source
    
    if source == DataSource.ALPHA_VANTAGE:
        av_client = get_alpha_vantage_client()
        if av_client:
            # Alpha Vantage's forex function doesn't have date filtering in the API call
            # so we'll filter the results ourselves
            all_data = av_client.get_forex_data(from_currency, to_currency, "full")
            return [d for d in all_data if start_date <= d["date"] <= end_date]
    elif source == DataSource.API:
        return api_get_forex_data(from_currency, to_currency, start_date, end_date)
    
    # Fall back to API if source not implemented or failed
    from tools.api import get_forex_data as api_get_forex_data
    return api_get_forex_data(from_currency, to_currency, start_date, end_date) 