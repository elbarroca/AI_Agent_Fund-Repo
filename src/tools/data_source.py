from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta

import pandas as pd

from data.models import (
    Price,
    FinancialMetrics,
    InsiderTrade,
    CompanyNews,
    LineItem
)

# Import data cache
from data.cache import get_cache

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
from tools.newsapi_client import NewsAPIClient

# Debug flag for verbose logging
_DEBUG = True  # Set to True for debugging and development, False for production

# Create global instances of clients
_yahoo_client = None
_alpha_vantage_client = None
_fmp_client = None
_fred_client = None
_tiingo_client = None
_newsapi_client = None

# Initialize cache
_cache = get_cache()

# Function to initialize the Yahoo Finance client on demand
def get_yahoo_client():
    """Get or create a Yahoo Finance client instance."""
    global _yahoo_client
    try:
        if _yahoo_client is None:
            print(f"Initializing Yahoo Finance client...")
            from tools.yahoo_finance import YahooFinanceClient
            _yahoo_client = YahooFinanceClient()
            print(f"Yahoo Finance client initialized successfully")
        return _yahoo_client
    except Exception as e:
        print(f"Error initializing Yahoo Finance client: {str(e)}")
        return None

class DataSource(str, Enum):
    """Enum for supported data sources"""
    API = "api"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINANCIAL_MODELING_PREP = "fmp"
    FRED = "fred"
    TIINGO = "tiingo"
    NEWSAPI = "newsapi"
    WATERFALL = "waterfall"  # Waterfall data source that tries all available sources


# Global setting for the default data source
_default_data_source = DataSource.API

def set_default_data_source(source: DataSource):
    """Set the default data source for all data fetching operations."""
    global _default_data_source
    _default_data_source = source
    print(f"Default data source set to: {source.value}")

# Function to initialize the Alpha Vantage client on demand
def get_alpha_vantage_client():
    """Get or create an Alpha Vantage client instance."""
    global _alpha_vantage_client
    try:
        if _alpha_vantage_client is None:
            print(f"Initializing Alpha Vantage client...")
            from tools.alpha_vantage import AlphaVantageClient
            _alpha_vantage_client = AlphaVantageClient()
            print(f"Alpha Vantage client initialized successfully")
        return _alpha_vantage_client
    except Exception as e:
        print(f"Error initializing Alpha Vantage client: {str(e)}")
        return None

# Function to initialize the FMP client on demand
def get_fmp_client():
    """Get or create an FMP client instance."""
    global _fmp_client
    try:
        if _fmp_client is None:
            # Import here to avoid circular imports
            print(f"Initializing FMP client...")
            from tools.financial_modeling_prep import FMPClient
            _fmp_client = FMPClient()
            print(f"FMP client initialized successfully")
        return _fmp_client
    except Exception as e:
        print(f"Error initializing FMP client: {str(e)}")
        return None

# Function to initialize the FRED client on demand
def get_fred_client():
    """Get or create a FRED client instance."""
    global _fred_client
    try:
        if _fred_client is None:
            # Import here to avoid circular imports
            print(f"Initializing FRED client...")
            from tools.fred import FredClient
            _fred_client = FredClient()
            print(f"FRED client initialized successfully")
        return _fred_client
    except Exception as e:
        print(f"Error initializing FRED client: {str(e)}")
        return None

# Function to initialize the Tiingo client on demand
def get_tiingo_client():
    """Get or create a Tiingo client instance."""
    global _tiingo_client
    try:
        if _tiingo_client is None:
            print(f"Initializing Tiingo client...")
            from tools.tiingo import TiingoClient
            _tiingo_client = TiingoClient()
            print(f"Tiingo client initialized successfully")
        return _tiingo_client
    except Exception as e:
        print(f"Error initializing Tiingo client: {str(e)}")
        return None

# Function to initialize the NewsAPI client on demand
def get_newsapi_client():
    """Get or create a NewsAPI client instance."""
    global _newsapi_client
    try:
        if _newsapi_client is None:
            print("Initializing NewsAPI client...")
            from tools.newsapi_client import NewsAPIClient
            _newsapi_client = NewsAPIClient()
            print("NewsAPI client initialized successfully")
        return _newsapi_client
    except Exception as e:
        print(f"Error initializing NewsAPI client: {str(e)}")
        return None

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
    
    # Prioritized waterfall mappings with fallbacks for each data type
    data_type_source_priority = {
        # Stock price data - order of preference
        "prices": [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE, DataSource.TIINGO, DataSource.FINANCIAL_MODELING_PREP],
        "historical_prices": [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE, DataSource.TIINGO, DataSource.FINANCIAL_MODELING_PREP],
        "intraday_prices": [DataSource.ALPHA_VANTAGE, DataSource.TIINGO, DataSource.YAHOO_FINANCE],
        
        # Fundamental data
        "financial_metrics": [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.TIINGO],
        "line_items": [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.ALPHA_VANTAGE],
        "ratios": [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP],
        "market_cap": [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.ALPHA_VANTAGE],
        
        # News and sentiment data
        "company_news": [DataSource.NEWSAPI, DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP],
        "market_news": [DataSource.NEWSAPI, DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP],
        "news": [DataSource.NEWSAPI, DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP],
        
        # Technical analysis
        "technical_indicators": [DataSource.ALPHA_VANTAGE, DataSource.FINANCIAL_MODELING_PREP, DataSource.TIINGO],
        
        # Ownership and insiders
        "institutional_ownership": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE],
        "insider_trades": [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP],
        "short_interest": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE],
        
        # Alternative data
        "forex": [DataSource.ALPHA_VANTAGE, DataSource.TIINGO, DataSource.YAHOO_FINANCE],
        "crypto": [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE],
        
        # Economic data
        "economic_data": [DataSource.FRED, DataSource.ALPHA_VANTAGE, DataSource.FINANCIAL_MODELING_PREP],
        "recession_indicators": [DataSource.FRED, DataSource.ALPHA_VANTAGE],
        "gdp": [DataSource.FRED, DataSource.ALPHA_VANTAGE],
        "inflation": [DataSource.FRED, DataSource.ALPHA_VANTAGE],
        "unemployment": [DataSource.FRED, DataSource.ALPHA_VANTAGE],
        "interest_rates": [DataSource.FRED, DataSource.ALPHA_VANTAGE],
        "yield_curve": [DataSource.FRED, DataSource.ALPHA_VANTAGE],
        
        # SEC and filings data
        "sec_filings": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE],
        "detailed_financials": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE],
        "analyst_estimates": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE],
        
        # Portfolio and social data (custom implementations)
        "portfolio_holdings": [DataSource.API],
        "social_media": [DataSource.API],
        "agent_signals": [DataSource.API]
    }
    
    # If we have prioritized sources for this data type
    if data_type in data_type_source_priority:
        # Try each source in order
        for source in data_type_source_priority[data_type]:
            # Check if client is available
            if source == DataSource.YAHOO_FINANCE:
                yahoo_client = get_yahoo_client()
                if yahoo_client is not None:
                    return source
            elif source == DataSource.ALPHA_VANTAGE:
                alpha_client = get_alpha_vantage_client()
                if alpha_client is not None:
                    return source
            elif source == DataSource.FINANCIAL_MODELING_PREP:
                fmp_client = get_fmp_client()
                if fmp_client is not None:
                    return source
            elif source == DataSource.FRED:
                fred_client = get_fred_client()
                if fred_client is not None:
                    return source
            elif source == DataSource.TIINGO:
                tiingo_client = get_tiingo_client()
                if tiingo_client is not None:
                    return source
            elif source == DataSource.NEWSAPI:
                newsapi_client = get_newsapi_client()
                if newsapi_client is not None:
                    return source
            elif source == DataSource.API:
                return source
                
        # If no source was available, default to API as last resort
        return DataSource.API
        
    # For anything not mapped, try Yahoo Finance if available, then API
    yahoo_client = get_yahoo_client()
    if yahoo_client is not None:
        return DataSource.YAHOO_FINANCE
    return DataSource.API


def get_prices(
    ticker: str, 
    start_date: str, 
    end_date: str,
    data_source: Optional[DataSource] = None
) -> List[Price]:
    """
    Fetch price data from the selected data source, with waterfall fallback.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_source: Optional data source override
        
    Returns:
        List of Price objects
    """
    # Use global default source if none specified
    if data_source is None:
        data_source = get_optimal_source_for_data_type("prices")
        if _DEBUG:
            print(f"Selected optimal data source for prices: {data_source}")
    
    # For waterfall, we try each source in order of preference
    if data_source == DataSource.WATERFALL:
        # Order of preference for price data
        sources = [
            DataSource.YAHOO_FINANCE,
            DataSource.ALPHA_VANTAGE,
            DataSource.TIINGO,
            DataSource.FINANCIAL_MODELING_PREP,
            DataSource.API
        ]
        
        for source in sources:
            # Skip sources without available clients
            if source == DataSource.YAHOO_FINANCE:
                client = get_yahoo_client()
                if client is None:
                    if _DEBUG:
                        print(f"Skipping {source} - client not available")
                    continue
            elif source == DataSource.ALPHA_VANTAGE:
                client = get_alpha_vantage_client()
                if client is None:
                    if _DEBUG:
                        print(f"Skipping {source} - client not available")
                    continue
            elif source == DataSource.TIINGO:
                client = get_tiingo_client()
                if client is None:
                    if _DEBUG:
                        print(f"Skipping {source} - client not available")
                    continue
            elif source == DataSource.FINANCIAL_MODELING_PREP:
                client = get_fmp_client()
                if client is None:
                    if _DEBUG:
                        print(f"Skipping {source} - client not available")
                    continue
                
            try:
                if _DEBUG:
                    print(f"Trying to fetch prices from {source} for {ticker}")
                prices = get_prices(ticker, start_date, end_date, source)
                if prices:
                    if _DEBUG:
                        print(f"Successfully fetched {len(prices)} prices from {source}")
                    return prices
                else:
                    if _DEBUG:
                        print(f"No prices returned from {source}")
            except Exception as e:
                if _DEBUG:
                    print(f"Error fetching prices from {source}: {str(e)}")
                continue
        
        # If all sources failed, return empty list
        if _DEBUG:
            print(f"All sources failed to retrieve prices for {ticker}")
        return []
    
    # For specific sources
    try:
        if data_source == DataSource.API:
            try:
                if _DEBUG:
                    print(f"Fetching prices from API for {ticker}")
                prices = api_get_prices(ticker, start_date, end_date)
                if prices:
                    if _DEBUG:
                        print(f"Successfully fetched {len(prices)} prices from API")
                    return prices
                else:
                    if _DEBUG:
                        print(f"API returned no prices for {ticker}")
            except Exception as e:
                if _DEBUG:
                    print(f"Error fetching prices from API: {str(e)}")
                return []
                
        elif data_source == DataSource.YAHOO_FINANCE:
            try:
                yahoo_client = get_yahoo_client()
                if not yahoo_client:
                    if _DEBUG:
                        print(f"Yahoo Finance client not available")
                    return []
                    
                if _DEBUG:
                    print(f"Fetching prices from Yahoo Finance for {ticker}")
                prices = yahoo_client.get_prices(ticker, start_date, end_date)
                if prices:
                    if _DEBUG:
                        print(f"Successfully fetched {len(prices)} prices from Yahoo Finance")
                    return prices
                else:
                    if _DEBUG:
                        print(f"Yahoo Finance returned no prices for {ticker}")
            except Exception as e:
                if _DEBUG:
                    print(f"Error fetching prices from Yahoo Finance: {str(e)}")
                return []
                
        # Similar handling for other data sources...
        # (Add detailed error handling for other sources as needed)
                
        # If source not handled or failed, return empty list
        if _DEBUG:
            print(f"Unhandled or failed data source for prices: {data_source}")
        return []
        
    except Exception as e:
        if _DEBUG:
            print(f"Unexpected error fetching prices: {str(e)}")
        return []


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
    data_source: Optional[DataSource] = None,
    timeout: int = 30  # Add timeout parameter with default of 30 seconds
) -> List[FinancialMetrics]:
    """
    Get financial metrics for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        period: Period type ("ttm", "annual", "quarterly")
        limit: Maximum number of periods to return
        data_source: Data source to use (optional)
        timeout: Maximum time in seconds to wait for each data source (default: 30)
        
    Returns:
        List of FinancialMetrics objects
    """
    # Use global default if not specified
    if data_source is None:
        data_source = _default_data_source
        
    # If using WATERFALL, try multiple sources in priority order
    if data_source == DataSource.WATERFALL:
        print(f"Using WATERFALL approach for {ticker} financial metrics")
        # Get from our waterfall priority list
        sources_to_try = data_type_source_priority.get("financial_metrics", 
            [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.ALPHA_VANTAGE, DataSource.API])
        print(f"Sources to try for financial metrics (in order): {[s.value for s in sources_to_try]}")
    else:
        # Otherwise, use only the specified source
        sources_to_try = [data_source]
        print(f"Using single source for financial metrics: {data_source.value}")
    
    # Try each source in priority order
    errors = {}
    for source in sources_to_try:
        try:
            metrics = []
            print(f"Attempting to get {ticker} financial metrics from {source.value}...")
            
            # Use signal handler for timeout on Unix-based systems
            import signal
            from contextlib import contextmanager

            @contextmanager
            def timeout_handler(seconds):
                def handle_timeout(signum, frame):
                    raise TimeoutError(f"Data source timed out after {seconds} seconds")
                
                # Set the timeout handler
                if hasattr(signal, 'SIGALRM'):  # Only on Unix systems
                    original_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, handle_timeout)
                    signal.alarm(seconds)
                
                try:
                    yield
                finally:
                    # Reset the alarm and restore original handler
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, original_handler)
            
            try:
                # Only use timeout on Unix systems (macOS, Linux) that support SIGALRM
                if hasattr(signal, 'SIGALRM'):
                    with timeout_handler(timeout):
                        if source == DataSource.API:
                            metrics = api_get_financial_metrics(ticker, end_date, period, limit)
                        elif source == DataSource.YAHOO_FINANCE:
                            yahoo_client = get_yahoo_client()
                            if yahoo_client:
                                metrics = yahoo_client.get_financial_metrics(ticker, end_date, period, limit)
                            else:
                                print(f"Yahoo Finance client initialization failed")
                        elif source == DataSource.FINANCIAL_MODELING_PREP:
                            fmp_client = get_fmp_client()
                            if fmp_client:
                                metrics = fmp_client.get_financial_metrics(ticker, end_date, period, limit)
                            else:
                                print(f"FMP client not initialized")
                else:
                    # On Windows or other systems without SIGALRM, just try without timeout
                    if source == DataSource.API:
                        metrics = api_get_financial_metrics(ticker, end_date, period, limit)
                    elif source == DataSource.YAHOO_FINANCE:
                        yahoo_client = get_yahoo_client()
                        if yahoo_client:
                            metrics = yahoo_client.get_financial_metrics(ticker, end_date, period, limit)
                        else:
                            print(f"Yahoo Finance client initialization failed")
                    elif source == DataSource.FINANCIAL_MODELING_PREP:
                        fmp_client = get_fmp_client()
                        if fmp_client:
                            metrics = fmp_client.get_financial_metrics(ticker, end_date, period, limit)
                        else:
                            print(f"FMP client not initialized")
            except TimeoutError as te:
                errors[source.value] = f"Timed out after {timeout} seconds"
                print(f"{source.value} timed out for {ticker} financial metrics: {str(te)}")
                continue
            
            # If we got valid data, return it
            if metrics:
                print(f"SUCCESS: Retrieved {ticker} financial metrics from {source.value} ({len(metrics)} records)")
                return metrics
            else:
                print(f"No financial metrics returned from {source.value}")
        except Exception as e:
            errors[source.value] = str(e)
            print(f"{source.value} error getting financial metrics for {ticker}: {str(e)}")
            continue
    
    # If we get here, all sources failed
    error_details = ', '.join([f"{source}: {error}" for source, error in errors.items()])
    raise Exception(f"Failed to retrieve financial metrics for {ticker} from any source. Errors: {error_details}")


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    data_source: Optional[DataSource] = None
) -> list[LineItem]:
    """Fetch financial line items from specified data source, with waterfall fallback."""
    global _cache
    
    # Use global default source if none specified, with optimal source for line items
    if data_source is None:
        data_source = get_optimal_source_for_data_type("line_items")
        if _DEBUG:
            print(f"Selected optimal data source for line items: {data_source}")
    
    # Check cache first
    cache_key = f"{ticker}_{','.join(line_items)}_{period}"
    if cached_data := _cache.get_line_items(cache_key):
        try:
            # Filter cached data by date and limit
            filtered_data = [LineItem(**item) for item in cached_data if item["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x.report_period, reverse=True)
            if filtered_data:
                if _DEBUG:
                    print(f"Retrieved {len(filtered_data)} line items from cache")
                return filtered_data[:limit]
        except Exception as e:
            print(f"Warning: Error processing cached line items: {str(e)}")
            # Continue to fetch new data if cache processing fails
    
    # For waterfall, we try each source in order of preference
    if data_source == DataSource.WATERFALL:
        # Order of preference for line items
        sources = [
            DataSource.YAHOO_FINANCE,
            DataSource.FINANCIAL_MODELING_PREP,
            DataSource.ALPHA_VANTAGE,
            DataSource.API
        ]
        
        all_results = []
        errors = {}
        
        for source in sources:
            # Skip sources without available clients
            if source == DataSource.YAHOO_FINANCE:
                client = get_yahoo_client()
                if client is None:
                    if _DEBUG:
                        print(f"Skipping {source} - client not available")
                    continue
            elif source == DataSource.ALPHA_VANTAGE:
                client = get_alpha_vantage_client()
                if client is None:
                    if _DEBUG:
                        print(f"Skipping {source} - client not available")
                    continue
            elif source == DataSource.FINANCIAL_MODELING_PREP:
                client = get_fmp_client()
                if client is None:
                    if _DEBUG:
                        print(f"Skipping {source} - client not available")
                    continue
                
            try:
                if _DEBUG:
                    print(f"Trying to fetch line items from {source} for {ticker}")
                source_results = search_line_items(ticker, line_items, end_date, period, limit, source)
                
                if source_results:
                    if _DEBUG:
                        print(f"Successfully fetched {len(source_results)} line items from {source}")
                    
                    # Combine results gathered so far
                    all_results.extend(source_results)
                    
                    # If we've gathered enough unique data points, we can stop
                    if has_sufficient_line_items(all_results, line_items):
                        break
                else:
                    if _DEBUG:
                        print(f"No line items returned from {source}")
            except Exception as e:
                errors[source.value] = str(e)
                if _DEBUG:
                    print(f"Error fetching line items from {source}: {str(e)}")
                continue
        
        # Return combined results from all sources
        if all_results:
            # Deduplicate, sort, and limit results
            unique_results = {}
            for item in all_results:
                # Use a combination of period and report_period as a key
                key = f"{item.ticker}_{item.report_period}_{item.period}"
                if key not in unique_results:
                    unique_results[key] = item
                else:
                    # Merge attributes from both items
                    for attr_name, attr_value in item.model_dump().items():
                        if attr_value is not None and getattr(unique_results[key], attr_name, None) is None:
                            setattr(unique_results[key], attr_name, attr_value)
            
            combined_results = list(unique_results.values())
            combined_results.sort(key=lambda x: x.report_period, reverse=True)
            
            # Cache the combined results
            _cache.set_line_items(cache_key, [item.model_dump() for item in combined_results])
            
            return combined_results[:limit]
            
        # If we got here and have no results, raise a more helpful error
        if errors:
            error_details = ', '.join([f"{source}: {error}" for source, error in errors.items()])
            print(f"Warning: Failed to retrieve line items from any source. Errors: {error_details}")
            # Return empty list instead of raising exception
            return []
        else:
            return []
    
    # For specific sources
    try:
        if data_source == DataSource.API:
            try:
                if _DEBUG:
                    print(f"Fetching line items from API for {ticker}")
                line_item_results = api_search_line_items(ticker, line_items, end_date, period, limit)
                if line_item_results:
                    if _DEBUG:
                        print(f"Successfully fetched {len(line_item_results)} line items from API")
                    return line_item_results
                else:
                    if _DEBUG:
                        print(f"API returned no line items for {ticker}")
            except Exception as e:
                if _DEBUG:
                    print(f"Error fetching line items from API: {str(e)}")
                return []
                
        elif data_source == DataSource.YAHOO_FINANCE:
            try:
                yahoo_client = get_yahoo_client()
                if not yahoo_client:
                    if _DEBUG:
                        print(f"Yahoo Finance client not available")
                    return []
                    
                if _DEBUG:
                    print(f"Fetching line items from Yahoo Finance for {ticker}")
                line_item_results = yahoo_client.search_line_items(ticker, line_items, end_date, period, limit)
                if line_item_results:
                    if _DEBUG:
                        print(f"Successfully fetched {len(line_item_results)} line items from Yahoo Finance")
                    return line_item_results
                else:
                    if _DEBUG:
                        print(f"Yahoo Finance returned no line items for {ticker}")
            except Exception as e:
                if _DEBUG:
                    print(f"Error fetching line items from Yahoo Finance: {str(e)}")
                return []
                
        elif data_source == DataSource.FINANCIAL_MODELING_PREP:
            try:
                fmp_client = get_fmp_client()
                if not fmp_client:
                    if _DEBUG:
                        print(f"FMP client not available")
                    return []
                    
                if _DEBUG:
                    print(f"Fetching line items from FMP for {ticker}")
                line_item_results = fmp_client.search_line_items(ticker, line_items, end_date, period, limit)
                if line_item_results:
                    if _DEBUG:
                        print(f"Successfully fetched {len(line_item_results)} line items from FMP")
                    return line_item_results
                else:
                    if _DEBUG:
                        print(f"FMP returned no line items for {ticker}")
            except Exception as e:
                if _DEBUG:
                    print(f"Error fetching line items from FMP: {str(e)}")
                return []
        
        elif data_source == DataSource.ALPHA_VANTAGE:
            try:
                alpha_client = get_alpha_vantage_client()
                if not alpha_client:
                    if _DEBUG:
                        print(f"Alpha Vantage client not available")
                    return []
                    
                if _DEBUG:
                    print(f"Fetching line items from Alpha Vantage for {ticker}")
                line_item_results = alpha_client.search_line_items(ticker, line_items, end_date, period, limit)
                if line_item_results:
                    if _DEBUG:
                        print(f"Successfully fetched {len(line_item_results)} line items from Alpha Vantage")
                    return line_item_results
                else:
                    if _DEBUG:
                        print(f"Alpha Vantage returned no line items for {ticker}")
            except Exception as e:
                if _DEBUG:
                    print(f"Error fetching line items from Alpha Vantage: {str(e)}")
                return []
                
        # If source not handled or failed, return empty list
        if _DEBUG:
            print(f"Unhandled or failed data source for line items: {data_source}")
        return []
        
    except Exception as e:
        if _DEBUG:
            print(f"Unexpected error fetching line items: {str(e)}")
        return []

def has_sufficient_line_items(results: list[LineItem], requested_items: list[str]) -> bool:
    """
    Check if the results contain sufficient line items data.
    
    Args:
        results: List of LineItem objects
        requested_items: List of requested line item names
        
    Returns:
        bool: True if there's sufficient data, False otherwise
    """
    if not results:
        return False
        
    # Count how many of the requested items we found
    found_items = set()
    
    # Create standardized versions of requested items for matching
    standardized_requested = [item.lower().replace(' ', '_').replace('-', '_') for item in requested_items]
    
    # Create mapping of common financial metric aliases
    aliases = {
        'revenue': ['total_revenue', 'sales', 'total_sales'],
        'netincome': ['net_income', 'profit', 'net_profit', 'net_earnings'],
        'totalassets': ['total_assets', 'assets'],
        'totalliabilities': ['total_liabilities', 'liabilities'],
        'operatingincome': ['operating_income', 'income_from_operations', 'operating_profit'],
        'grossprofit': ['gross_profit', 'gross_margin'],
        'ebitda': ['ebit', 'earnings_before_interest_taxes_depreciation_amortization'],
        'eps': ['earnings_per_share', 'diluted_eps', 'basic_eps'],
        'pe_ratio': ['price_earnings_ratio', 'price_to_earnings'],
        'market_cap': ['marketcap', 'market_capitalization'],
        'book_value': ['book_value_per_share', 'bvps']
    }
    
    # Create reverse lookup from alias to standard name
    alias_to_standard = {}
    for standard, alias_list in aliases.items():
        for alias in alias_list:
            alias_to_standard[alias] = standard
        # Also map the standard name to itself
        alias_to_standard[standard] = standard
        
    for result in results:
        # Skip if this result doesn't have a proper report period
        if not hasattr(result, 'report_period') or not result.report_period:
            continue
            
        # Check each attribute of the result
        result_dict = result.model_dump()
        
        for field_name, field_value in result_dict.items():
            # Skip null values and metadata fields
            if field_value is None or field_name in ['ticker', 'period', 'report_period', 'currency']:
                continue
                
            # Standardize the field name
            field_std = field_name.lower().replace(' ', '_').replace('-', '_')
            
            # Check if this field directly matches any requested item
            if field_std in standardized_requested:
                found_items.add(field_std)
                continue
                
            # Check for matches via aliases
            found_through_alias = False
            for std_req in standardized_requested:
                # Get all possible aliases for this requested item
                req_aliases = aliases.get(std_req, [])
                
                # If the field matches any alias, count it as found
                if field_std in req_aliases:
                    found_items.add(std_req)
                    found_through_alias = True
                    break
                    
                # If the field is an alias pointing to a standard name that's requested
                if field_std in alias_to_standard and alias_to_standard[field_std] in standardized_requested:
                    found_items.add(alias_to_standard[field_std])
                    found_through_alias = True
                    break
                    
            if found_through_alias:
                continue
                
            # Check partial matches (e.g., "revenue_growth" matches "revenue")
            for std_req in standardized_requested:
                if std_req in field_std:
                    found_items.add(std_req)
                    break
    
    # If we found most of the requested items, that's sufficient
    threshold = 0.6  # 60% of requested items should be found
    sufficiency_ratio = len(found_items) / len(standardized_requested)
    
    return sufficiency_ratio >= threshold


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
    # Use global default if not specified
    if data_source is None:
        data_source = _default_data_source
        
    # If using WATERFALL, try multiple sources in priority order
    if data_source == DataSource.WATERFALL:
        print(f"Using WATERFALL approach for {ticker} insider trades")
        # Get from our waterfall priority list
        sources_to_try = data_type_source_priority.get("insider_trades", 
            [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API])
        print(f"Sources to try for insider trades (in order): {[s.value for s in sources_to_try]}")
    else:
        # Otherwise, use only the specified source
        sources_to_try = [data_source]
        print(f"Using single source for insider trades: {data_source.value}")
    
    # Try each source in priority order
    errors = {}
    for source in sources_to_try:
        try:
            trades = []
            print(f"Attempting to get {ticker} insider trades from {source.value}...")
            
            if source == DataSource.API:
                trades = api_get_insider_trades(ticker, end_date, start_date, limit)
            elif source == DataSource.YAHOO_FINANCE:
                try:
                    yahoo_client = get_yahoo_client()
                    if yahoo_client:
                        trades = yahoo_client.get_insider_trades(ticker, end_date, start_date, limit)
                    else:
                        print(f"Yahoo Finance client initialization failed")
                except Exception as e:
                    print(f"Error using Yahoo Finance client: {str(e)}")
            elif source == DataSource.FINANCIAL_MODELING_PREP:
                fmp_client = get_fmp_client()
                if fmp_client:
                    trades = fmp_client.get_insider_trades(ticker, end_date, start_date, limit)
                else:
                    print(f"FMP client not initialized")
            
            # If we got valid data, return it
            if trades:
                print(f"SUCCESS: Retrieved {ticker} insider trades from {source.value} ({len(trades)} records)")
                return trades
            else:
                print(f"No insider trades returned from {source.value}")
        except Exception as e:
            errors[source.value] = str(e)
            print(f"{source.value} error getting insider trades for {ticker}: {str(e)}")
            continue
    
    # If we get here, all sources failed - but this is not critical, return empty list
    print(f"Warning: Failed to retrieve insider trades for {ticker} from any source.")
    return []


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
    # Use global default if not specified
    if data_source is None:
        data_source = _default_data_source
        
    # Check if this is an economic keyword rather than a ticker
    economic_keywords = ["inflation", "gdp", "federal reserve", "interest rates", 
                         "economy", "recession", "unemployment", "monetary policy", 
                         "fiscal policy", "treasury", "fomc", "bond market", "yield curve"]
    
    is_economic_topic = ticker.lower() in economic_keywords or ' ' in ticker
    
    # If using WATERFALL, try multiple sources in priority order
    if data_source == DataSource.WATERFALL:
        print(f"Using WATERFALL approach for {ticker} news")
        # Get from our waterfall priority list
        if is_economic_topic:
            sources_to_try = data_type_source_priority.get("market_news", 
                [DataSource.NEWSAPI, DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API])
            print(f"Sources to try for market news (in order): {[s.value for s in sources_to_try]}")
        else:
            sources_to_try = data_type_source_priority.get("company_news", 
                [DataSource.NEWSAPI, DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API])
            print(f"Sources to try for company news (in order): {[s.value for s in sources_to_try]}")
    else:
        # Otherwise, use only the specified source
        sources_to_try = [data_source]
        print(f"Using single source for news: {data_source.value}")
    
    # Try each source in priority order
    errors = {}
    for source in sources_to_try:
        try:
            news = []
            print(f"Attempting to get {ticker} news from {source.value}...")
            
            if source == DataSource.API:
                news = api_get_company_news(ticker, end_date, start_date, limit)
            elif source == DataSource.YAHOO_FINANCE:
                try:
                    yahoo_client = get_yahoo_client()
                    if yahoo_client:
                        news = yahoo_client.get_company_news(ticker, end_date, start_date, limit)
                    else:
                        print(f"Yahoo Finance client initialization failed")
                except Exception as e:
                    print(f"Error using Yahoo Finance client: {str(e)}")
            elif source == DataSource.NEWSAPI:
                news_client = get_newsapi_client()
                if news_client:
                    news = news_client.get_company_news(ticker, end_date, start_date, limit)
                else:
                    print(f"NewsAPI client not initialized")
            elif source == DataSource.FINANCIAL_MODELING_PREP:
                fmp_client = get_fmp_client()
                if fmp_client:
                    news = fmp_client.get_company_news(ticker, end_date, start_date, limit)
                else:
                    print(f"FMP client not initialized")
            
            # If we got valid data, return it
            if news:
                print(f"SUCCESS: Retrieved {ticker} news from {source.value} ({len(news)} items)")
                return news
            else:
                print(f"No news returned from {source.value}")
        except Exception as e:
            errors[source.value] = str(e)
            print(f"{source.value} error getting news for {ticker}: {str(e)}")
            continue
    
    # If we get here, all sources failed - but this is not critical, return empty list
    print(f"Warning: Failed to retrieve news for {ticker} from any source.")
    return []


def get_market_cap(
    ticker: str,
    end_date: str,
    data_source: Optional[DataSource] = None,
    timeout: int = 30  # Add timeout parameter
) -> float:
    """
    Get market cap for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        data_source: Data source to use (optional)
        timeout: Maximum time in seconds to wait for each data source (default: 30)
        
    Returns:
        Market cap value as float
    """
    # Use global default if not specified
    if data_source is None:
        data_source = _default_data_source
        
    # If using WATERFALL, try multiple sources in priority order
    if data_source == DataSource.WATERFALL:
        # Get from our waterfall priority list
        sources_to_try = data_type_source_priority.get("market_cap", 
            [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.ALPHA_VANTAGE, DataSource.API])
    else:
        # Otherwise, use only the specified source
        sources_to_try = [data_source]
    
    # Try each source in priority order
    errors = {}
    for source in sources_to_try:
        try:
            market_cap = None
            
            # Use signal handler for timeout on Unix-based systems
            import signal
            from contextlib import contextmanager

            @contextmanager
            def timeout_handler(seconds):
                def handle_timeout(signum, frame):
                    raise TimeoutError(f"Data source timed out after {seconds} seconds")
                
                # Set the timeout handler
                if hasattr(signal, 'SIGALRM'):  # Only on Unix systems
                    original_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, handle_timeout)
                    signal.alarm(seconds)
                
                try:
                    yield
                finally:
                    # Reset the alarm and restore original handler
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, original_handler)
            
            try:
                # Only use timeout on Unix systems (macOS, Linux) that support SIGALRM
                if hasattr(signal, 'SIGALRM'):
                    with timeout_handler(timeout):
                        if source == DataSource.API:
                            market_cap = api_get_market_cap(ticker, end_date)
                        elif source == DataSource.YAHOO_FINANCE:
                            yahoo_client = get_yahoo_client()
                            if yahoo_client:
                                market_cap = yahoo_client.get_market_cap(ticker, end_date)
                        elif source == DataSource.FINANCIAL_MODELING_PREP:
                            fmp_client = get_fmp_client()
                            if fmp_client:
                                market_cap = fmp_client.get_market_cap(ticker, end_date)
                else:
                    # On Windows or other systems without SIGALRM, just try without timeout
                    if source == DataSource.API:
                        market_cap = api_get_market_cap(ticker, end_date)
                    elif source == DataSource.YAHOO_FINANCE:
                        yahoo_client = get_yahoo_client()
                        if yahoo_client:
                            market_cap = yahoo_client.get_market_cap(ticker, end_date)
                    elif source == DataSource.FINANCIAL_MODELING_PREP:
                        fmp_client = get_fmp_client()
                        if fmp_client:
                            market_cap = fmp_client.get_market_cap(ticker, end_date)
            except TimeoutError as te:
                errors[source.value] = f"Timed out after {timeout} seconds"
                print(f"{source.value} timed out for {ticker} market cap: {str(te)}")
                continue
            
            # If we got valid data, return it
            if market_cap is not None:
                return market_cap
        except Exception as e:
            errors[source.value] = str(e)
            continue
    
    # If we get here, all sources failed
    error_details = ', '.join([f"{source}: {error}" for source, error in errors.items()])
    print(f"Warning: Failed to retrieve market cap for {ticker}. Errors: {error_details}")
    return 0.0  # Return 0 instead of raising exception


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
    elif source == DataSource.YAHOO_FINANCE:
        # Fallback to Yahoo Finance basic info
        try:
            yahoo_metrics = _yahoo_client.get_financial_metrics(ticker, end_date)
            if yahoo_metrics and len(yahoo_metrics) > 0:
                metrics = yahoo_metrics[0].model_dump()
                return {
                    "price_target": metrics.get("target_price"),
                    "recommendation": metrics.get("recommendation", "Hold"),
                    "number_of_analysts": metrics.get("number_of_analysts", 0),
                }
            return {"price_target": None, "recommendation": "Hold", "number_of_analysts": 0}
        except Exception as e:
            print(f"Error getting analyst estimates from Yahoo Finance: {e}")
            return {"price_target": None, "recommendation": "Hold", "number_of_analysts": 0}
    
    # Return empty data if no sources are available
    return {"price_target": None, "recommendation": "Hold", "number_of_analysts": 0}


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
    
    # Import here to avoid circular imports
    from tools.api import get_economic_data as api_get_economic_data
    
    if source == DataSource.FRED:
        fred_client = get_fred_client()
        if fred_client:
            try:
                return fred_client.get_economic_data(indicator, start_date, end_date)
            except Exception as e:
                print(f"FRED error getting economic data: {e}")
                # Fall through to API
    
    # Fall back to API if source not implemented or failed
    try:
        return api_get_economic_data(indicator, start_date, end_date)
    except Exception as e:
        print(f"API error getting economic data: {e}")
        return {"indicator": indicator, "data": []}


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
        yahoo_client = get_yahoo_client()
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
    # Use global default if not specified
    if data_source is None:
        data_source = _default_data_source
        
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


# Default source priority for each data type
data_type_source_priority = {
    "prices": [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "historical_prices": [DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "intraday_prices": [DataSource.ALPHA_VANTAGE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "financial_metrics": [DataSource.FINANCIAL_MODELING_PREP, DataSource.ALPHA_VANTAGE, DataSource.YAHOO_FINANCE, DataSource.API],
    "market_cap": [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "company_news": [DataSource.NEWSAPI, DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE, DataSource.API],
    "technical_indicators": [DataSource.ALPHA_VANTAGE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "economic_data": [DataSource.FRED, DataSource.ALPHA_VANTAGE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "insider_trades": [DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "market_news": [DataSource.NEWSAPI, DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "news": [DataSource.NEWSAPI, DataSource.YAHOO_FINANCE, DataSource.FINANCIAL_MODELING_PREP, DataSource.API],
    "institutional_ownership": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE, DataSource.API],
    "short_interest": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE, DataSource.API],
    "sec_filings": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE, DataSource.API],
    "detailed_financials": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE, DataSource.API],
    "analyst_estimates": [DataSource.FINANCIAL_MODELING_PREP, DataSource.YAHOO_FINANCE, DataSource.API],
    "portfolio_holdings": [DataSource.API],
    "social_media": [DataSource.API],
    "agent_signals": [DataSource.API]
}

# General-purpose waterfall data retrieval function
def get_data_with_waterfall(
    data_type: str,
    *args,
    data_source: Optional[DataSource] = None,
    **kwargs
) -> Any:
    """
    Attempt to get data from the specified source, falling back to alternatives if needed.
    
    Args:
        data_type: Type of data to retrieve (e.g., 'prices', 'company_news')
        *args: Positional arguments to pass to the data retrieval function
        data_source: Optional preferred data source to try first
        **kwargs: Keyword arguments to pass to the data retrieval function
        
    Returns:
        The requested data from the first successful source
    """
    # Get the data retrieval function
    if data_type == "prices":
        data_function = get_prices
    elif data_type == "financial_metrics":
        data_function = get_financial_metrics
    elif data_type == "line_items":
        data_function = search_line_items
    elif data_type == "market_cap":
        data_function = get_market_cap
    elif data_type == "company_news":
        data_function = get_company_news
    elif data_type == "insider_trades":
        data_function = get_insider_trades
    elif data_type == "technical_indicators":
        data_function = get_technical_indicators
    elif data_type == "economic_data":
        data_function = get_economic_data
    elif data_type == "short_interest":
        data_function = get_short_interest
    elif data_type == "analyst_estimates":
        data_function = get_analyst_estimates
    elif data_type == "sec_filings":
        data_function = get_sec_filings
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    # Determine the sources to try
    if data_source:
        # If a source is specified, try it first, then the rest in order
        preferred_source = data_source
        other_sources = [s for s in data_type_source_priority.get(data_type, []) if s != preferred_source]
        sources_to_try = [preferred_source] + other_sources
    else:
        # Otherwise use the default fallback order
        sources_to_try = data_type_source_priority.get(data_type, [])
    
    # Add special handling for economic news keywords in company_news
    if data_type == "company_news" and len(args) > 0:
        ticker = args[0]
        economic_keywords = ["inflation", "gdp", "federal reserve", "interest rates", 
                            "economy", "recession", "unemployment", "monetary policy", 
                            "fiscal policy", "treasury", "fomc", "bond market", "yield curve"]
        
        is_economic_topic = ticker.lower() in economic_keywords or ' ' in ticker
        
        if is_economic_topic:
            # For economic topics, prioritize NewsAPI and skip sources that require valid tickers
            sources_to_try = [DataSource.NEWSAPI, DataSource.API]
    
    # Try each source in order
    errors = {}
    for source in sources_to_try:
        if _DEBUG:
            print(f"Trying to get {data_type} data from {source}")
        try:
            result = data_function(*args, data_source=source, **kwargs)
            if result:
                if _DEBUG:
                    print(f"Retrieved {data_type} data from {source}")
                return result
            else:
                errors[source] = "Returned empty result"
        except Exception as e:
            errors[source] = str(e)
            if _DEBUG:
                print(f"Error getting {data_type} data from {source}: {str(e)}")
            continue
    
    # If we're here, we failed to get data from any source
    # For company_news with economic keywords, try our mock data as a last resort
    if data_type == "company_news" and len(args) > 0:
        ticker = args[0]
        economic_keywords = ["inflation", "gdp", "federal reserve", "interest rates", 
                            "economy", "recession", "unemployment", "monetary policy", 
                            "fiscal policy", "treasury", "fomc", "bond market", "yield curve"]
        
        is_economic_topic = ticker.lower() in economic_keywords or ' ' in ticker
        
        if is_economic_topic:
            if _DEBUG:
                print(f"All sources failed for economic topic '{ticker}', using mock data")
            try:
                # Extract the necessary parameters from args
                end_date = args[1] if len(args) > 1 else datetime.now().strftime("%Y-%m-%d")
                start_date = args[2] if len(args) > 2 else None
                limit = kwargs.get("limit", 10)
                
                return generate_mock_economic_news(ticker, end_date, start_date, limit)
            except Exception as e:
                if _DEBUG:
                    print(f"Failed to generate mock economic news: {e}")
                errors["mock"] = str(e)
    
    # All sources failed, raise an error with details
    error_details = ", ".join([f"{s}: {e}" for s, e in errors.items()])
    raise Exception(f"Failed to retrieve {data_type} data from any source. Errors: {error_details}")


def generate_mock_economic_news(
    topic: str, 
    end_date: str, 
    start_date: Optional[str] = None,
    limit: int = 10
) -> List[CompanyNews]:
    """
    Generate mock news data for economic topics when actual APIs fail.
    This serves as a final fallback to ensure agents have at least some data to work with.
    
    Args:
        topic: Economic topic keyword (e.g., "Inflation", "Federal Reserve")
        end_date: End date in YYYY-MM-DD format
        start_date: Optional start date in YYYY-MM-DD format
        limit: Maximum number of results to generate
        
    Returns:
        List of CompanyNews objects
    """
    if _DEBUG:
        print(f"Generating mock economic news for {topic}")
    
    # Normalize the topic
    normalized_topic = topic.lower()
    
    # Parse dates
    end_date_obj = datetime.fromisoformat(end_date)
    if start_date:
        start_date_obj = datetime.fromisoformat(start_date)
    else:
        # Default to 30 days before end date
        start_date_obj = end_date_obj - timedelta(days=30)
    
    # Limit the number of articles to generate
    num_articles = min(limit, 10)  # Cap at 10 to avoid generating too much mock data
    
    # Generate dates between start and end date
    date_range = (end_date_obj - start_date_obj).days
    if date_range <= 0:
        date_range = 1
    
    # Template news data based on economic topics
    mock_news_templates = {
        "inflation": [
            {"title": "Inflation Rates Show Signs of Stability", "sentiment": "neutral"},
            {"title": "Consumer Price Index Rises Unexpectedly", "sentiment": "negative"},
            {"title": "Federal Reserve Responds to Inflation Concerns", "sentiment": "neutral"},
            {"title": "Markets React to Latest Inflation Data", "sentiment": "negative"},
            {"title": "Analysts Predict Inflation Decline in Coming Months", "sentiment": "positive"}
        ],
        "federal reserve": [
            {"title": "Federal Reserve Announces Interest Rate Decision", "sentiment": "neutral"},
            {"title": "Fed Chair Speaks on Monetary Policy Outlook", "sentiment": "neutral"},
            {"title": "Markets Anticipate Federal Reserve Meeting", "sentiment": "neutral"},
            {"title": "Federal Reserve Minutes Reveal Policy Discussions", "sentiment": "neutral"},
            {"title": "Economists React to Fed's Latest Announcements", "sentiment": "neutral"}
        ],
        "interest rates": [
            {"title": "Interest Rates Expected to Remain Steady", "sentiment": "neutral"},
            {"title": "Rising Interest Rates Impact Housing Market", "sentiment": "negative"},
            {"title": "Central Bank Signals Potential Rate Cuts", "sentiment": "positive"},
            {"title": "Bond Markets Respond to Interest Rate Outlook", "sentiment": "neutral"},
            {"title": "Interest Rate Policy and Economic Growth Analysis", "sentiment": "neutral"}
        ],
        "gdp": [
            {"title": "GDP Growth Exceeds Expectations in Latest Quarter", "sentiment": "positive"},
            {"title": "Economic Indicators Point to Slowing GDP", "sentiment": "negative"},
            {"title": "Analysts Revise GDP Forecasts for Coming Year", "sentiment": "neutral"},
            {"title": "Global GDP Trends and Domestic Implications", "sentiment": "neutral"},
            {"title": "GDP Report Reveals Sector-by-Sector Performance", "sentiment": "neutral"}
        ],
        "economy": [
            {"title": "Economic Outlook Remains Positive Despite Challenges", "sentiment": "positive"},
            {"title": "Leading Economic Indicators Signal Caution", "sentiment": "negative"},
            {"title": "Economy Shows Resilience in Face of Global Pressures", "sentiment": "positive"},
            {"title": "Economists Debate Direction of Economic Policy", "sentiment": "neutral"},
            {"title": "Consumer Spending Drives Economic Activity", "sentiment": "positive"}
        ],
        "default": [
            {"title": f"Analysis: Recent Developments in {topic}", "sentiment": "neutral"},
            {"title": f"Experts Weigh In On {topic} Trends", "sentiment": "neutral"},
            {"title": f"Market Implications of {topic} Changes", "sentiment": "neutral"},
            {"title": f"Historical Perspective on {topic}", "sentiment": "neutral"},
            {"title": f"Future Outlook: {topic} in Focus", "sentiment": "neutral"}
        ]
    }
    
    # Get appropriate templates or use default
    templates = mock_news_templates.get(normalized_topic, mock_news_templates["default"])
    
    # Generate mock news articles
    mock_news = []
    for i in range(num_articles):
        # Select a template and date
        template = templates[i % len(templates)]
        news_date = start_date_obj + timedelta(days=(i * date_range // num_articles))
        
        # Create news item
        news_item = CompanyNews(
            ticker=topic,
            title=template["title"],
            date=news_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            source="MockEconomicNews",
            url=f"https://example.com/economic-news/{normalized_topic.replace(' ', '-')}/{i}",
            author="AI Data Service",
            sentiment=template["sentiment"]
        )
        mock_news.append(news_item)
    
    return mock_news


def get_optimal_sources_for_agent(agent_name: str, agent_data_requirements: Dict[str, Any]) -> Dict[str, DataSource]:
    """
    Determine the optimal data sources for each of an agent's data requirements.
    
    Args:
        agent_name: Name of the agent
        agent_data_requirements: Dictionary containing the agent's data requirements
        
    Returns:
        Dictionary mapping data types to optimal data sources
    """
    # Get the agent's data needs
    if agent_name not in agent_data_requirements:
        raise ValueError(f"Unknown agent: {agent_name}")
        
    data_needs = agent_data_requirements[agent_name].get("data_needs", [])
    
    # Map each data need to its optimal source
    optimal_sources = {}
    for need in data_needs:
        # Handle special case of line items
        if need.startswith("line_items:"):
            data_type = "line_items"
        else:
            data_type = need
            
        # Get the optimal source for this data type
        optimal_sources[data_type] = get_optimal_source_for_data_type(data_type)
        
    return optimal_sources


def collect_data_for_agent(
    agent_name: str, 
    agent_data_requirements: Dict[str, Any],
    ticker: str,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """
    Collect all required data for an agent using the waterfall approach.
    
    Args:
        agent_name: Name of the agent
        agent_data_requirements: Dictionary containing the agent's data requirements
        ticker: Stock ticker
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary containing all the collected data
    """
    # Get the agent's data needs
    if agent_name not in agent_data_requirements:
        raise ValueError(f"Unknown agent: {agent_name}")
        
    data_needs = agent_data_requirements[agent_name].get("data_needs", [])
    
    # Get optimal sources for this agent
    optimal_sources = get_optimal_sources_for_agent(agent_name, agent_data_requirements)
    
    # Collect data for each need
    collected_data = {}
    for need in data_needs:
        try:
            # Handle special case of line items
            if need.startswith("line_items:"):
                data_type = "line_items"
                # Extract the fields from the need string
                fields = need.split(":", 1)[1].split(",")
                # Collect the data with specified fields
                data = get_data_with_waterfall(
                    data_type, 
                    ticker, 
                    fields,
                    end_date=end_date,
                    data_source=optimal_sources.get(data_type)
                )
            elif need == "prices" or need == "historical_prices":
                data = get_data_with_waterfall(
                    need, 
                    ticker, 
                    start_date, 
                    end_date,
                    data_source=optimal_sources.get(need)
                )
            elif need == "financial_metrics":
                data = get_data_with_waterfall(
                    need, 
                    ticker, 
                    end_date,
                    data_source=optimal_sources.get(need)
                )
            elif need == "market_cap":
                data = get_data_with_waterfall(
                    need, 
                    ticker, 
                    end_date,
                    data_source=optimal_sources.get(need)
                )
            elif need == "news":
                data = get_data_with_waterfall(
                    "company_news", 
                    ticker, 
                    end_date,
                    start_date=start_date,
                    data_source=optimal_sources.get(need)
                )
            elif need == "technical_indicators":
                # Collect multiple technical indicators
                indicators = ["RSI", "MACD", "SMA", "EMA", "BBANDS"]
                data = {}
                for indicator in indicators:
                    try:
                        data[indicator] = get_data_with_waterfall(
                            need, 
                            ticker, 
                            indicator,
                            data_source=optimal_sources.get(need)
                        )
                    except Exception as e:
                        if _DEBUG:
                            print(f"Error retrieving {indicator}: {str(e)}")
            elif need == "economic_data":
                # Collect multiple economic indicators
                indicators = ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS"]
                data = {}
                for indicator in indicators:
                    try:
                        data[indicator] = get_data_with_waterfall(
                            need, 
                            indicator, 
                            start_date, 
                            end_date,
                            data_source=optimal_sources.get(need)
                        )
                    except Exception as e:
                        if _DEBUG:
                            print(f"Error retrieving {indicator}: {str(e)}")
            elif need == "insider_trades":
                data = get_data_with_waterfall(
                    need, 
                    ticker, 
                    start_date, 
                    end_date,
                    data_source=optimal_sources.get(need)
                )
            elif need == "short_interest":
                data = get_data_with_waterfall(
                    need, 
                    ticker, 
                    end_date,
                    data_source=optimal_sources.get(need)
                )
            elif need == "sec_filings":
                # Get 10-K and 10-Q filings
                filing_types = ["10-K", "10-Q"]
                data = {}
                for filing_type in filing_types:
                    try:
                        data[filing_type] = get_data_with_waterfall(
                            need, 
                            ticker, 
                            filing_type,
                            data_source=optimal_sources.get(need)
                        )
                    except Exception as e:
                        if _DEBUG:
                            print(f"Error retrieving {filing_type} filings: {str(e)}")
            # Special case for portfolio_holdings and agent_signals which are mocked
            elif need in ["portfolio_holdings", "agent_signals", "social_media"]:
                data = mock_data_for_special_needs(need, ticker)
            else:
                if _DEBUG:
                    print(f"Warning: Unknown data need '{need}' for agent {agent_name}")
                continue
                
            # Store the collected data
            collected_data[need] = data
                
        except Exception as e:
            if _DEBUG:
                print(f"Error collecting {need} for {agent_name}: {str(e)}")
            collected_data[need] = None
    
    return collected_data


def mock_data_for_special_needs(need: str, ticker: str) -> Any:
    """
    Generate mock data for special needs that don't have real API implementations.
    
    Args:
        need: The special data need
        ticker: The ticker symbol
        
    Returns:
        Mock data for the special need
    """
    if need == "portfolio_holdings":
        return {
            "holdings": [ticker, "MSFT", "AMZN"],
            "allocation": {ticker: 0.33, "MSFT": 0.33, "AMZN": 0.34},
            "cash": 100000.0
        }
    elif need == "agent_signals":
        return {
            "warren_buffett": {"score": 7.5, "signal": "bullish", "confidence": 0.8},
            "cathie_wood": {"score": 8.2, "signal": "bullish", "confidence": 0.9},
            "jim_cramer": {"score": 6.0, "signal": "neutral", "confidence": 0.6},
            "technicals": {"score": 7.0, "signal": "bullish", "confidence": 0.7},
            "sentiment": {"score": 6.5, "signal": "neutral", "confidence": 0.65}
        }
    elif need == "social_media":
        return {
            "sentiment_score": 0.72,
            "mention_count": 1250,
            "positive_mentions": 850,
            "negative_mentions": 150,
            "neutral_mentions": 250
        }
    else:
        raise ValueError(f"Unknown special need: {need}")


def get_data_source_status() -> Dict[str, bool]:
    """
    Check which data sources are available.
    
    Returns:
        Dict mapping data source names to availability boolean
    """
    global _cache, _yahoo_client, _alpha_vantage_client, _fmp_client, _fred_client, _tiingo_client, _newsapi_client
    
    status = {
        "yahoo_finance": False,
        "alpha_vantage": False,
        "fmp": False,
        "fred": False,
        "tiingo": False,
        "newsapi": False,
        "api": True  # API is always considered available
    }
    
    # Check Yahoo Finance
    try:
        yahoo_client = get_yahoo_client()
        status["yahoo_finance"] = yahoo_client is not None
    except Exception:
        status["yahoo_finance"] = False
    
    # Check Alpha Vantage
    try:
        alpha_vantage_client = get_alpha_vantage_client()
        status["alpha_vantage"] = alpha_vantage_client is not None
    except Exception:
        status["alpha_vantage"] = False
        
    # Check FMP
    try:
        fmp_client = get_fmp_client()
        status["fmp"] = fmp_client is not None
    except Exception:
        status["fmp"] = False
        
    # Check FRED
    try:
        fred_client = get_fred_client()
        status["fred"] = fred_client is not None
    except Exception:
        status["fred"] = False
        
    # Check Tiingo
    try:
        tiingo_client = get_tiingo_client()
        status["tiingo"] = tiingo_client is not None
    except Exception:
        status["tiingo"] = False
        
    # Check NewsAPI
    try:
        newsapi_client = get_newsapi_client()
        status["newsapi"] = newsapi_client is not None
    except Exception:
        status["newsapi"] = False
        
    return status