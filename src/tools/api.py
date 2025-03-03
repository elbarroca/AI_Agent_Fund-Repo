import os
import time
import requests
import pandas as pd
from pydantic import ValidationError

from data.models import (
    Price,
    PriceResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyNews,
    CompanyNewsResponse,
    BalanceSheetItem,
    BalanceSheetResponse,
    IncomeStatementItem,
    IncomeStatementResponse,
    CashFlowItem,
    CashFlowResponse
)

from data.cache import get_cache

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        try:
            # Filter cached data by date and limit
            filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x.report_period, reverse=True)
            if filtered_data:
                return filtered_data[:limit]
        except Exception as e:
            print(f"Warning: Error processing cached financial metrics: {str(e)}")
            # Continue to fetch from API if cache processing fails

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    # Add retry mechanism for rate limiting
    max_retries = 3
    delay_seconds = 2  # Start with 2 seconds delay
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
            response = requests.get(url, headers=headers)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', delay_seconds))
                print(f"Rate limited (429). Waiting {retry_after} seconds before retry. Attempt {attempt+1}/{max_retries}")
                time.sleep(retry_after)
                # Double the delay for next attempt (exponential backoff)
                delay_seconds *= 2
                continue
                
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                # Return empty list instead of raising exception
                return []
            
            # Preprocess the response data to ensure it matches our model
            response_data = response.json()
            
            # Ensure we have a proper structure for the metrics
            if "financial_metrics" not in response_data:
                print(f"Unexpected API response format - no financial_metrics field")
                return []
                
            metrics_list = []
            for metric in response_data["financial_metrics"]:
                # Ensure ticker is present
                if "ticker" not in metric:
                    metric["ticker"] = ticker
                    
                # Ensure report_period is present
                if "report_period" not in metric and "calendar_date" in metric:
                    metric["report_period"] = metric["calendar_date"]
                elif "report_period" not in metric:
                    # Skip metrics without report_period
                    continue
                    
                # Add defaults for other required fields if missing
                if "period" not in metric:
                    metric["period"] = period
                if "currency" not in metric:
                    metric["currency"] = "USD"
                    
                try:
                    # Try to create a FinancialMetrics object
                    metrics_list.append(FinancialMetrics(**metric))
                except Exception as e:
                    print(f"Error processing metric: {str(e)}")
                    # Skip this metric and continue
                    continue
                    
            if not metrics_list:
                print(f"No valid financial metrics found after processing")
                return []
                
            # Sort by report_period (newest first)
            metrics_list.sort(key=lambda x: x.report_period, reverse=True)
            
            # Cache the valid metrics
            _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics_list])
            
            return metrics_list[:limit]
            
        except Exception as e:
            print(f"Error fetching financial metrics from API (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(delay_seconds)
            # Double the delay for next attempt (exponential backoff)
            delay_seconds *= 2
    
    print(f"All attempts failed for financial metrics")
    return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Search financial line items with retry and improved error handling."""
    # Check cache first
    cache_key = f"{ticker}_{','.join(line_items)}_{period}"
    if cached_data := _cache.get_line_items(cache_key):
        try:
            # Filter cached data by date and limit
            filtered_data = [LineItem(**item) for item in cached_data if item["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x.report_period, reverse=True)
            if filtered_data:
                print(f"Retrieved {ticker} line items from cache, {len(filtered_data)} records")
                return filtered_data[:limit]
        except Exception as e:
            print(f"Warning: Error processing cached line items: {str(e)}")
            # Continue to API if cache processing fails

    # If not in cache, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    # Add retry mechanism for rate limiting
    max_retries = 3
    delay_seconds = 2  # Start with 2 seconds delay
    
    for attempt in range(max_retries):
        try:
            # Prepare line_items param for URL
            item_param = ",".join(line_items)
            url = f"https://api.financialdatasets.ai/search-line-items/?ticker={ticker}&line_items={item_param}&report_period_lte={end_date}&limit={limit}&period={period}"
            
            print(f"Attempt {attempt+1}: Fetching line items for {ticker} from API")
            response = requests.get(url, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', delay_seconds))
                print(f"Rate limited (429). Waiting {retry_after} seconds before retry. Attempt {attempt+1}/{max_retries}")
                time.sleep(retry_after)
                # Double the delay for next attempt (exponential backoff)
                delay_seconds *= 2
                continue
            
            # Handle 404 errors gracefully
            if response.status_code == 404:
                print(f"API Error: 404 Not Found - The requested line items couldn't be found")
                # Instead of returning empty, try to construct some basic line items from fundamental data
                return generate_fallback_line_items(ticker, line_items, end_date, period, limit)
                
            if response.status_code != 200:
                print(f"API Error searching line items: {response.status_code} - {response.text}")
                if attempt == max_retries - 1:  # Last attempt
                    return generate_fallback_line_items(ticker, line_items, end_date, period, limit)
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue

            # Preprocess the response data
            response_data = response.json()
            
            # Ensure we have a proper structure
            if "search_results" not in response_data:
                print(f"Unexpected API response format - no search_results field")
                if attempt == max_retries - 1:  # Last attempt
                    return generate_fallback_line_items(ticker, line_items, end_date, period, limit)
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue
                
            result_list = []
            for item in response_data["search_results"]:
                # Ensure required fields are present
                if "ticker" not in item:
                    item["ticker"] = ticker
                    
                # Ensure report_period is present
                if "report_period" not in item:
                    continue
                    
                # Add defaults for other required fields if missing
                if "period" not in item:
                    item["period"] = period
                if "currency" not in item:
                    item["currency"] = "USD"
                    
                try:
                    # Try to create a LineItem object
                    result_list.append(LineItem(**item))
                except Exception as e:
                    print(f"Error processing line item: {str(e)}")
                    continue
                    
            if not result_list:
                print(f"No valid line items found in API response")
                if attempt == max_retries - 1:  # Last attempt
                    return generate_fallback_line_items(ticker, line_items, end_date, period, limit)
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue
                
            # Sort by report_period (newest first)
            result_list.sort(key=lambda x: x.report_period, reverse=True)
            
            # Cache the results
            _cache.set_line_items(cache_key, [r.model_dump() for r in result_list])
            
            print(f"Successfully retrieved {len(result_list)} line items for {ticker}")
            return result_list[:limit]
            
        except Exception as e:
            print(f"Error fetching line items from API (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:  # Last attempt
                return generate_fallback_line_items(ticker, line_items, end_date, period, limit)
            time.sleep(delay_seconds)
            # Double the delay for next attempt (exponential backoff)
            delay_seconds *= 2
    
    # If we get here, all attempts failed, use fallback
    print(f"All attempts failed for line items")
    return generate_fallback_line_items(ticker, line_items, end_date, period, limit)


def generate_fallback_line_items(ticker: str, line_items: list[str], end_date: str, period: str, limit: int) -> list[LineItem]:
    """
    Generate fallback line items by fetching financial metrics and converting to line items.
    This is used when the API fails to return valid line items.
    """
    print(f"Generating fallback line items for {ticker} from financial metrics")
    
    try:
        # Fetch financial metrics as a fallback
        metrics = get_financial_metrics(ticker, end_date, period, limit)
        if not metrics:
            print(f"No financial metrics available for fallback")
            return []
            
        # Convert financial metrics to line items
        result_list = []
        
        # Build a map of common financial metrics to line item names
        metrics_to_line_items = {
            "revenue": ["revenue", "total_revenue"],
            "gross_margin": ["gross_margin", "gross_profit_margin"],
            "operating_margin": ["operating_margin", "operating_profit_margin"],
            "net_margin": ["net_margin", "profit_margin", "net_profit_margin"],
            "return_on_equity": ["roe", "return_on_equity"],
            "return_on_assets": ["roa", "return_on_assets"],
            "current_ratio": ["current_ratio"],
            "debt_to_equity": ["debt_to_equity", "d2e"],
            "price_to_earnings_ratio": ["pe_ratio", "price_to_earnings"],
            "price_to_book_ratio": ["pb_ratio", "price_to_book"],
            "price_to_sales_ratio": ["ps_ratio", "price_to_sales"],
            "earnings_per_share": ["eps", "earnings_per_share"],
            "market_cap": ["market_cap", "market_capitalization"]
        }
        
        # Match requested line items to metrics
        requested_metrics = set()
        for item in line_items:
            item_lower = item.lower().replace(" ", "_")
            for metric, variations in metrics_to_line_items.items():
                if item_lower in variations or any(var in item_lower for var in variations):
                    requested_metrics.add(metric)
        
        # Process each metric object
        for metric in metrics:
            # Create a line item for each reporting period
            line_item = LineItem(
                ticker=ticker,
                report_period=metric.report_period,
                period=metric.period,
                currency=metric.currency
            )
            
            # Add requested metrics as attributes
            for metric_name in requested_metrics:
                metric_value = getattr(metric, metric_name, None)
                if metric_value is not None:
                    # Convert metric name to standardized line item name
                    for line_item_name in metrics_to_line_items.get(metric_name, [metric_name]):
                        setattr(line_item, line_item_name, metric_value)
            
            # Only add line items that have at least one requested metric
            has_data = False
            for item in line_items:
                item_lower = item.lower().replace(" ", "_")
                if hasattr(line_item, item_lower) and getattr(line_item, item_lower) is not None:
                    has_data = True
                    break
            
            if has_data:
                result_list.append(line_item)
        
        # Cache the results
        if result_list:
            cache_key = f"{ticker}_{','.join(line_items)}_{period}"
            _cache.set_line_items(cache_key, [r.model_dump() for r in result_list])
            
        return result_list[:limit]
    
    except Exception as e:
        print(f"Error generating fallback line items: {str(e)}")
        return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news



def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    financial_metrics = get_financial_metrics(ticker, end_date)
    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)


def get_intraday_prices(
    ticker: str,
    interval: str = "5min",
    output_size: str = "full"
) -> list[Price]:
    """
    Fetch intraday price data using AlphaVantage.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval between data points (e.g., "1min", "5min", "15min", "30min", "60min")
        output_size: Amount of data to return ("compact" or "full")
        
    Returns:
        List of Price objects with intraday data
    """
    # Import here to avoid circular imports
    from tools.alpha_vantage import AlphaVantageClient
    
    # Create AlphaVantage client
    client = AlphaVantageClient()
    
    # Get intraday prices
    return client.get_intraday_prices(ticker, interval, output_size)


def get_technical_indicators(
    ticker: str,
    indicator: str,
    time_period: int = 14,
    series_type: str = "close"
) -> dict[str, any]:
    """
    Fetch technical indicator data using AlphaVantage.
    
    Args:
        ticker: Stock ticker symbol
        indicator: Technical indicator to calculate (e.g., "RSI", "MACD", "SMA", etc.)
        time_period: Time period to consider for the indicator (e.g., 14 days for RSI)
        series_type: Price series to use (e.g., "close", "open", "high", "low")
        
    Returns:
        Dictionary with technical indicator data
    """
    # Import here to avoid circular imports
    from tools.alpha_vantage import AlphaVantageClient
    
    # Create AlphaVantage client
    client = AlphaVantageClient()
    
    # Get technical indicators
    return client.get_technical_indicators(ticker, indicator, time_period, series_type)


def get_economic_data(
    indicator: str,
    start_date: str,
    end_date: str
) -> dict[str, any]:
    """
    Fetch economic data using AlphaVantage or a fallback source.
    
    Args:
        indicator: Economic indicator code (e.g., 'GDP', 'INFLATION', etc.)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary with economic indicator data
    """
    # For now, return a minimal implementation that doesn't use financial datasets API
    # This would ideally be implemented with FRED or another source
    return {
        "indicator": indicator,
        "data": [],
        "message": "Economic data functionality not fully implemented yet. Would use AlphaVantage or FRED."
    }


def get_forex_data(
    from_currency: str,
    to_currency: str,
    output_size: str = "full"
) -> list[dict[str, any]]:
    """
    Fetch forex (currency exchange) data using AlphaVantage.
    
    Args:
        from_currency: Base currency code (e.g., 'USD')
        to_currency: Target currency code (e.g., 'EUR')
        output_size: Amount of data to return ("compact" or "full")
        
    Returns:
        List of dictionary objects with forex data
    """
    # Import here to avoid circular imports
    from tools.alpha_vantage import AlphaVantageClient
    
    # Create AlphaVantage client
    client = AlphaVantageClient()
    
    # Get forex data
    return client.get_forex_data(from_currency, to_currency, output_size)


def get_short_interest(
    ticker: str,
    end_date: str
) -> dict[str, any]:
    """
    Fetch short interest data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary with short interest data
    """
    # This is a placeholder implementation since we're avoiding financial datasets API
    return {
        "ticker": ticker,
        "as_of_date": end_date,
        "short_interest": None,
        "short_interest_percent": None,
        "message": "Short interest data functionality not fully implemented. Would use a service like FinancialModelingPrep."
    }


def get_sec_filings(
    ticker: str,
    filing_type: str,
    limit: int = 10
) -> list[dict[str, any]]:
    """
    Fetch SEC filings for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        filing_type: Type of filing (e.g., '10-K', '10-Q', '8-K', etc.)
        limit: Maximum number of filings to return
        
    Returns:
        List of dictionaries with SEC filing data
    """
    # This is a placeholder implementation since we're avoiding financial datasets API
    return [
        {
            "ticker": ticker,
            "filing_type": filing_type,
            "filing_date": None,
            "report_url": None,
            "message": "SEC filings functionality not fully implemented. Would use a service like SEC Edgar or FinancialModelingPrep."
        }
    ]
