import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from data.models import (
    Price,
    FinancialMetrics,
    InsiderTrade,
    CompanyNews,
    LineItem
)

from data.cache import get_cache

# Global cache instance
_cache = get_cache()


class FMPClient:
    """Financial Modeling Prep API client for accessing detailed financial data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FMP client.
        
        Args:
            api_key: Financial Modeling Prep API key. If not provided, will check FMP_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """
        Fetch historical price data from FMP.
        
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
            filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
            if filtered_data:
                return filtered_data
                
        try:
            params = {
                "apikey": self.api_key,
                "from": start_date,
                "to": end_date
            }
            
            response = requests.get(f"{self.base_url}/historical-price-full/{ticker}", params=params)
            data = response.json()
            
            if "historical" not in data:
                return []
                
            prices = []
            for item in data["historical"]:
                price = Price(
                    time=item["date"],
                    open=float(item["open"]),
                    close=float(item["close"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    volume=int(item["volume"])
                )
                prices.append(price)
            
            # Sort by date in descending order (newest first)
            prices.sort(key=lambda x: x.time, reverse=True)
            
            # Cache the results
            _cache.set_prices(ticker, [p.model_dump() for p in prices])
            
            return prices
            
        except Exception as e:
            print(f"FMP error getting prices for {ticker}: {str(e)}")
            return []
    
    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[FinancialMetrics]:
        """
        Fetch detailed financial metrics from FMP.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date in YYYY-MM-DD format
            period: Period type (e.g., "ttm" for trailing twelve months)
            limit: Maximum number of results
            
        Returns:
            List of FinancialMetrics objects
        """
        # Check cache first
        if cached_data := _cache.get_financial_metrics(ticker):
            # Filter cached data by date and limit
            filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x.report_period, reverse=True)
            if filtered_data:
                return filtered_data[:limit]
                
        try:
            # Adjust period for FMP endpoints
            period_param = "quarter" if period == "ttm" else "annual"
            
            # Fetch key stats and ratios
            params = {"apikey": self.api_key, "limit": limit}
            
            # Get company profile for general info
            profile_response = requests.get(f"{self.base_url}/profile/{ticker}", params=params)
            profile_data = profile_response.json()
            
            if not profile_data or not isinstance(profile_data, list):
                return []
                
            profile = profile_data[0]
            
            # Get financial ratios
            ratios_response = requests.get(f"{self.base_url}/ratios/{ticker}", params=params)
            ratios_data = ratios_response.json()
            
            # Get income statement
            income_response = requests.get(f"{self.base_url}/income-statement/{ticker}", params=params)
            income_data = income_response.json()
            
            # Get balance sheet
            balance_response = requests.get(f"{self.base_url}/balance-sheet-statement/{ticker}", params=params)
            balance_data = balance_response.json()
            
            # Get cash flow statement
            cashflow_response = requests.get(f"{self.base_url}/cash-flow-statement/{ticker}", params=params)
            cashflow_data = cashflow_response.json()
            
            # Get key metrics
            metrics_response = requests.get(f"{self.base_url}/key-metrics/{ticker}", params=params)
            metrics_data = metrics_response.json()
            
            metrics_list = []
            
            for i in range(min(len(income_data), limit)):
                # Get the date from the income statement
                date = income_data[i]["date"] if i < len(income_data) else end_date
                
                # Skip if beyond end_date
                if date > end_date:
                    continue
                
                # Extract metrics from various endpoints
                metrics = FinancialMetrics(
                    ticker=ticker,
                    report_period=date,
                    market_cap=profile.get("mktCap"),
                    shares_outstanding=profile.get("sharesOutstanding"),
                    pe_ratio=profile.get("peRatio"),
                    price_to_book_ratio=profile.get("priceToBookRatio"),
                    price_to_sales_ratio=profile.get("priceToSalesRatio"),
                    price_to_earnings_ratio=profile.get("peRatio"),
                    earnings_per_share=profile.get("eps"),
                    dividend_yield=profile.get("lastDiv", 0) / profile.get("price", 1) * 100 if profile.get("price") else None,
                    beta=profile.get("beta"),
                    
                    # Income statement data
                    revenue=income_data[i].get("revenue") if i < len(income_data) else None,
                    gross_profit=income_data[i].get("grossProfit") if i < len(income_data) else None,
                    operating_income=income_data[i].get("operatingIncome") if i < len(income_data) else None,
                    net_income=income_data[i].get("netIncome") if i < len(income_data) else None,
                    
                    # Balance sheet data
                    total_assets=balance_data[i].get("totalAssets") if i < len(balance_data) else None,
                    total_liabilities=balance_data[i].get("totalLiabilities") if i < len(balance_data) else None,
                    
                    # Ratios
                    debt_to_equity=ratios_data[i].get("debtToEquityRatio") if i < len(ratios_data) else None,
                    profit_margin=ratios_data[i].get("netProfitMargin") * 100 if i < len(ratios_data) and ratios_data[i].get("netProfitMargin") else None,
                    operating_margin=ratios_data[i].get("operatingProfitMargin") * 100 if i < len(ratios_data) and ratios_data[i].get("operatingProfitMargin") else None,
                    return_on_equity=ratios_data[i].get("returnOnEquity") * 100 if i < len(ratios_data) and ratios_data[i].get("returnOnEquity") else None,
                    return_on_assets=ratios_data[i].get("returnOnAssets") * 100 if i < len(ratios_data) and ratios_data[i].get("returnOnAssets") else None,
                    
                    # Growth metrics from key metrics
                    revenue_growth=metrics_data[i].get("revenueGrowth") if i < len(metrics_data) else None,
                    earnings_growth=metrics_data[i].get("netIncomeGrowth") if i < len(metrics_data) else None,
                    
                    # Key metrics
                    free_cash_flow_per_share=metrics_data[i].get("freeCashFlowPerShare") if i < len(metrics_data) else None,
                    book_value_per_share=metrics_data[i].get("bookValuePerShare") if i < len(metrics_data) else None,
                    current_ratio=metrics_data[i].get("currentRatio") if i < len(metrics_data) else None,
                )
                
                metrics_list.append(metrics)
            
            if not metrics_list:
                return []
                
            # Cache the results
            _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics_list])
            return metrics_list
            
        except Exception as e:
            print(f"FMP error getting financial metrics for {ticker}: {str(e)}")
            return []
    
    def search_line_items(
        self,
        ticker: str,
        line_items: List[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[LineItem]:
        """
        Search for specific financial line items from FMP.
        
        Args:
            ticker: Stock ticker symbol
            line_items: List of line item identifiers
            end_date: End date in YYYY-MM-DD format
            period: Period type (e.g., "ttm" for trailing twelve months)
            limit: Maximum number of results
            
        Returns:
            List of LineItem objects
        """
        try:
            # Adjust period for FMP endpoints
            period_param = "quarter" if period == "ttm" else "annual"
            
            params = {"apikey": self.api_key, "limit": limit}
            
            # Get income statement
            income_response = requests.get(f"{self.base_url}/income-statement/{ticker}", params=params)
            income_data = income_response.json()
            
            # Get balance sheet
            balance_response = requests.get(f"{self.base_url}/balance-sheet-statement/{ticker}", params=params)
            balance_data = balance_response.json()
            
            # Get cash flow statement
            cashflow_response = requests.get(f"{self.base_url}/cash-flow-statement/{ticker}", params=params)
            cashflow_data = cashflow_response.json()
            
            # Get key metrics
            metrics_response = requests.get(f"{self.base_url}/key-metrics/{ticker}", params=params)
            metrics_data = metrics_response.json()
            
            results = []
            
            # Mapping of common line item names to FMP field names
            line_item_mapping = {
                "revenue": {"statement": "income", "field": "revenue"},
                "net_income": {"statement": "income", "field": "netIncome"},
                "operating_income": {"statement": "income", "field": "operatingIncome"},
                "gross_profit": {"statement": "income", "field": "grossProfit"},
                "total_assets": {"statement": "balance", "field": "totalAssets"},
                "total_liabilities": {"statement": "balance", "field": "totalLiabilities"},
                "cash_and_equivalents": {"statement": "balance", "field": "cashAndCashEquivalents"},
                "debt_to_equity": {"statement": "metrics", "field": "debtToEquityRatio"},
                "free_cash_flow": {"statement": "cashflow", "field": "freeCashFlow"},
                "capital_expenditure": {"statement": "cashflow", "field": "capitalExpenditure"},
                "depreciation_and_amortization": {"statement": "cashflow", "field": "depreciationAndAmortization"},
                "working_capital": {"statement": "balance", "field": "totalCurrentAssets", "subtract": "totalCurrentLiabilities"},
                "book_value_per_share": {"statement": "metrics", "field": "bookValuePerShare"},
                "earnings_per_share": {"statement": "income", "field": "eps"},
                "dividends_and_other_cash_distributions": {"statement": "cashflow", "field": "dividendPayout"},
                "outstanding_shares": {"statement": "metrics", "field": "sharesOutstanding"},
            }
            
            for item in line_items:
                found = False
                mapping = line_item_mapping.get(item)
                
                if mapping:
                    statement_type = mapping["statement"]
                    field = mapping["field"]
                    subtract_field = mapping.get("subtract")
                    
                    # Get the appropriate dataset based on statement type
                    if statement_type == "income":
                        dataset = income_data
                        name = "Income Statement"
                    elif statement_type == "balance":
                        dataset = balance_data
                        name = "Balance Sheet"
                    elif statement_type == "cashflow":
                        dataset = cashflow_data
                        name = "Cash Flow"
                    elif statement_type == "metrics":
                        dataset = metrics_data
                        name = "Key Metrics"
                    else:
                        dataset = []
                        name = "Unknown"
                    
                    for i, statement in enumerate(dataset):
                        if i >= limit:
                            break
                            
                        date = statement.get("date")
                        if date and date <= end_date:
                            value = statement.get(field)
                            
                            # If we need to subtract a field (e.g., for working capital)
                            if subtract_field and value is not None:
                                subtract_value = statement.get(subtract_field)
                                if subtract_value is not None:
                                    value = value - subtract_value
                            
                            line_item = LineItem(
                                ticker=ticker,
                                line_item=item,
                                report_period=date,
                                value=float(value) if value is not None else None,
                                unit="USD",
                                statement_type=name
                            )
                            
                            results.append(line_item)
                            found = True
                
                if not found:
                    # If we couldn't find the item, add a placeholder
                    line_item = LineItem(
                        ticker=ticker,
                        line_item=item,
                        report_period=end_date,
                        value=None,
                        unit="USD",
                        statement_type="Not Found"
                    )
                    results.append(line_item)
            
            return results
            
        except Exception as e:
            print(f"FMP error searching line items for {ticker}: {str(e)}")
            return []
    
    def get_market_cap(self, ticker: str, end_date: str) -> float:
        """
        Fetch market cap from FMP.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Market capitalization value or None
        """
        try:
            params = {"apikey": self.api_key}
            
            response = requests.get(f"{self.base_url}/profile/{ticker}", params=params)
            data = response.json()
            
            if not data or not isinstance(data, list):
                return None
                
            profile = data[0]
            
            return profile.get("mktCap")
            
        except Exception as e:
            print(f"FMP error getting market cap for {ticker}: {str(e)}")
            return None
    
    def get_analyst_estimates(self, ticker: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch analyst estimates from FMP.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with analyst estimates
        """
        try:
            params = {"apikey": self.api_key}
            
            response = requests.get(f"{self.base_url}/analyst-estimates/{ticker}", params=params)
            data = response.json()
            
            if not data:
                return {}
                
            # Extract the most recent estimate before end_date
            estimates = {}
            for estimate in data:
                if estimate["date"] <= end_date:
                    estimates = {
                        "date": estimate["date"],
                        "estimatedRevenue": estimate.get("estimatedRevenue"),
                        "estimatedEbitda": estimate.get("estimatedEbitda"),
                        "estimatedEbit": estimate.get("estimatedEbit"),
                        "estimatedNetIncome": estimate.get("estimatedNetIncome"),
                        "estimatedSga": estimate.get("estimatedSga"),
                        "estimatedEps": estimate.get("estimatedEps"),
                        "numberAnalystEstimatedRevenue": estimate.get("numberAnalystEstimatedRevenue"),
                        "numberAnalystsEstimatedEps": estimate.get("numberAnalystsEstimatedEps")
                    }
                    break
            
            return estimates
            
        except Exception as e:
            print(f"FMP error getting analyst estimates for {ticker}: {str(e)}")
            return {}
    
    def get_short_interest(self, ticker: str, end_date: str) -> Dict[str, Any]:
        """
        Fetch short interest data from FMP.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with short interest data
        """
        try:
            params = {"apikey": self.api_key}
            
            response = requests.get(f"{self.base_url}/short-interest/{ticker}", params=params)
            data = response.json()
            
            if not data:
                return {}
                
            # Find the most recent short interest data before end_date
            short_interest = {}
            for item in data:
                if item["date"] <= end_date:
                    short_interest = {
                        "date": item["date"],
                        "short_percent_of_float": item.get("shortPercentOfFloat"),
                        "short_ratio": item.get("shortRatio"),
                        "short_squeeze_rating": item.get("shortSqueezeRating"),
                        "short_interest": item.get("shortInterest")
                    }
                    break
            
            return short_interest
            
        except Exception as e:
            print(f"FMP error getting short interest data for {ticker}: {str(e)}")
            return {}
    
    def get_sec_filings(self, ticker: str, filing_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch SEC filings from FMP.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of SEC filing (e.g., "10-K", "10-Q", "8-K", etc.)
            limit: Maximum number of filings to retrieve
            
        Returns:
            List of dictionaries with SEC filings
        """
        try:
            params = {"apikey": self.api_key, "type": filing_type}
            
            response = requests.get(f"{self.base_url}/sec_filings/{ticker}", params=params)
            data = response.json()
            
            if not data:
                return []
                
            # Limit the number of results
            filings = data[:limit]
            
            return filings
            
        except Exception as e:
            print(f"FMP error getting SEC filings for {ticker}: {str(e)}")
            return []


def fmp_to_df(prices: List[Price]) -> pd.DataFrame:
    """
    Convert FMP price data to a DataFrame.
    
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