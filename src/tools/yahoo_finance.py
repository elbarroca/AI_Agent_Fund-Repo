import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

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


class YahooFinanceClient:
    """Yahoo Finance data client that matches API response structures"""
    
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """Fetch price data from Yahoo Finance."""
        # Check cache first
        if cached_data := _cache.get_prices(ticker):
            # Filter cached data by date range and convert to Price objects
            filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
            if filtered_data:
                return filtered_data
                
        try:
            data = yf.download(
                ticker,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d"),
                progress=False
            )
            
            if data.empty:
                return []
                
            prices = [
                Price(
                    time=str(date.date()),
                    open=float(row["Open"]),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"])
                ) for date, row in data.iterrows()
            ]
            
            # Cache the results as dicts
            _cache.set_prices(ticker, [p.model_dump() for p in prices])
            return prices
            
        except Exception as e:
            print(f"Yahoo Finance error getting prices for {ticker}: {str(e)}")
            return []

    def get_financial_metrics(
        self, 
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[FinancialMetrics]:
        """Fetch financial metrics from Yahoo Finance."""
        # Check cache first
        if cached_data := _cache.get_financial_metrics(ticker):
            # Filter cached data by date and limit
            filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x.report_period, reverse=True)
            if filtered_data:
                return filtered_data[:limit]
                
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get quarterly financials if available
            quarterly_financials = stock.quarterly_financials
            annual_financials = stock.financials
            
            # Use most recent available financials
            financials = quarterly_financials if not quarterly_financials.empty else annual_financials
            
            # Parse balance sheet data
            balance_sheet = stock.balance_sheet
            
            # Create a list to store metrics for different periods
            metrics_list = []
            
            # Add current period metrics
            current_metrics = FinancialMetrics(
                ticker=ticker,
                report_period=end_date,
                market_cap=info.get("marketCap"),
                shares_outstanding=info.get("sharesOutstanding"),
                pe_ratio=info.get("trailingPE"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                pb_ratio=info.get("priceToBook"),
                eps=info.get("trailingEps"),
                dividend_yield=info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
                beta=info.get("beta"),
                enterprise_value=info.get("enterpriseValue"),
                peg_ratio=info.get("pegRatio"),
                revenue=financials.get("Total Revenue", pd.Series()).iloc[0] if not financials.empty else None,
                gross_profit=financials.get("Gross Profit", pd.Series()).iloc[0] if not financials.empty else None,
                operating_income=financials.get("Operating Income", pd.Series()).iloc[0] if not financials.empty else None,
                net_income=financials.get("Net Income", pd.Series()).iloc[0] if not financials.empty else None,
                total_assets=balance_sheet.get("Total Assets", pd.Series()).iloc[0] if not balance_sheet.empty else None,
                total_liabilities=balance_sheet.get("Total Liabilities Net Minority Interest", pd.Series()).iloc[0] if not balance_sheet.empty else None,
                debt_to_equity=info.get("debtToEquity"),
                profit_margin=info.get("profitMargins") * 100 if info.get("profitMargins") else None,
                return_on_equity=info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None,
                return_on_assets=info.get("returnOnAssets") * 100 if info.get("returnOnAssets") else None,
            )
            
            metrics_list.append(current_metrics)
            
            # If we have historical quarterly data, add those too
            if not quarterly_financials.empty and len(quarterly_financials.columns) > 1:
                for i, col in enumerate(quarterly_financials.columns[1:], 1):
                    if i >= limit:
                        break
                        
                    period_date = str(col.date())
                    if period_date > end_date:
                        continue
                        
                    # Create metrics for historical periods
                    historical_metrics = FinancialMetrics(
                        ticker=ticker,
                        report_period=period_date,
                        revenue=quarterly_financials.get("Total Revenue", pd.Series()).iloc[i] if "Total Revenue" in quarterly_financials else None,
                        net_income=quarterly_financials.get("Net Income", pd.Series()).iloc[i] if "Net Income" in quarterly_financials else None,
                        # Add other metrics as available
                    )
                    
                    metrics_list.append(historical_metrics)
            
            if not metrics_list:
                return []
                
            # Cache the results as dicts
            _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics_list])
            return metrics_list[:limit]
            
        except Exception as e:
            print(f"Yahoo Finance error getting financial metrics for {ticker}: {str(e)}")
            return []

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[InsiderTrade]:
        """Fetch insider trades from Yahoo Finance."""
        # Check cache first
        if cached_data := _cache.get_insider_trades(ticker):
            # Filter cached data by date range
            filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                            if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                            and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
            filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
            if filtered_data:
                return filtered_data
                
        try:
            stock = yf.Ticker(ticker)
            insider_trades_df = stock.insider_trades
            
            if insider_trades_df is None or insider_trades_df.empty:
                return []
                
            insider_trades = []
            
            for _, row in insider_trades_df.iterrows():
                transaction_date = str(row.get('Transaction Date', row.get('Date', datetime.now())).date())
                
                # Skip if outside date range
                if start_date and transaction_date < start_date:
                    continue
                if transaction_date > end_date:
                    continue
                    
                # Get the insider name
                insider_name = row.get('Insider', '')
                
                # Determine transaction type
                transaction_type = row.get('Transaction')
                shares = float(row.get('Shares', 0))
                is_purchase = transaction_type == 'Buy' or (transaction_type == 'Sale' and shares < 0)
                
                insider_trade = InsiderTrade(
                    ticker=ticker,
                    filing_date=transaction_date,  # Use transaction date as filing date
                    transaction_date=transaction_date,
                    insider_name=insider_name,
                    insider_title=row.get('Title', ''),
                    transaction_type='P' if is_purchase else 'S',  # P for purchase, S for sale
                    price=float(row.get('Price', 0)),
                    shares=abs(shares),
                    value=abs(float(row.get('Value', 0))),
                    shares_total=None,  # Yahoo doesn't provide total holdings
                )
                
                insider_trades.append(insider_trade)
                
                if len(insider_trades) >= limit:
                    break
            
            if not insider_trades:
                return []
                
            # Cache the results
            _cache.set_insider_trades(ticker, [trade.model_dump() for trade in insider_trades])
            return insider_trades
            
        except Exception as e:
            print(f"Yahoo Finance error getting insider trades for {ticker}: {str(e)}")
            return []

    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[CompanyNews]:
        """Fetch company news from Yahoo Finance."""
        # Check cache first
        if cached_data := _cache.get_company_news(ticker):
            # Filter cached data by date range
            filtered_data = [CompanyNews(**news) for news in cached_data 
                            if (start_date is None or news["date"] >= start_date)
                            and news["date"] <= end_date]
            filtered_data.sort(key=lambda x: x.date, reverse=True)
            if filtered_data:
                return filtered_data
                
        try:
            stock = yf.Ticker(ticker)
            news_data = stock.news
            
            if not news_data:
                return []
                
            company_news = []
            
            for news_item in news_data:
                # Convert timestamp to date string
                news_date = datetime.fromtimestamp(news_item.get('providerPublishTime', 0)).strftime("%Y-%m-%d")
                
                # Skip if outside date range
                if start_date and news_date < start_date:
                    continue
                if news_date > end_date:
                    continue
                    
                news = CompanyNews(
                    ticker=ticker,
                    date=news_date,
                    title=news_item.get('title', ''),
                    description=news_item.get('summary', ''),
                    source=news_item.get('publisher', ''),
                    url=news_item.get('link', ''),
                )
                
                company_news.append(news)
                
                if len(company_news) >= limit:
                    break
            
            if not company_news:
                return []
                
            # Cache the results
            _cache.set_company_news(ticker, [news.model_dump() for news in company_news])
            return company_news
            
        except Exception as e:
            print(f"Yahoo Finance error getting company news for {ticker}: {str(e)}")
            return []

    def search_line_items(
        self,
        ticker: str,
        line_items: List[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[LineItem]:
        """Search financial line items from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            results = []
            
            # Mapping of common line item names to Yahoo Finance column names
            line_item_mapping = {
                "revenue": "Total Revenue",
                "netIncome": "Net Income",
                "grossProfit": "Gross Profit",
                "operatingIncome": "Operating Income",
                "ebitda": "EBITDA",
                "totalAssets": "Total Assets",
                "totalLiabilities": "Total Liabilities Net Minority Interest",
                "totalEquity": "Total Equity Gross Minority Interest",
                "cashAndEquivalents": "Cash And Cash Equivalents",
                "freeCashFlow": "Free Cash Flow",
                # Add more mappings as needed
            }
            
            for item in line_items:
                # Try to find the item in our mapping
                yahoo_item = line_item_mapping.get(item, item)
                
                # Look for the item in each statement
                found = False
                
                for statement, name in [
                    (income_stmt, "Income Statement"),
                    (balance_sheet, "Balance Sheet"),
                    (cash_flow, "Cash Flow")
                ]:
                    if statement is not None and not statement.empty and yahoo_item in statement.index:
                        # Get the data for the specified line item
                        item_data = statement.loc[yahoo_item]
                        
                        # Convert to LineItem objects
                        for i, (date, value) in enumerate(item_data.items()):
                            if i >= limit:
                                break
                                
                            period_date = str(date.date())
                            if period_date > end_date:
                                continue
                                
                            line_item = LineItem(
                                ticker=ticker,
                                line_item=item,
                                report_period=period_date,
                                value=float(value) if not pd.isna(value) else None,
                                unit="USD",  # Assuming USD
                                statement_type=name
                            )
                            
                            results.append(line_item)
                        
                        found = True
                        break
                
                if not found:
                    # If we couldn't find the item, add a placeholder with None value
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
            print(f"Yahoo Finance error searching line items for {ticker}: {str(e)}")
            return []

    def get_market_cap(self, ticker: str, end_date: str) -> float:
        """Fetch market cap from Yahoo Finance."""
        try:
            financial_metrics = self.get_financial_metrics(ticker, end_date)
            if not financial_metrics:
                return None
                
            market_cap = financial_metrics[0].market_cap
            return market_cap
            
        except Exception as e:
            print(f"Yahoo Finance error getting market cap for {ticker}: {str(e)}")
            return None


# Helper function to convert to dataframe, similar to the API version
def prices_to_df(prices: List[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data and convert to dataframe."""
    client = YahooFinanceClient()
    prices = client.get_prices(ticker, start_date, end_date)
    return prices_to_df(prices) 