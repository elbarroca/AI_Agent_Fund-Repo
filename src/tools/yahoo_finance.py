import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from data.models import (
    Price,
    FinancialMetrics,
    InsiderTrade,
    CompanyNews,
    LineItem,
    BalanceSheetItem,
    IncomeStatementItem,
    CashFlowItem,
    AssetProfile
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
                # Add source attribution if missing
                for price in filtered_data:
                    if price.source == "unknown":
                        price.source = "yahoo_finance"
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
                
            prices = []
            # Convert DataFrame to list of Price objects
            for idx, row in data.iterrows():
                time_str = idx.strftime("%Y-%m-%d")
                price = Price(
                    ticker=ticker,
                    time=time_str,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    source="yahoo_finance"
                )
                prices.append(price)
                
            # Cache the results
            _cache.store_prices(ticker, [p.model_dump() for p in prices])
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
            
            # Helper function to safely get DataFrame values
            def safe_get_value(df, key):
                if df is None or df.empty or key not in df:
                    return None
                series = df.get(key)
                if series is None or len(series) == 0:
                    return None
                return series.iloc[0]
            
            # Calculate some financial ratios
            total_assets = safe_get_value(balance_sheet, "Total Assets")
            total_liabilities = safe_get_value(balance_sheet, "Total Liabilities Net Minority Interest")
            revenue = safe_get_value(financials, "Total Revenue")
            net_income = safe_get_value(financials, "Net Income")
            
            # Initialize with default values for required fields
            market_cap = info.get("marketCap")
            
            # Add current period metrics
            current_metrics = FinancialMetrics(
                ticker=ticker,
                calendar_date=end_date,  # Use the provided end date as calendar date
                report_period=end_date,
                period=period,
                currency="USD",  # Assuming USD as default currency
                
                # Market and valuation metrics
                market_cap=market_cap,
                enterprise_value=info.get("enterpriseValue"),
                price_to_earnings_ratio=info.get("trailingPE"),
                price_to_book_ratio=info.get("priceToBook"),
                price_to_sales_ratio=info.get("priceToSalesTrailing12Months"),
                enterprise_value_to_ebitda_ratio=None,  # Not directly available from yfinance
                enterprise_value_to_revenue_ratio=None if not revenue or not info.get("enterpriseValue") else info.get("enterpriseValue") / revenue,
                free_cash_flow_yield=None,  # Calculate if needed
                peg_ratio=info.get("pegRatio"),
                
                # Profitability metrics
                gross_margin=info.get("grossMargins", 0) * 100 if info.get("grossMargins") else None,
                operating_margin=info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else None,
                net_margin=info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
                
                # Return metrics
                return_on_equity=info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
                return_on_assets=info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else None,
                return_on_invested_capital=None,  # Not directly available
                
                # Efficiency metrics
                asset_turnover=None if not total_assets or not revenue else revenue / total_assets,
                inventory_turnover=None,  # Would need COGS and average inventory
                receivables_turnover=None,  # Would need average receivables
                days_sales_outstanding=None,  # Would need average receivables and daily revenue
                operating_cycle=None,  # Would need inventory days and receivable days
                working_capital_turnover=None,  # Would need working capital
                
                # Liquidity metrics
                current_ratio=info.get("currentRatio"),
                quick_ratio=info.get("quickRatio"),
                cash_ratio=None,  # Would need cash and current liabilities
                operating_cash_flow_ratio=None,  # Would need operating cash flow and current liabilities
                
                # Solvency metrics
                debt_to_equity=info.get("debtToEquity"),
                debt_to_assets=None if not total_assets else (total_liabilities / total_assets if total_liabilities else None),
                interest_coverage=None,  # Would need EBIT and interest expense
                
                # Growth metrics
                revenue_growth=info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None,
                earnings_growth=info.get("earningsGrowth", 0) * 100 if info.get("earningsGrowth") else None,
                book_value_growth=None,  # Not directly available
                earnings_per_share_growth=None,  # Would need historical EPS
                free_cash_flow_growth=None,  # Would need historical FCF
                operating_income_growth=None,  # Would need historical operating income
                ebitda_growth=None,  # Would need historical EBITDA
                
                # Per share metrics
                payout_ratio=info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else None,
                earnings_per_share=info.get("trailingEps"),
                book_value_per_share=None,  # Would need book value and shares outstanding
                free_cash_flow_per_share=None,  # Would need FCF and shares outstanding
            )
            
            metrics_list.append(current_metrics)
            
            # Cache the results as dicts
            _cache.set_financial_metrics(ticker, [m.model_dump() for m in metrics_list])
            return metrics_list
            
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
                # Add source attribution if missing
                for trade in filtered_data:
                    if trade.source == "unknown":
                        trade.source = "yahoo_finance"
                return filtered_data
                
        try:
            stock = yf.Ticker(ticker)
            
            # Check if insider_trades attribute exists and handle gracefully
            if not hasattr(stock, 'insider_trades'):
                print(f"Yahoo Finance: insider_trades attribute not available for {ticker}")
                return []
            
            insider_trades_df = stock.insider_trades
            
            if insider_trades_df is None or insider_trades_df.empty:
                return []
                
            insider_trades = []
            
            for _, row in insider_trades_df.iterrows():
                # Safely get transaction date with fallback to filing date
                transaction_date = None
                if 'Transaction Date' in row:
                    transaction_date = row['Transaction Date']
                elif 'Date' in row:
                    transaction_date = row['Date']
                
                if transaction_date is not None:
                    if hasattr(transaction_date, 'date'):
                        transaction_date = str(transaction_date.date())
                    else:
                        transaction_date = str(transaction_date)
                
                # Skip if outside date range
                if transaction_date and ((start_date and transaction_date < start_date) or transaction_date > end_date):
                    continue
                    
                # Get values safely
                shares = 0
                if 'Shares' in row:
                    shares = float(row['Shares']) if not pd.isna(row['Shares']) else 0
                
                value = 0
                if 'Value' in row:
                    value = float(row['Value']) if not pd.isna(row['Value']) else 0
                
                insider_trade = InsiderTrade(
                    ticker=ticker,
                    issuer=ticker,
                    name=str(row.get('Insider', 'Unknown')),
                    title=str(row.get('Title', 'Unknown')),
                    is_board_director=None,
                    transaction_date=transaction_date,
                    transaction_shares=shares,
                    transaction_price_per_share=None,
                    transaction_value=value,
                    shares_owned_before_transaction=None,
                    shares_owned_after_transaction=None,
                    security_title=None,
                    filing_date=str(row.get('Filing Date', datetime.now().date())),
                    source="yahoo_finance"  # Add source attribution
                )
                insider_trades.append(insider_trade)
                
                if len(insider_trades) >= limit:
                    break
            
            # Cache the results as dicts
            _cache.set_insider_trades(ticker, [t.model_dump() for t in insider_trades])
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
        try:
            if cached_data := _cache.get_company_news(ticker):
                # Filter cached data by date range
                filtered_data = [CompanyNews(**news) for news in cached_data 
                                if (start_date is None or news["date"] >= start_date)
                                and news["date"] <= end_date]
                filtered_data.sort(key=lambda x: x.date, reverse=True)
                if filtered_data:
                    return filtered_data[:limit]
        except Exception as e:
            print(f"Cache error in Yahoo Finance get_company_news: {e}")
            # Continue to fetch new data if cache fails
                
        try:
            stock = yf.Ticker(ticker)
            news_items = []
            
            # First try the news attribute which gives more detailed information
            try:
                if hasattr(stock, 'news') and stock.news:
                    for news in stock.news[:limit]:
                        # Format the date
                        timestamp = news.get('providerPublishTime')
                        if timestamp:
                            date_obj = datetime.fromtimestamp(timestamp)
                            news_date = date_obj.strftime("%Y-%m-%d")
                        else:
                            news_date = datetime.now().strftime("%Y-%m-%d")
                            
                        # Skip if outside date range
                        if start_date and news_date < start_date:
                            continue
                        if news_date > end_date:
                            continue
                            
                        news_item = CompanyNews(
                            ticker=ticker,
                            title=news.get('title', ''),
                            date=news_date,
                            source=news.get('publisher', ''),
                            url=news.get('link', ''),
                            summary=news.get('summary', ''),
                            author=news.get('author', 'Yahoo Finance')  # Use 'Yahoo Finance' as default author
                        )
                        news_items.append(news_item)
                        
                    if news_items:
                        # Cache the results as dicts
                        try:
                            _cache.set_company_news(ticker, [n.model_dump() for n in news_items])
                        except Exception as e:
                            print(f"Cache error in Yahoo Finance when storing news: {e}")
                        return news_items
            except Exception as e:
                print(f"Error getting news from stock.news attribute: {e}")
                # Continue to fallback method
                
            # Fallback: Use the ticker's property methods
            try:
                # Try different ways to get news from yfinance as a fallback
                news_data = None
                
                # Try calendar property first
                try:
                    if hasattr(stock, 'calendar') and stock.calendar is not None and isinstance(stock.calendar, pd.DataFrame) and not stock.calendar.empty:
                        news_data = stock.calendar
                except Exception as e:
                    print(f"Error accessing calendar data: {e}")
                
                # Try actions property next
                if news_data is None:
                    try:
                        if hasattr(stock, 'actions') and stock.actions is not None and isinstance(stock.actions, pd.DataFrame) and not stock.actions.empty:
                            news_data = stock.actions
                    except Exception as e:
                        print(f"Error accessing actions data: {e}")
                    
                # Try recommendations property last
                if news_data is None:
                    try:
                        if hasattr(stock, 'recommendations') and stock.recommendations is not None and isinstance(stock.recommendations, pd.DataFrame) and not stock.recommendations.empty:
                            news_data = stock.recommendations
                    except Exception as e:
                        print(f"Error accessing recommendations data: {e}")
                
                # If we got some data, try to parse it
                if news_data is not None and isinstance(news_data, pd.DataFrame) and not news_data.empty:
                    for date, row in news_data.iterrows():
                        news_date = str(date.date()) if hasattr(date, 'date') else str(date)
                        
                        # Skip if outside date range
                        if start_date and news_date < start_date:
                            continue
                        if news_date > end_date:
                            continue
                            
                        # Try to extract a title from columns
                        title = ""
                        for col in row.index:
                            if "title" in col.lower() or "name" in col.lower() or "event" in col.lower():
                                title = str(row[col])
                                break
                        
                        # If no specific title found, use the first column that's not a number
                        if not title:
                            for col in row.index:
                                val = row[col]
                                if not pd.isna(val) and not isinstance(val, (int, float)):
                                    title = f"{col}: {val}"
                                    break
                        
                        # Last resort: use date and ticker as title
                        if not title:
                            title = f"{ticker} update on {news_date}"
                        
                        news_item = CompanyNews(
                            ticker=ticker,
                            title=title[:100],  # Limit title length
                            date=news_date,
                            source="Yahoo Finance",
                            url=f"https://finance.yahoo.com/quote/{ticker}",
                            summary=str(dict(row))[:200],  # Convert row to string and limit summary length
                            author="Yahoo Finance"  # Set default author
                        )
                        news_items.append(news_item)
                        
                        if len(news_items) >= limit:
                            break
                            
                    if news_items:
                        # Cache the results as dicts
                        try:
                            _cache.set_company_news(ticker, [n.model_dump() for n in news_items])
                        except Exception as e:
                            print(f"Cache error in Yahoo Finance when storing news from properties: {e}")
                        return news_items
            except Exception as e:
                print(f"Error in news fallback method: {e}")
                # Continue to final fallback
                
            # Final fallback: Create a single minimal news item with company info
            try:
                # Safely access company information
                company_info = getattr(stock, 'info', {})
                if not company_info or not isinstance(company_info, dict):
                    company_info = {}
                
                company_name = company_info.get('longName', ticker)
                sector = company_info.get('sector', '')
                industry = company_info.get('industry', '')
                
                news_item = CompanyNews(
                    ticker=ticker,
                    title=f"Information for {company_name}",
                    date=end_date,
                    source="Yahoo Finance",
                    url=f"https://finance.yahoo.com/quote/{ticker}",
                    summary=f"Company information for {company_name}: {sector} - {industry}",
                    author="Yahoo Finance"  # Set default author
                )
                
                # Cache the result
                try:
                    _cache.set_company_news(ticker, [news_item.model_dump()])
                except Exception as e:
                    print(f"Cache error in Yahoo Finance when storing minimal news: {e}")
                return [news_item]
            except Exception as e:
                print(f"Error in minimal news fallback: {e}")
                return []
            
        except Exception as e:
            print(f"Yahoo Finance error getting company news for {ticker}: {str(e)}")
            return []

    def search_line_items(
        self, 
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10
    ) -> List[LineItem]:
        """Extract financial line items from Yahoo Finance statements."""
        # Check cache first
        cache_key = f"{ticker}_{','.join(line_items)}_{period}"
        if cached_data := _cache.get_line_items(cache_key):
            # Filter cached data by date and limit
            filtered_data = [LineItem(**item) for item in cached_data if item["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x.report_period, reverse=True)
            if filtered_data:
                return filtered_data[:limit]
                
        try:
            stock = yf.Ticker(ticker)
            
            # Determine appropriate financial statements based on line_items requested
            quarterly_income = stock.quarterly_income_stmt
            quarterly_balance = stock.quarterly_balance_sheet
            quarterly_cash = stock.quarterly_cashflow
            
            annual_income = stock.income_stmt
            annual_balance = stock.balance_sheet
            annual_cash = stock.cashflow
            
            # Initialize results list
            results = []
            
            # Function to standardize column names across different statements
            def standardize_name(name):
                # Convert camelCase or PascalCase to snake_case
                import re
                s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
                # Remove any special characters and replace spaces with underscores
                return re.sub(r'[^a-z0-9_]', '', s2.replace(' ', '_'))
            
            # Create mappings for common financial line items to their possible Yahoo Finance names
            line_item_mapping = {
                # Income Statement Items
                "revenue": ["total revenue", "totalrevenue", "revenue"],
                "gross_profit": ["gross profit", "grossprofit"],
                "operating_income": ["operating income", "operatingincome", "ebit"],
                "net_income": ["net income", "netincome"],
                "eps": ["basic eps", "eps", "earnings per share"],
                
                # Balance Sheet Items
                "total_assets": ["total assets", "totalassets"],
                "total_liabilities": ["total liabilities", "totalliabilities"],
                "total_equity": ["total stockholder equity", "total equity", "stockholderequity"],
                "cash_and_equivalents": ["cash and cash equivalents", "cash and short term investments"],
                "debt": ["total debt", "long term debt"],
                
                # Cash Flow Items
                "operating_cash_flow": ["operating cash flow", "cash flow from operating activities"],
                "capital_expenditures": ["capital expenditures", "property plant and equipment net change"],
                "free_cash_flow": ["free cash flow"]
            }
            
            # Determine which statements to use based on period
            if period.lower() in ["quarterly", "q"]:
                income_stmt = quarterly_income if not quarterly_income.empty else annual_income
                balance_sheet = quarterly_balance if not quarterly_balance.empty else annual_balance
                cash_flow = quarterly_cash if not quarterly_cash.empty else annual_cash
                period_type = "quarterly"
            else:
                income_stmt = annual_income if not annual_income.empty else quarterly_income
                balance_sheet = annual_balance if not annual_balance.empty else quarterly_balance
                cash_flow = annual_cash if not annual_cash.empty else quarterly_cash
                period_type = "annual"
                
            # Combine all statements for easier searching
            all_statements = {
                "income_stmt": income_stmt,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow
            }
            
            # Process dates for each statement
            processed_dates = set()
            
            for stmt_name, stmt in all_statements.items():
                if stmt is None or stmt.empty:
                    continue
                    
                # Process each date/period column in the statement
                for date_col in stmt.columns:
                    # Convert date to string format
                    date_str = date_col.strftime('%Y-%m-%d') if hasattr(date_col, 'strftime') else str(date_col)
                    
                    # Skip if we've already processed this date or if it's after end_date
                    if date_str in processed_dates or date_str > end_date:
                        continue
                        
                    # Create a new LineItem for this reporting period
                    line_item = LineItem(
                        ticker=ticker,
                        report_period=date_str,
                        period=period_type,
                        currency="USD"  # Yahoo Finance typically reports in USD
                    )
                    
                    # Track if we found any of the requested line items
                    found_any = False
                    
                    # Extract each requested line item
                    for requested_item in line_items:
                        # Check if this item has a mapping
                        possible_names = line_item_mapping.get(requested_item.lower(), [requested_item.lower()])
                        
                        # Try each possible name
                        for possible_name in possible_names:
                            # Try to find in each statement
                            for s_name, s in all_statements.items():
                                if s is None or s.empty:
                                    continue
                                    
                                # Look for exact or close matches in the index
                                found = False
                                for idx in s.index:
                                    idx_str = str(idx).lower()
                                    if possible_name.lower() in idx_str or standardize_name(idx_str) == standardize_name(possible_name):
                                        # Found a match, extract the value
                                        try:
                                            value = float(s.loc[idx, date_col])
                                            # Add as attribute to the LineItem
                                            setattr(line_item, standardize_name(requested_item), value)
                                            found = True
                                            found_any = True
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                
                                if found:
                                    break
                    
                    # Only add this period if we found at least one requested line item
                    if found_any:
                        results.append(line_item)
                    
                    # Mark this date as processed
                    processed_dates.add(date_str)
            
            # Sort by report_period (newest first) and limit results
            results.sort(key=lambda x: x.report_period, reverse=True)
            results = results[:limit]
            
            # Cache the results
            if results:
                _cache.set_line_items(cache_key, [r.model_dump() for r in results])
                
            return results
            
        except Exception as e:
            print(f"Yahoo Finance error searching line items for {ticker}: {str(e)}")
            return []

    def get_market_cap(self, ticker: str, end_date: str) -> float:
        """Fetch market cap from Yahoo Finance."""
        # Check cache first
        if cached_data := _cache.get_market_cap(ticker):
            # Filter cached data by date
            if cached_data.get(end_date):
                return cached_data[end_date]
                
        try:
            # Get market cap directly from ticker info instead of financial metrics
            stock = yf.Ticker(ticker)
            info = stock.info
            
            market_cap = info.get("marketCap")
            
            # Cache the result
            if market_cap is not None:
                _cache.set_market_cap(ticker, {end_date: market_cap})
                
            return market_cap
            
        except Exception as e:
            print(f"Yahoo Finance error getting market cap for {ticker}: {str(e)}")
            return None

    def get_balance_sheet(self, ticker: str, end_date: str, quarterly: bool = False) -> dict:
        """Fetch balance sheet data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date for data retrieval
            quarterly: If True, retrieve quarterly data instead of annual
            
        Returns:
            Dictionary containing balance sheet data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get the balance sheet data
            if quarterly:
                balance_sheet = stock.quarterly_balance_sheet
            else:
                balance_sheet = stock.balance_sheet
                
            if balance_sheet is None or balance_sheet.empty:
                print(f"No balance sheet data available for {ticker}")
                return {}
                
            # Convert the balance sheet to a dictionary
            balance_sheet_dict = {}
            for column in balance_sheet.columns:
                date_str = column.strftime('%Y-%m-%d')
                if date_str <= end_date:
                    balance_sheet_dict[date_str] = {}
                    for index, value in balance_sheet[column].items():
                        if pd.notna(value):
                            balance_sheet_dict[date_str][index] = float(value)
            
            return balance_sheet_dict
            
        except Exception as e:
            print(f"Yahoo Finance error getting balance sheet for {ticker}: {str(e)}")
            return {}
    
    def get_income_statement(self, ticker: str, end_date: str, quarterly: bool = False) -> dict:
        """Fetch income statement data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date for data retrieval
            quarterly: If True, retrieve quarterly data instead of annual
            
        Returns:
            Dictionary containing income statement data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get the income statement data
            if quarterly:
                income_stmt = stock.quarterly_financials
            else:
                income_stmt = stock.financials
                
            if income_stmt is None or income_stmt.empty:
                print(f"No income statement data available for {ticker}")
                return {}
                
            # Convert the income statement to a dictionary
            income_stmt_dict = {}
            for column in income_stmt.columns:
                date_str = column.strftime('%Y-%m-%d')
                if date_str <= end_date:
                    income_stmt_dict[date_str] = {}
                    for index, value in income_stmt[column].items():
                        if pd.notna(value):
                            income_stmt_dict[date_str][index] = float(value)
            
            return income_stmt_dict
            
        except Exception as e:
            print(f"Yahoo Finance error getting income statement for {ticker}: {str(e)}")
            return {}
    
    def get_cash_flow(self, ticker: str, end_date: str, quarterly: bool = False) -> dict:
        """Fetch cash flow data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date for data retrieval
            quarterly: If True, retrieve quarterly data instead of annual
            
        Returns:
            Dictionary containing cash flow data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get the cash flow data
            if quarterly:
                cash_flow = stock.quarterly_cashflow
            else:
                cash_flow = stock.cashflow
                
            if cash_flow is None or cash_flow.empty:
                print(f"No cash flow data available for {ticker}")
                return {}
                
            # Convert the cash flow to a dictionary
            cash_flow_dict = {}
            for column in cash_flow.columns:
                date_str = column.strftime('%Y-%m-%d')
                if date_str <= end_date:
                    cash_flow_dict[date_str] = {}
                    for index, value in cash_flow[column].items():
                        if pd.notna(value):
                            cash_flow_dict[date_str][index] = float(value)
            
            return cash_flow_dict
            
        except Exception as e:
            print(f"Yahoo Finance error getting cash flow for {ticker}: {str(e)}")
            return {}
            
    def get_asset_profile(self, ticker: str) -> dict:
        """Fetch asset profile data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing asset profile data
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Select relevant profile information
            profile_fields = [
                'sector', 'industry', 'fullTimeEmployees', 'longBusinessSummary',
                'country', 'state', 'city', 'address1', 'phone', 'website',
                'exchange', 'exchangeTimezoneName', 'marketCap', 'sharesOutstanding'
            ]
            
            profile = {field: info.get(field) for field in profile_fields if field in info}
            
            # Add additional company officers if available
            if 'companyOfficers' in info:
                profile['companyOfficers'] = info['companyOfficers']
                
            return profile
            
        except Exception as e:
            print(f"Yahoo Finance error getting asset profile for {ticker}: {str(e)}")
            return {}
            
    def get_historical_data(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Fetch historical price data with more options from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval (1d, 1wk, 1mo, etc.)
            
        Returns:
            DataFrame containing historical price data
        """
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                print(f"No historical data available for {ticker}")
                return pd.DataFrame()
                
            # Format the DataFrame
            df.index = df.index.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            print(f"Yahoo Finance error getting historical data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_balance_sheet_items(self, ticker: str, end_date: str, quarterly: bool = False) -> List[BalanceSheetItem]:
        """Fetch balance sheet data from Yahoo Finance and return as model objects.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date for data retrieval
            quarterly: If True, retrieve quarterly data instead of annual
            
        Returns:
            List of BalanceSheetItem objects
        """
        try:
            # Get raw balance sheet data
            balance_sheet_dict = self.get_balance_sheet(ticker, end_date, quarterly)
            
            # Convert to model objects
            items = []
            report_period = "quarterly" if quarterly else "annual"
            
            for date, values in balance_sheet_dict.items():
                for item_name, value in values.items():
                    item = BalanceSheetItem(
                        ticker=ticker,
                        report_date=date,
                        item_name=item_name,
                        value=value,
                        report_period=report_period,
                        currency="USD"  # Default currency
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            print(f"Error converting balance sheet data to models for {ticker}: {str(e)}")
            return []
    
    def get_income_statement_items(self, ticker: str, end_date: str, quarterly: bool = False) -> List[IncomeStatementItem]:
        """Fetch income statement data from Yahoo Finance and return as model objects.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date for data retrieval
            quarterly: If True, retrieve quarterly data instead of annual
            
        Returns:
            List of IncomeStatementItem objects
        """
        try:
            # Get raw income statement data
            income_stmt_dict = self.get_income_statement(ticker, end_date, quarterly)
            
            # Convert to model objects
            items = []
            report_period = "quarterly" if quarterly else "annual"
            
            for date, values in income_stmt_dict.items():
                for item_name, value in values.items():
                    item = IncomeStatementItem(
                        ticker=ticker,
                        report_date=date,
                        item_name=item_name,
                        value=value,
                        report_period=report_period,
                        currency="USD"  # Default currency
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            print(f"Error converting income statement data to models for {ticker}: {str(e)}")
            return []
    
    def get_cash_flow_items(self, ticker: str, end_date: str, quarterly: bool = False) -> List[CashFlowItem]:
        """Fetch cash flow data from Yahoo Finance and return as model objects.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date for data retrieval
            quarterly: If True, retrieve quarterly data instead of annual
            
        Returns:
            List of CashFlowItem objects
        """
        try:
            # Get raw cash flow data
            cash_flow_dict = self.get_cash_flow(ticker, end_date, quarterly)
            
            # Convert to model objects
            items = []
            report_period = "quarterly" if quarterly else "annual"
            
            for date, values in cash_flow_dict.items():
                for item_name, value in values.items():
                    item = CashFlowItem(
                        ticker=ticker,
                        report_date=date,
                        item_name=item_name,
                        value=value,
                        report_period=report_period,
                        currency="USD"  # Default currency
                    )
                    items.append(item)
            
            return items
            
        except Exception as e:
            print(f"Error converting cash flow data to models for {ticker}: {str(e)}")
            return []
    
    def get_company_profile(self, ticker: str) -> AssetProfile:
        """Fetch company profile data from Yahoo Finance and return as a model object.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            AssetProfile object
        """
        try:
            # Get raw profile data
            profile_dict = self.get_asset_profile(ticker)
            
            # Convert to model object
            profile = AssetProfile(
                ticker=ticker,
                sector=profile_dict.get('sector'),
                industry=profile_dict.get('industry'),
                employees=profile_dict.get('fullTimeEmployees'),
                description=profile_dict.get('longBusinessSummary'),
                country=profile_dict.get('country'),
                state=profile_dict.get('state'),
                city=profile_dict.get('city'),
                address=profile_dict.get('address1'),
                phone=profile_dict.get('phone'),
                website=profile_dict.get('website'),
                exchange=profile_dict.get('exchange'),
                timezone=profile_dict.get('exchangeTimezoneName'),
                market_cap=profile_dict.get('marketCap'),
                shares_outstanding=profile_dict.get('sharesOutstanding'),
                officers=profile_dict.get('companyOfficers')
            )
            
            return profile
            
        except Exception as e:
            print(f"Error converting profile data to model for {ticker}: {str(e)}")
            # Return an empty profile with just the ticker
            return AssetProfile(ticker=ticker)


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