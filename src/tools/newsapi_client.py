"""NewsAPI client for fetching company news."""
import os
from typing import List, Optional
from datetime import datetime, timedelta
from newsapi import NewsApiClient

from data.models import CompanyNews
from data.cache import get_cache

# Initialize cache
_cache = get_cache()

class NewsAPIClient:
    """Client for interacting with NewsAPI."""
    
    def __init__(self):
        """Initialize the NewsAPI client."""
        self.api_key = os.environ.get("NEWSAPI_KEY", "")
        self.client = None
        if self.api_key:
            self.client = NewsApiClient(api_key=self.api_key)
    
    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[CompanyNews]:
        """Fetch company news from NewsAPI.
        
        Args:
            ticker: Stock ticker symbol
            end_date: End date in YYYY-MM-DD format
            start_date: Start date in YYYY-MM-DD format (optional)
            limit: Maximum number of news articles to return
            
        Returns:
            List of CompanyNews objects
        """
        if not self.client:
            print("NewsAPI key not found. Set NEWSAPI_KEY environment variable.")
            return []
            
        # Check cache first
        try:
            cached_data = _cache.get_company_news(ticker)
            if cached_data:
                # Filter by date if we have start_date and end_date
                if start_date and end_date:
                    filtered_data = [
                        news for news in cached_data 
                        if start_date <= news.get('date', '').split('T')[0] <= end_date
                    ]
                    if filtered_data:
                        # Convert cached data to CompanyNews objects
                        return [CompanyNews(**news) for news in filtered_data]
        except Exception as e:
            print(f"Cache error in get_company_news: {e}")
            # Continue to fetch new data if cache fails
            
        try:
            # Get company name for better search results
            company_name = self._get_company_name(ticker)
            query = company_name or ticker
            
            # If no start date provided, default to 30 days prior
            if not start_date:
                start_date = (datetime.fromisoformat(end_date) - timedelta(days=30)).strftime("%Y-%m-%d")
                
            # NewsAPI free plan has date restrictions - ensure we're not requesting too far back
            now = datetime.now()
            # Most NewsAPI free plans only allow going back 1 month from current date
            max_days_ago = 30
            min_allowed_date = (now - timedelta(days=max_days_ago)).strftime("%Y-%m-%d")
            
            # Adjust start_date if it's too far back for the plan limits
            if start_date < min_allowed_date:
                print(f"Adjusting start_date from {start_date} to {min_allowed_date} due to NewsAPI plan limitations")
                start_date = min_allowed_date
                
            # Ensure end_date isn't in the future
            if datetime.fromisoformat(end_date) > now:
                end_date = now.strftime("%Y-%m-%d")
            
            # Fetch news from NewsAPI
            try:
                response = self.client.get_everything(
                    q=query,
                    from_param=start_date,
                    to=end_date,
                    language='en',
                    sort_by='publishedAt',
                    page_size=min(100, limit)  # API max is 100 per page
                )
                
                articles = response.get("articles", [])
                news_list = []
                
                for article in articles[:limit]:
                    # Fix validation errors by ensuring all required fields are present and correct types
                    news_item = CompanyNews(
                        ticker=ticker,
                        date=article.get("publishedAt", ""),
                        title=article.get("title", "No Title"),
                        author=article.get("author", "Unknown Author"),  # Use default if missing
                        source=article.get("source", {}).get("name", "NewsAPI"),
                        url=article.get("url", ""),
                        sentiment="neutral"  # Use string value instead of float
                    )
                    news_list.append(news_item)
                
                # Cache the results
                try:
                    _cache.set_company_news(ticker, [news.model_dump() for news in news_list])
                except Exception as e:
                    print(f"Cache error when storing company news: {e}")
                
                return news_list
            except Exception as e:
                print(f"NewsAPI API error: {e}")
                # If we get here, try a simpler query with fewer parameters
                try:
                    # Simplified request with fewer parameters
                    response = self.client.get_everything(
                        q=query,
                        language='en',
                        page_size=min(100, limit)
                    )
                    
                    articles = response.get("articles", [])
                    news_list = []
                    
                    for article in articles[:limit]:
                        # Fix validation errors by ensuring all required fields are present and correct types
                        news_item = CompanyNews(
                            ticker=ticker,
                            date=article.get("publishedAt", ""),
                            title=article.get("title", "No Title"),
                            author=article.get("author", "Unknown Author"),  # Use default if missing
                            source=article.get("source", {}).get("name", "NewsAPI"),
                            url=article.get("url", ""),
                            sentiment="neutral"  # Use string value instead of float
                        )
                        news_list.append(news_item)
                    
                    # Cache the results
                    try:
                        _cache.set_company_news(ticker, [news.model_dump() for news in news_list])
                    except Exception as e:
                        print(f"Cache error when storing company news: {e}")
                    
                    return news_list
                except Exception as nested_e:
                    print(f"Secondary NewsAPI error: {nested_e}")
                    # Fall through to return empty list
                
        except Exception as e:
            print(f"NewsAPI error getting news for {ticker}: {str(e)}")
        
        # If we get here, all attempts failed
        return []
            
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker.
        
        This is a simple mapping for common tickers. In a real system,
        you would use a more comprehensive database or API.
        """
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan',
            'BAC': 'Bank of America',
            'WMT': 'Walmart'
        }
        
        return company_names.get(ticker, ticker)

# Singleton instance
_newsapi_client = NewsAPIClient()

def get_newsapi_client() -> NewsAPIClient:
    """Get the NewsAPI client instance."""
    return _newsapi_client 