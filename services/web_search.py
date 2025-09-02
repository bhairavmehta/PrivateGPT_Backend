import asyncio
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import random
import httpx
import re
import time
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse, quote_plus, urljoin
from duckduckgo_search import DDGS
import json
from datetime import datetime, timedelta
from config import Settings
import undetected_chromedriver as uc

logger = logging.getLogger(__name__)

class RobustWebSearchService:
    """Base class for robust web search services with scraping capabilities."""
    
    _driver = None
    _driver_creation_lock = asyncio.Lock()
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.timeout = self.settings.search_timeout
        self.user_agent = self.settings.user_agent

    @classmethod
    async def get_driver(cls):
        """Get the Selenium WebDriver, creating it if it doesn't exist."""
        async with cls._driver_creation_lock:
            if cls._driver is None:
                try:
                    logger.info("Initializing undetected-chromedriver...")
                    options = uc.ChromeOptions()
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    options.add_argument('--headless')
                    options.add_argument('--disable-gpu')
                    options.add_argument('--disable-blink-features=AutomationControlled')
                    options.add_argument('--disable-extensions')
                    options.add_argument('--disable-plugins-discovery')
                    cls._driver = uc.Chrome(options=options)  # Auto-detect version
                    logger.info("✅ undetected-chromedriver initialized successfully.")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize undetected-chromedriver: {e}")
                    cls._driver = None
            return cls._driver

    async def search_and_scrape(self, query: str, max_results: int = 5, retries: int = 2) -> List[Dict[str, Any]]:
        """Search and scrape with multiple query variants for better success rate."""
        search_variants = self._generate_search_variants(query)
        
        for i, variant_query in enumerate(search_variants):
            logger.info(f"Trying search variant {i+1}/{len(search_variants)}: '{variant_query}'")
            
            try:
                # Perform web search
                search_results = await self.search(variant_query, max_results=max_results)

                if search_results:
                    logger.info(f"✅ Search successful with variant {i+1}: '{variant_query}' - found {len(search_results)} results")

                    # Scrape content for each result concurrently
                    scraping_tasks = [
                        self.scrape(result["url"], retries=retries) for result in search_results
                    ]
                    scraped_contents = await asyncio.gather(*scraping_tasks, return_exceptions=True)

                    # Combine search results with scraped content
                    final_results = []
                    for j, result in enumerate(search_results):
                        content = scraped_contents[j]
                        if isinstance(content, Exception):
                            logger.warning(f"Scraping failed for {result['url']}: {content}")
                            result["scraped_content"] = None
                        else:
                            result["scraped_content"] = content
                        final_results.append(result)
                    
                    logger.info(f"Completed scraping for {len(search_results)} results")
                    return final_results
                else:
                    logger.warning(f"❌ No results found for variant {i+1}: '{variant_query}'")
            
            except Exception as e:
                logger.error(f"❌ Search failed for variant {i+1}: '{variant_query}' - {e}")
                continue
        
        # If all variants fail, log and return empty
        logger.warning(f"❌ All search variants failed for query: '{query}'")
        return []

    def _generate_search_variants(self, query: str) -> List[str]:
        """Generate multiple search query variants for better success rate."""
        variants = []
        
        # Clean the original query
        cleaned_query = query.strip().strip('"').strip("'").strip()
        
        # Variant 1: Original cleaned query (without quotes)
        if cleaned_query:
            variants.append(cleaned_query)
        
        # Variant 2: Remove common phrases and make it broader
        broad_query = cleaned_query.lower()
        # Remove specific view/perspective words
        broad_query = re.sub(r'\b(top view|aerial view|from above|bird.?s eye view|overhead view)\b', '', broad_query, flags=re.IGNORECASE)
        # Remove "of" and connecting words
        broad_query = re.sub(r'\b(of|with|and|in|at|on)\b', ' ', broad_query, flags=re.IGNORECASE)
        # Clean up extra spaces
        broad_query = re.sub(r'\s+', ' ', broad_query).strip()
        if broad_query and broad_query != cleaned_query.lower():
            variants.append(broad_query)
        
        # Variant 3: Extract key nouns only
        key_words = []
        words = cleaned_query.split()
        for word in words:
            word = word.lower().strip('.,!?;:')
            # Keep important nouns and adjectives, skip common words
            if (word not in ['the', 'a', 'an', 'and', 'or', 'but', 'with', 'from', 'to', 'of', 'in', 'on', 'at', 'by', 'for'] 
                and len(word) > 2):
                key_words.append(word)
        
        if len(key_words) > 1:
            key_query = ' '.join(key_words[:4])  # Use first 4 key words
            if key_query not in [v.lower() for v in variants]:
                variants.append(key_query)
        
        # Variant 4: Most important single keywords if we have them
        important_keywords = []
        for word in words:
            word = word.lower().strip('.,!?;:')
            if word in ['industrial', 'factory', 'building', 'warehouse', 'complex', 'facility', 'plant', 'site', 'area', 'zone', 'park']:
                important_keywords.append(word)
        
        if important_keywords:
            # Try the most specific keyword first
            for keyword in important_keywords[:2]:  # Try top 2 keywords
                if keyword not in [v.lower() for v in variants]:
                    variants.append(keyword)
        
        # Variant 5: Generic fallback if all else fails
        if len(variants) < 3:
            if 'industrial' in cleaned_query.lower():
                variants.append('industrial facility')
            elif 'factory' in cleaned_query.lower():
                variants.append('factory building')
            elif 'building' in cleaned_query.lower():
                variants.append('commercial building')
            else:
                variants.append('aerial photography')
        
        # Remove duplicates while preserving order
        unique_variants = []
        seen = set()
        for variant in variants:
            variant_lower = variant.lower()
            if variant_lower not in seen and len(variant.strip()) > 0:
                unique_variants.append(variant)
                seen.add(variant_lower)
        
        logger.info(f"Generated {len(unique_variants)} search variants: {unique_variants}")
        return unique_variants[:5]  # Limit to 5 variants to avoid too many API calls

    async def scrape(self, url: str, retries: int = 2, delay: int = 1) -> Optional[str]:
        """Scrape content from a given URL with retry logic."""
        logger.info(f"Scraping result: {url}")
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    headers = {
                        "User-Agent": random.choice(self.settings.user_agents),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Referer": "https://www.google.com/"
                    }
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    
                    # Use BeautifulSoup to parse and extract clean text
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    return soup.get_text(separator="\n", strip=True)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} with httpx failed for {url}: {e}")
                if attempt < retries:
                    # Fallback to Selenium on the last retry attempt
                    if attempt == retries - 1:
                        logger.info(f"Falling back to Selenium for {url}")
                        try:
                            driver = await self.get_driver()
                            if driver:
                                driver.get(url)
                                await asyncio.sleep(3)  # Wait for page to load
                                page_source = driver.page_source
                                soup = BeautifulSoup(page_source, "html.parser")
                                for script in soup(["script", "style"]):
                                    script.extract()
                                return soup.get_text(separator="\n", strip=True)
                        except Exception as selenium_e:
                            logger.error(f"Selenium scraping also failed for {url}: {selenium_e}")
                            return None
                    
                    await asyncio.sleep(delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All {retries + 1} scraping attempts failed for {url}.")
                    return None
        return None

    def _get_rate_limit_sleep(self) -> float:
        """Calculate sleep time to respect rate limiting."""
        current_time = datetime.now()
        if BraveWebSearchService._brave_last_request_time:
            time_since_last = (current_time - BraveWebSearchService._brave_last_request_time).total_seconds()
            if time_since_last < 1.0:
                sleep_time = 1.0 - time_since_last
                logger.info(f"⏱️  Rate limiting: sleeping {sleep_time:.2f}s for Brave Search")
                return sleep_time
        return 0.0

    def format_search_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into a comprehensive context string for LLM analysis."""
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            snippet = result.get('snippet', '')
            scraped_content = result.get('scraped_content', '')
            
            # Use scraped content if available and meaningful, otherwise use snippet
            content = scraped_content if scraped_content and len(scraped_content) > 50 else snippet
            
            # Format the source entry with more comprehensive content (increased from 1000 to 4000)
            source_entry = f"Source {i}: {title}\n"
            source_entry += f"URL: {url}\n"
            source_entry += f"Content: {content[:4000]}...\n"  # Increased from 1000 to 4000 for much more context
            source_entry += "-" * 40 + "\n"
            
            context_parts.append(source_entry)
        
        return "\n".join(context_parts)

    def close(self):
        """Clean up resources (no resources to clean for this simple implementation)."""
        pass

class BraveWebSearchService(RobustWebSearchService):
    """Brave Search implementation of the web search service."""
    _brave_last_request_time: Optional[datetime] = None

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not self.settings.brave_search_api_key:
            raise ValueError("Brave Search API key is not set in settings")

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search using Brave Search API."""
        logger.info(f"Starting Brave search for: '{query}'")
        
        try:
            results = await self._search_brave(query, max_results)
            if results:
                logger.info(f"✅ Brave search success: {len(results)} results")
                return results
            else:
                logger.warning("❌ Brave search returned no results")
                return []
        except Exception as e:
            logger.error(f"❌ Brave search failed: {e}")
            return []
    
    async def _search_brave(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Enhanced Brave Search API integration with rate limiting."""
        try:
            # Implement rate limiting (1 request per second)
            current_time = datetime.now()
            if BraveWebSearchService._brave_last_request_time:
                time_since_last = (current_time - BraveWebSearchService._brave_last_request_time).total_seconds()
                if time_since_last < 1.0:
                    sleep_time = 1.0 - time_since_last
                    logger.info(f"⏱️  Rate limiting: sleeping {sleep_time:.2f}s for Brave Search")
                    await asyncio.sleep(sleep_time)
            
            # Update last request time
            BraveWebSearchService._brave_last_request_time = datetime.now()
            
            # Prepare API request
            search_url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.settings.brave_search_api_key,
                "User-Agent": getattr(self.settings, 'user_agent', 'LocalLLM-Backend/1.0')
            }
            
            params = {
                "q": query,
                "count": min(max_results, 20),  # Brave API allows max 20 results
                "search_lang": "en",
                "country": "US",
                "safesearch": "moderate",
                "freshness": "all",
                "text_decorations": False,
                "spellcheck": True
            }
            
            logger.info(f"🔍 Brave Search API request: {query} (max_results: {max_results})")
            
            # Make API request with timeout
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(search_url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"✅ Brave Search API response received")
                
                # Parse results
                results = []
                web_results = data.get("web", {}).get("results", [])
                
                if not web_results:
                    logger.warning("No web results found in Brave Search response")
                    return []
                
                logger.info(f"📊 Processing {len(web_results)} Brave Search results")
                
                for i, result in enumerate(web_results[:max_results]):
                    try:
                        title = result.get("title", "").strip()
                        url = result.get("url", "").strip()
                        description = result.get("description", "").strip()
                        
                        # Validate result quality
                        if (url and title and len(title) > 5 and 
                            url.startswith(('http://', 'https://')) and
                            self._is_valid_search_result(url, title, query)):
                            
                            results.append({
                                "title": title[:200],
                                "url": url,
                                "snippet": description[:500] if description else "",
                                "source": "Brave Search"
                            })
                            
                            logger.info(f"🎯 Brave result {len(results)}: {title[:50]}...")
                        else:
                            logger.debug(f"Skipping low-quality Brave result: {title[:50]}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing Brave result {i}: {e}")
                        continue
                
                if results:
                    logger.info(f"✅ Brave Search successful: {len(results)} quality results")
                else:
                    logger.warning("❌ No valid results after filtering Brave Search response")
                
                return results
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error("🚫 Brave Search rate limit exceeded")
            elif e.response.status_code == 401:
                logger.error("🔑 Brave Search API authentication failed - check API key")
            elif e.response.status_code == 403:
                logger.error("🚫 Brave Search API access forbidden - check subscription")
            else:
                logger.error(f"🚫 Brave Search HTTP error {e.response.status_code}: {e}")
            return []
        except httpx.TimeoutException:
            logger.error("⏰ Brave Search API timeout")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"📄 Failed to parse Brave Search API response: {e}")
            return []
        except Exception as e:
            logger.error(f"💥 Brave Search API error: {e}")
            return []
    
    def _is_valid_search_result(self, url: str, title: str, query: str) -> bool:
        """Validate if a search result is relevant and not spam."""
        # Filter out clearly irrelevant results
        spam_indicators = [
            'bing.com/search',
            'google.com/search', 
            'javascript:',
            'mailto:',
            'about:blank'
        ]
        
        # Filter out stock photo and commercial image sites
        stock_photo_sites = [
            'shutterstock.com',
            'gettyimages.com',
            'istockphoto.com',
            'stock.adobe.com',
            'alamy.com',
            'dreamstime.com',
            'depositphotos.com',
            'stockvault.net',
            'pexels.com',
            'unsplash.com',
            'pixabay.com',
            'freepik.com',
            'vecteezy.com',
            '123rf.com',
            'fotolia.com',
            'canstockphoto.com'
        ]
        
        # Filter out Pinterest and other image aggregators
        image_aggregators = [
            'pinterest.com',
            'pinterest.co.uk',
            'pinterest.ca',
            'pinterest.de',
            'pinterest.fr',
            'pinterest.it',
            'pinterest.es',
            'pinterest.com.au',
            'pinterest.jp',
            'pinterest.kr',
            'in.pinterest.com',
            'br.pinterest.com',
            'tumblr.com',
            'flickr.com',
            'photobucket.com',
            'imgur.com'
        ]
        
        # Combine all filtered sites
        filtered_sites = spam_indicators + stock_photo_sites + image_aggregators
        
        # Check for filtered URLs
        url_lower = url.lower()
        for indicator in filtered_sites:
            if indicator in url_lower:
                return False
        
        # Filter out search result URLs that contain stock photo keywords
        stock_keywords_in_url = [
            '/stock-photo',
            '/stock-image',
            '/stock-vector',
            '/royalty-free',
            '/images/search',
            '/photos/search',
            '/search?k=',
            '/search?q=',
            '/search/',
            '/vectors/',
            '/illustrations/'
        ]
        
        for keyword in stock_keywords_in_url:
            if keyword in url_lower:
                return False
        
        # Check for overly generic titles that don't match query
        generic_titles = [
            'home',
            'index',
            'default',
            'untitled',
            'page not found',
            '404',
            'error'
        ]
        
        title_lower = title.lower()
        for generic in generic_titles:
            if title_lower == generic:
                return False
        
        # Filter out titles that are clearly stock photo listings
        stock_title_indicators = [
            'stock photos',
            'stock images',
            'royalty-free',
            'shutterstock',
            'getty images',
            'adobe stock',
            'istock',
            'browse',
            'download',
            'buy photos',
            'purchase',
            'license',
            'vectors',
            'illustrations',
            'clipart'
        ]
        
        for indicator in stock_title_indicators:
            if indicator in title_lower:
                return False
        
        # Prefer informational and educational content
        preferred_domains = [
            'wikipedia.org',
            'britannica.com',
            'edu',
            'gov',
            'mil',
            'org',
            'reuters.com',
            'bbc.com',
            'cnn.com',
            'npr.org',
            'nationalgeographic.com',
            'smithsonianmag.com',
            'history.com',
            'sciencedirect.com',
            'jstor.org',
            'arxiv.org',
            'researchgate.net'
        ]
        
        # Boost score for preferred domains (this doesn't affect filtering but could be used for ranking)
        for domain in preferred_domains:
            if domain in url_lower:
                return True
        
        # If we get here, it's probably a valid non-stock result
        return True

# Maintain backwards compatibility
class RobustWebSearchService(BraveWebSearchService):
    """Backwards compatible wrapper."""
    pass

class WebSearchService(BraveWebSearchService):
    """Backwards compatible wrapper."""
    pass

class PrivacyWebSearchService(BraveWebSearchService):
    """Privacy-focused version using the same Brave methods."""
    pass

# Factory function to get the appropriate search service
def get_search_service(settings: Any) -> BraveWebSearchService:
    """Factory function to get the Brave search service."""
    return BraveWebSearchService(settings)