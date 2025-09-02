import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse
from collections import defaultdict

from services.web_search import RobustWebSearchService

logger = logging.getLogger(__name__)

class DeepResearchService:
    """Deep research service for comprehensive topic analysis"""
    
    def __init__(self, settings, llm_manager):
        self.settings = settings
        self.llm_manager = llm_manager
        self.web_search = RobustWebSearchService(settings)
    
    async def conduct_research(self, query: str, max_sources: int = 10, depth: int = 2, model_type: str = "llama3") -> Dict[str, Any]:
        """Conduct comprehensive research on a given query with fallback strategies."""
        logger.info(f"Starting deep research on: {query}")
        
        # Generate alternative search queries
        search_variants = self._generate_search_variants(query)
        
        research_results = {
            "query": query,
            "sources": [],
            "summary": "",
            "findings": [],
            "related_topics": [],
            "analysis": "",
            "confidence_score": 0.0,
            "key_points": []
        }
        
        # Try each search variant until we get results
        for i, variant_query in enumerate(search_variants):
            logger.info(f"Deep research trying variant {i+1}/{len(search_variants)}: '{variant_query}'")
        
            try:
                # Phase 1: Gather initial sources with variant query
                initial_sources = await self._gather_initial_sources(variant_query, max_sources)
                
                if initial_sources:
                    logger.info(f"✅ Deep research successful with variant {i+1}: '{variant_query}' - found {len(initial_sources)} sources")
                    research_results["sources"] = initial_sources
                    research_results["query"] = variant_query  # Update with successful query
                    break
                else:
                    logger.warning(f"❌ No sources found for deep research variant {i+1}: '{variant_query}'")
                    
            except Exception as e:
                logger.error(f"❌ Deep research failed for variant {i+1}: '{variant_query}' - {e}")
                continue
        
        # If we found sources, process them further
        if research_results["sources"]:
            # Phase 2: Validate and filter sources
            validated_sources = await self._validate_sources(research_results["sources"])
            research_results["sources"] = validated_sources
            
            # Phase 3: Extract key information and generate analysis
            if validated_sources:
                key_info = await self._extract_key_information(validated_sources)
                research_results.update(key_info)
                
                # Generate comprehensive analysis
                analysis = await self._generate_analysis(query, research_results, model_type)
                research_results["analysis"] = analysis
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(validated_sources)
                research_results["confidence_score"] = confidence_score
                
                logger.info(f"Deep research completed. Found {len(validated_sources)} validated sources with confidence {confidence_score}")
            else:
                logger.warning("No sources passed validation in deep research")
        else:
            logger.warning(f"❌ All deep research variants failed for query: '{query}'")
        
        return research_results
    
    async def _gather_initial_sources(self, query: str, max_sources: int) -> List[Dict[str, Any]]:
        """Gather initial sources through web search and scraping"""
        # Perform comprehensive search and scraping
        sources = await self.web_search.search_and_scrape(query, max_sources)
        
        # Enhance sources with metadata
        enhanced_sources = []
        for source in sources:
            enhanced_source = {
                **source,
                "research_phase": "initial",
                "relevance_score": self._calculate_relevance(source, query),
                "credibility_score": self._assess_credibility(source),
                "content_quality": self._assess_content_quality(source)
            }
            enhanced_sources.append(enhanced_source)
        
        # Sort by relevance and quality
        enhanced_sources.sort(
            key=lambda x: x["relevance_score"] * x["credibility_score"] * x["content_quality"],
            reverse=True
        )
        
        return enhanced_sources[:max_sources]
    
    async def _generate_follow_up_queries(
        self, 
        original_query: str, 
        sources: List[Dict[str, Any]],
        model_type: str = "default"
    ) -> List[str]:
        """Generate follow-up search queries based on initial findings using an LLM."""
        
        source_snippets = "\n".join([f"- {s.get('snippet', '')}" for s in sources[:5]])
        
        prompt = f"Based on the initial search results for '{original_query}', generate 2-3 new, more specific search queries to find deeper information. Focus on key entities, technical terms, or unanswered questions. Initial results:\n{source_snippets}"
        
        try:
            response = await self.llm_manager.generate_response(
                message=prompt,
                model_type=model_type, # Use the passed model type
                max_tokens=100,
                temperature=0.4
            )
            
            # The response will likely be a numbered or bulleted list.
            # We need to parse it into a list of strings.
            text = response.get('text', '')
            queries = [q.strip() for q in text.split('\n') if q.strip()]
        
            # Clean up the queries
            cleaned_queries = []
            for q in queries:
                # Remove any surrounding quotes and formatting like **Query:**
                cleaned_q = re.sub(r'^\s*\*\*Query:\*\*\s*', '', q, flags=re.IGNORECASE)
                cleaned_q = cleaned_q.strip().strip('"').strip("'")
                # Remove numbering like "1. " or "- "
                cleaned_q = re.sub(r'^\d+\.\s*', '', cleaned_q)
                cleaned_q = re.sub(r'^-+\s*', '', cleaned_q)
                if cleaned_q:
                    cleaned_queries.append(cleaned_q)
                
            return cleaned_queries[:3]  # Limit to 3 queries
        except Exception as e:
            logger.error(f"Failed to generate follow-up queries: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text (simplified implementation)"""
        # Remove common words and extract meaningful terms
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", 
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "this", "that", "these", "those", "what", "when", "where", "why", "how"
        }
        
        # Extract words that might be important
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [word for word in words if word not in common_words]
        
        # Count frequency and return most common
        word_counts = defaultdict(int)
        for word in keywords:
            word_counts[word] += 1
        
        # Return top keywords
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:10] if count > 1]
    
    def _is_valid_research_source(self, url: str, title: str, content: str = "") -> bool:
        """Validate if a research source is relevant and not commercial/spam."""
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
                logger.debug(f"Filtering out source (blocked domain): {url}")
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
                logger.debug(f"Filtering out source (stock photo URL): {url}")
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
                logger.debug(f"Filtering out source (generic title): {title}")
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
                logger.debug(f"Filtering out source (stock photo title): {title}")
                return False
        
        # Prefer informational and educational content
        preferred_domains = [
            'wikipedia.org',
            'britannica.com',
            '.edu',
            '.gov',
            '.mil',
            '.org',
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
            'researchgate.net',
            'nature.com',
            'science.org',
            'ieee.org',
            'acm.org'
        ]
        
        # Boost score for preferred domains
        for domain in preferred_domains:
            if domain in url_lower:
                logger.debug(f"Prioritizing high-quality source: {url}")
                return True
        
        # Additional content-based filtering for deep research
        if content:
            content_lower = content.lower()
            # Filter out content that's clearly commercial
            commercial_indicators = [
                'buy now',
                'purchase',
                'download',
                'royalty-free',
                'license',
                'subscription',
                'premium',
                'watermark'
            ]
            
            commercial_count = sum(1 for indicator in commercial_indicators if indicator in content_lower)
            if commercial_count > 2:  # If too many commercial indicators
                logger.debug(f"Filtering out source (commercial content): {url}")
                return False
        
        # If we get here, it's probably a valid non-stock research source
        return True
    
    async def _validate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter sources to ensure quality for research."""
        if not sources:
            return []
        
        validated_sources = []
        for source in sources:
            url = source.get('url', '')
            title = source.get('title', '')
            content = source.get('scraped_content', '') or source.get('snippet', '')
            
            # Apply enhanced filtering
            if self._is_valid_research_source(url, title, content):
                # Additional quality checks
                if len(title) > 5 and len(content) > 50:  # Ensure substantial content
                    validated_sources.append(source)
                    logger.debug(f"✅ Validated research source: {title[:50]}...")
                else:
                    logger.debug(f"❌ Source too short: {title[:30]}...")
            else:
                logger.debug(f"❌ Filtered out source: {title[:50]}...")
        
        logger.info(f"Research source validation: {len(validated_sources)}/{len(sources)} sources passed filters")
        return validated_sources
    
    def _are_sources_similar(self, source1: Dict[str, Any], source2: Dict[str, Any]) -> bool:
        """Check if two sources are similar/duplicates"""
        # Check URL similarity
        url1_domain = urlparse(source1.get("url", "")).netloc
        url2_domain = urlparse(source2.get("url", "")).netloc
        
        if url1_domain == url2_domain:
            # Same domain, check title similarity
            title1 = source1.get("title", "").lower()
            title2 = source2.get("title", "").lower()
            
            # Simple similarity check
            if title1 in title2 or title2 in title1:
                return True
        
        return False
    
    async def _extract_key_information(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key information and patterns from sources"""
        all_content = []
        for source in sources:
            content = source.get("scraped_content", "") or source.get("snippet", "")
            if content:
                all_content.append({
                    "content": content,
                    "source": source.get("url", ""),
                    "title": source.get("title", "")
                })
        
        if not all_content:
            return {"key_points": [], "summary": "", "contradictions": []}
        
        # Extract key points (simplified - in practice, use NLP)
        key_points = self._extract_key_points(all_content)
        
        # Generate summary
        summary = self._generate_summary(all_content)
        
        # Identify potential contradictions
        contradictions = self._identify_contradictions(all_content)
        
        return {
            "key_points": key_points,
            "summary": summary,
            "contradictions": contradictions
        }
    
    def _extract_key_points(self, content_list: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from content (simplified implementation)"""
        key_points = []
        
        for item in content_list[:5]:  # Limit to top 5 sources
            content = item["content"]
            
            # Look for sentences that might contain key information
            sentences = content.split('. ')
            for sentence in sentences:
                if len(sentence) > 50 and len(sentence) < 200:
                    # Look for sentences with numbers, important keywords
                    if any(word in sentence.lower() for word in ['study', 'research', 'found', 'shows', 'percent', '%', 'significant']):
                        key_points.append(sentence.strip() + '.')
                        if len(key_points) >= 10:  # Limit key points
                            break
            
            if len(key_points) >= 10:
                break
        
        return key_points
    
    def _generate_summary(self, content_list: List[Dict[str, Any]]) -> str:
        """Generate a summary of the research findings"""
        if not content_list:
            return ""
        
        # Simple summary generation (first significant paragraphs from each source)
        summary_parts = []
        
        for item in content_list[:3]:  # Top 3 sources
            content = item["content"]
            # Take first substantial paragraph
            paragraphs = content.split('\n\n')
            for paragraph in paragraphs:
                if len(paragraph) > 100:
                    summary_parts.append(paragraph[:300] + "...")
                    break
        
        return "\n\n".join(summary_parts)
    
    def _identify_contradictions(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Identify potential contradictions in the research (simplified)"""
        # This is a simplified implementation
        # In practice, you'd use NLP to identify conflicting statements
        contradictions = []
        
        # Look for conflicting keywords/phrases
        conflicting_patterns = [
            (["increased", "rising", "growing"], ["decreased", "falling", "declining"]),
            (["effective", "successful"], ["ineffective", "failed"]),
            (["safe"], ["dangerous", "harmful"]),
            (["proven"], ["unproven", "disputed"])
        ]
        
        for positive_terms, negative_terms in conflicting_patterns:
            positive_sources = []
            negative_sources = []
            
            for item in content_list:
                content = item["content"].lower()
                if any(term in content for term in positive_terms):
                    positive_sources.append(item)
                if any(term in content for term in negative_terms):
                    negative_sources.append(item)
            
            if positive_sources and negative_sources:
                contradictions.append({
                    "type": "conflicting_claims",
                    "positive_sources": [s["title"] for s in positive_sources[:2]],
                    "negative_sources": [s["title"] for s in negative_sources[:2]],
                    "description": f"Conflicting information found regarding {positive_terms[0]} vs {negative_terms[0]}"
                })
        
        return contradictions[:3]  # Limit contradictions
    
    async def _generate_analysis(self, query: str, research_data: Dict[str, Any], model_type: str = "default") -> str:
        """Generate comprehensive analysis using LLM"""
        try:
            # Prepare context for analysis
            sources_summary = ""
            for i, source in enumerate(research_data["sources"][:10], 1):
                content = source.get("scraped_content", "") or source.get("snippet", "")
                sources_summary += f"\nSource {i}: {source.get('title', 'Unknown')}\n"
                sources_summary += f"Content: {content[:4000]}...\n"  # Increased from 800 to 4000 for comprehensive context
                sources_summary += f"URL: {source.get('url', 'Unknown')}\n"
            
            analysis_prompt = f"""
Based on the following comprehensive research data about "{query}", provide a detailed and comprehensive analysis:

RESEARCH SOURCES:

{sources_summary}

KEY POINTS FOUND:
{research_data.get('key_points', [])}

SUMMARY:
{research_data.get('summary', 'No summary available')}

Please provide a detailed, comprehensive analysis that includes:

1. **Overview of the Current State of Knowledge**
   - Summarize the current understanding of the topic based on the research
   - Highlight key developments, trends, or changes mentioned in the sources

2. **Key Findings and Trends**  
   - Extract and explain the most important findings from the research
   - Identify patterns, trends, or recurring themes across sources
   - Include specific statistics, data points, or metrics where available

3. **Different Perspectives or Viewpoints**
   - Present different angles or approaches mentioned in the sources
   - Compare and contrast various viewpoints if they exist
   - Highlight any disagreements or debates in the field

4. **Implications and Significance**
   - Explain why these findings matter and their potential impact
   - Discuss real-world applications or consequences
   - Address how this information affects stakeholders or the broader field

5. **Areas Where More Research Might Be Needed**
   - Identify gaps in current knowledge based on the research
   - Suggest areas that need further investigation
   - Point out limitations or uncertainties mentioned in the sources

6. **Confidence Level in the Findings**
   - Assess the reliability and quality of the information
   - Note any conflicting information or areas of uncertainty
   - Provide an overall confidence assessment

IMPORTANT: Use the search context above to provide specific, detailed, and well-informed responses. Reference specific findings from the sources when possible. Make the analysis comprehensive, substantive, and valuable to the user.

Make the analysis comprehensive but well-structured, objective, and directly relevant to the user's query: "{query}"
"""
            
            # Generate analysis using the requested model type with increased tokens
            response = await self.llm_manager.generate_response(
                message=analysis_prompt,
                model_type=model_type,  # Use the passed model type instead of hardcoded
                max_tokens=8000,  # Increased from 1024 to 8000 for comprehensive responses
                temperature=0.3
            )
            
            return response["text"]
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return f"Analysis generation failed: {str(e)}"
    
    def _calculate_relevance(self, source: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for a source"""
        title = source.get("title", "").lower()
        content = source.get("scraped_content", "") or source.get("snippet", "")
        content = content.lower()
        
        query_words = query.lower().split()
        
        relevance = 0.0
        
        # Title relevance (weighted more heavily)
        for word in query_words:
            if word in title:
                relevance += 0.3
        
        # Content relevance
        for word in query_words:
            if word in content:
                relevance += 0.1
        
        # Length bonus (longer content might be more comprehensive)
        content_length = len(content)
        if content_length > 1000:
            relevance += 0.2
        elif content_length > 500:
            relevance += 0.1
        
        return min(relevance, 1.0)  # Cap at 1.0
    
    def _assess_credibility(self, source: Dict[str, Any]) -> float:
        """Assess credibility of a source"""
        url = source.get("url", "").lower()
        
        # Domain-based credibility scoring
        high_credibility_domains = [
            ".edu", ".gov", ".org", 
            "nature.com", "science.org", "ncbi.nlm.nih.gov",
            "reuters.com", "bbc.com", "npr.org"
        ]
        
        medium_credibility_domains = [
            "wikipedia.org", "britannica.com"
        ]
        
        credibility = 0.5  # Base score
        
        for domain in high_credibility_domains:
            if domain in url:
                credibility = 0.9
                break
        
        for domain in medium_credibility_domains:
            if domain in url:
                credibility = 0.7
                break
        
        return credibility
    
    def _assess_content_quality(self, source: Dict[str, Any]) -> float:
        """Assess quality of content"""
        content = source.get("scraped_content", "") or source.get("snippet", "")
        
        if not content:
            return 0.1
        
        quality = 0.5  # Base score
        
        # Length factor
        if len(content) > 1000:
            quality += 0.2
        elif len(content) > 500:
            quality += 0.1
        
        # Check for quality indicators
        quality_indicators = [
            "research", "study", "analysis", "data", "statistics",
            "peer-reviewed", "journal", "university", "institute"
        ]
        
        for indicator in quality_indicators:
            if indicator in content.lower():
                quality += 0.05
        
        return min(quality, 1.0)
    
    def _calculate_confidence_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the research"""
        if not sources:
            return 0.0
        
        total_score = 0.0
        for source in sources:
            relevance = source.get("relevance_score", 0.5)
            credibility = source.get("credibility_score", 0.5)
            quality = source.get("content_quality", 0.5)
            
            source_score = (relevance * 0.4 + credibility * 0.4 + quality * 0.2)
            total_score += source_score
        
        average_score = total_score / len(sources)
        
        # Adjust based on number of sources
        source_count_factor = min(len(sources) / 10, 1.0)  # More sources = higher confidence
        
        confidence = average_score * source_count_factor
        
        return round(confidence, 2)

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
        
        logger.info(f"Generated {len(unique_variants)} deep research variants: {unique_variants}")
        return unique_variants[:5]  # Limit to 5 variants to avoid too many API calls
