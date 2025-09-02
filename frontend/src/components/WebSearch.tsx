
import React, { useState } from 'react';
import { Search, Globe } from 'lucide-react';

interface WebSearchProps {
  onSearchResults: (results: string) => void;
  disabled?: boolean;
}

const WebSearch: React.FC<WebSearchProps> = ({ onSearchResults, disabled }) => {
  const [isSearching, setIsSearching] = useState(false);

  const performWebSearch = async (query: string) => {
    setIsSearching(true);
    console.log('Performing web search for:', query);

    try {
      // This is a mock implementation - in a real app, you'd integrate with a search API
      // like Google Custom Search, Bing Web Search, or SerpAPI
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate API call
      
      const mockResults = `Web search results for "${query}":

1. Latest information about ${query} from recent sources
2. Comprehensive overview and current developments
3. Expert analysis and trending discussions
4. Related news and updates from the past week

Note: This is a mock implementation. To enable real web search, integrate with a search API like Google Custom Search or SerpAPI.`;

      onSearchResults(mockResults);
    } catch (error) {
      console.error('Web search error:', error);
      onSearchResults('Web search temporarily unavailable. Please try again later.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleClick = () => {
    const query = prompt('Enter your search query:');
    if (query?.trim()) {
      performWebSearch(query.trim());
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={disabled || isSearching}
      className="p-2 text-green-500 hover:text-green-700 dark:text-green-400 dark:hover:text-green-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      aria-label="Web search"
      title={isSearching ? 'Searching...' : 'Perform web search'}
    >
      {isSearching ? (
        <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
      ) : (
        <Globe className="w-5 h-5" />
      )}
    </button>
  );
};

export default WebSearch;
