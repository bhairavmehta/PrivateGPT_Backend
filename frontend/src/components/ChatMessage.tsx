import React from 'react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { ExternalLink, User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ChatMessageProps {
  message: {
    id: string;
    content: string;
    isUser: boolean;
    isCodeResponse?: boolean;
    sources?: Array<{
      title: string;
      url: string;
      snippet?: string;
    }>;
    search_query?: string;
    attachments?: File[];
    // New source API fields
    source_id?: string;
    source_count?: number;
    search_type?: 'web_search' | 'deep_research';
  };
  isCodeResponse?: boolean;
  isStreaming?: boolean;
  sources?: Array<{
    title: string;
    url: string;
    snippet?: string;
  }>;
  search_query?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ 
  message, 
  isCodeResponse, 
  isStreaming, 
  sources, 
  search_query 
}) => {
  // Use sources from props first, then from message
  const displaySources = sources || message.sources;
  const displaySearchQuery = search_query || message.search_query;
  const searchType = message.search_type;
  const sourceCount = message.source_count || displaySources?.length || 0;

  // Get search type display name
  const getSearchTypeDisplay = (type?: string) => {
    switch (type) {
      case 'web_search':
        return { icon: '🌐', name: 'Web Search' };
      case 'deep_research':
        return { icon: '🧠', name: 'Deep Research' };
      default:
        return { icon: '📚', name: 'Reference' };
    }
  };

  const searchTypeInfo = getSearchTypeDisplay(searchType);

  return (
    <div className={`mb-6 ${message.isUser ? 'ml-12' : 'mr-12'}`}>
      <div className={`flex gap-3 ${message.isUser ? 'justify-end' : 'justify-start'}`}>
        {!message.isUser && (
          <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0 mt-1">
            <Bot className="w-4 h-4 text-primary-foreground" />
          </div>
        )}
        
        <div className={`max-w-[85%] ${message.isUser ? 'order-first' : ''}`}>
          <Card className={`${
            message.isUser 
              ? 'bg-primary text-primary-foreground ml-auto' 
              : 'bg-muted'
          }`}>
            <CardContent className="p-4">
              {message.isUser ? (
                <div className="whitespace-pre-wrap">{message.content}</div>
              ) : (
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <ReactMarkdown
                    components={{
                      code: ({ className, children, ...props }: any) => {
                        const match = /language-(\w+)/.exec(className || '');
                        const isInline = !match;
                        
                        if (isInline) {
                          return (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        }
                        
                        return (
                          <SyntaxHighlighter
                            style={vscDarkPlus as any}
                            language={match[1]}
                            PreTag="div"
                          >
                            {String(children).replace(/\n$/, '')}
                          </SyntaxHighlighter>
                        );
                      },
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              )}
              
              {/* Display file attachments */}
              {message.attachments && message.attachments.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {message.attachments.map((file, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      📎 {file.name}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Display search query if available */}
          {!message.isUser && displaySearchQuery && (
            <div className="mt-2 text-xs text-muted-foreground">
              🔍 Search Query: "{displaySearchQuery}"
            </div>
          )}

          {/* Display reference sources */}
          {!message.isUser && displaySources && displaySources.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium text-foreground mb-2 flex items-center gap-2">
                {searchTypeInfo.icon} {searchTypeInfo.name} Sources
                <Badge variant="outline" className="text-xs">
                  {sourceCount} source{sourceCount !== 1 ? 's' : ''}
                </Badge>
                {searchType && (
                  <Badge variant="secondary" className="text-xs">
                    {searchType === 'deep_research' ? 'Enhanced' : 'Standard'}
                  </Badge>
                )}
              </div>
              <div className="space-y-2">
                {displaySources.map((source, index) => (
                  <Card key={index} className="border border-border/50">
                    <CardContent className="p-3">
                      <div className="flex items-start gap-3">
                        <Badge 
                          variant="secondary" 
                          className="text-xs px-2 py-1 font-mono flex-shrink-0"
                        >
                          {index + 1}
                        </Badge>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between gap-2">
                            <h4 className="text-sm font-medium text-foreground leading-tight line-clamp-2">
                              {source.title}
                            </h4>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0 flex-shrink-0"
                              onClick={() => window.open(source.url, '_blank')}
                              title="Open source in new tab"
                            >
                              <ExternalLink className="w-3 h-3" />
                            </Button>
                          </div>
                          {source.snippet && (
                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                              {source.snippet}
                            </p>
                          )}
                          <p className="text-xs text-muted-foreground mt-1 truncate">
                            {source.url}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
              {searchType && (
                <div className="mt-2 text-xs text-muted-foreground">
                  ✨ Powered by {searchType === 'deep_research' ? 'Advanced Deep Research' : 'Fast Web Search'}
                </div>
              )}
            </div>
          )}

          {/* Streaming indicator */}
          {isStreaming && (
            <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
              <div className="animate-pulse w-1 h-1 bg-current rounded-full"></div>
              <div className="animate-pulse w-1 h-1 bg-current rounded-full" style={{ animationDelay: '0.2s' }}></div>
              <div className="animate-pulse w-1 h-1 bg-current rounded-full" style={{ animationDelay: '0.4s' }}></div>
              <span>AI is typing...</span>
            </div>
          )}
        </div>
        
        {message.isUser && (
          <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center flex-shrink-0 mt-1">
            <User className="w-4 h-4 text-secondary-foreground" />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage; 