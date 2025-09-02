import React from 'react';

// Custom markdown parser for common elements
const parseMarkdown = (text: string) => {
  if (!text) return '';
  
  let html = text;
  
  // Headers with numbered sections (### 1. **Title**) 
  html = html.replace(/^### (\d+\.\s*)\*\*(.*?)\*\*/gm, '<h3 class="text-lg font-semibold mt-6 mb-3 text-foreground flex items-center gap-2"><span class="text-blue-600 font-bold">$1</span><strong>$2</strong></h3>');
  
  // Regular headers (### to h3, ## to h2, # to h1)
  html = html.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold mt-6 mb-3 text-foreground">$1</h3>');
  html = html.replace(/^## (.*$)/gm, '<h2 class="text-xl font-semibold mt-6 mb-3 text-foreground">$1</h2>');
  html = html.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold mt-6 mb-4 text-foreground">$1</h1>');
  
  // Bold text (**text** or __text__)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-foreground">$1</strong>');
  html = html.replace(/__(.*?)__/g, '<strong class="font-semibold text-foreground">$1</strong>');
  
  // Italic text (*text* or _text_) - but avoid conflicts with bold
  html = html.replace(/(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)/g, '<em class="italic">$1</em>');
  html = html.replace(/(?<!_)_(?!_)([^_]+?)_(?!_)/g, '<em class="italic">$1</em>');
  
  // Links [text](url)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer">$1</a>');
  
  // Unordered lists (- item or * item)
  html = html.replace(/^[\s]*[-*+]\s(.+)$/gm, '<li class="ml-4 list-disc my-1">$1</li>');
  
  // Numbered lists (1. item, 2. item, etc)
  html = html.replace(/^[\s]*(\d+\.)\s(.+)$/gm, '<li class="ml-4 list-decimal my-1"><span class="font-medium text-blue-600 dark:text-blue-400">$1</span> $2</li>');
  
  // Wrap consecutive list items in ul/ol tags
  html = html.replace(/((?:<li class="ml-4 list-disc[^"]*"[^>]*>[^<]*<\/li>\s*)+)/g, '<ul class="my-3 space-y-1">$1</ul>');
  html = html.replace(/((?:<li class="ml-4 list-decimal[^"]*"[^>]*>.*?<\/li>\s*)+)/gs, '<ol class="my-3 space-y-1">$1</ol>');
  
  // Inline code `code`
  html = html.replace(/`([^`]+)`/g, '<code class="bg-muted text-muted-foreground px-1.5 py-0.5 rounded text-sm font-mono">$1</code>');
  
  // Line breaks (double line breaks become paragraph breaks)
  html = html.replace(/\n\n/g, '</p><p class="mb-3">');
  html = html.replace(/\n/g, '<br>');
  
  // Wrap in paragraph if not already wrapped
  if (!html.includes('<h1>') && !html.includes('<h2>') && !html.includes('<h3>') && !html.includes('<ul>') && !html.includes('<ol>')) {
    html = '<p class="mb-3">' + html + '</p>';
  } else {
    html = '<div>' + html + '</div>';
  }
  
  return html;
};

const MarkdownTest: React.FC = () => {
  const testMarkdown = `There are many ways to make the Snake game more advanced. Here are some suggestions:

### 1. **Speed Increase Mechanism**
As the player progresses, increase the speed of the snake.

### 2. **Dynamic Food Spawning**
Place food items at different locations on the screen, possibly with different colors or properties.

Here's a code snippet:

\`\`\`python
if length_of_snake > 5:
    snake_speed = 20
elif length_of_snake > 10:
    snake_speed = 30
# Add more conditions as needed
\`\`\`

Additional features:
- Power-ups that give special abilities
- Obstacles that the snake must avoid
- Multiple game modes
- High score tracking

You can also use \`pygame.time.Clock()\` to control the frame rate.`;

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Markdown Rendering Test</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h2 className="text-lg font-semibold mb-3">Raw Markdown:</h2>
          <pre className="bg-muted p-4 rounded text-sm whitespace-pre-wrap">
            {testMarkdown}
          </pre>
        </div>
        
        <div>
          <h2 className="text-lg font-semibold mb-3">Rendered Output:</h2>
          <div 
            className="bg-card border border-border p-4 rounded prose prose-sm max-w-none break-words overflow-hidden"
            dangerouslySetInnerHTML={{ __html: parseMarkdown(testMarkdown) }}
          />
        </div>
      </div>
    </div>
  );
};

export default MarkdownTest; 