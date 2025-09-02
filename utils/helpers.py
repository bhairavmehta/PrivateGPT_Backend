import os
import re
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
import base64
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate if file type is allowed"""
    file_extension = get_file_extension(filename)
    return file_extension.lower() in [ext.lower() for ext in allowed_extensions]

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix

def get_mime_type(filename: str) -> str:
    """Get MIME type from filename"""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove path separators and dangerous characters
    filename = os.path.basename(filename)
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\.{2,}', '.', filename)  # Multiple dots to single dot
    filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def generate_file_hash(content: bytes) -> str:
    """Generate MD5 hash for file content"""
    return hashlib.md5(content).hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def extract_text_preview(text: str, max_length: int = 200) -> str:
    """Extract preview text from larger content"""
    if len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    preview = text[:max_length]
    last_space = preview.rfind(' ')
    
    if last_space > max_length * 0.8:  # If space is reasonably close to end
        preview = preview[:last_space]
    
    return preview + "..."

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove control characters except tabs and newlines
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    return text

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for LLM context"""
    if not results:
        return "No search results found."
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        snippet = result.get('snippet', 'No description available')
        
        formatted_result = f"""
Result {i}:
Title: {title}
URL: {url}
Description: {snippet}
"""
        formatted_results.append(formatted_result.strip())
    
    return "\n\n".join(formatted_results)

def format_research_summary(research_data: Dict[str, Any]) -> str:
    """Format research data for display"""
    summary_parts = []
    
    if research_data.get('summary'):
        summary_parts.append(f"**Summary:** {research_data['summary']}")
    
    if research_data.get('key_points'):
        key_points = research_data['key_points']
        if isinstance(key_points, list):
            points_text = "\n".join([f"• {point}" for point in key_points])
        else:
            points_text = str(key_points)
        summary_parts.append(f"**Key Points:**\n{points_text}")
    
    if research_data.get('contradictions'):
        contradictions = research_data['contradictions']
        if isinstance(contradictions, list) and contradictions:
            contra_text = "\n".join([f"• {contra.get('description', str(contra))}" for contra in contradictions])
            summary_parts.append(f"**Contradictions Found:**\n{contra_text}")
    
    if research_data.get('confidence_score'):
        summary_parts.append(f"**Confidence Score:** {research_data['confidence_score']:.2f}")
    
    return "\n\n".join(summary_parts)

def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode image file as base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None

def decode_base64_image(base64_string: str, output_path: str) -> bool:
    """Decode base64 string to image file"""
    try:
        image_data = base64.b64decode(base64_string)
        with open(output_path, 'wb') as image_file:
            image_file.write(image_data)
        return True
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return False

def is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico', '.svg'}
    extension = get_file_extension(filename).lower()
    return extension in image_extensions

def is_document_file(filename: str) -> bool:
    """Check if file is a document based on extension"""
    doc_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.odt', '.pages'}
    extension = get_file_extension(filename).lower()
    return extension in doc_extensions

def is_code_file(filename: str) -> bool:
    """Check if file is a code file based on extension"""
    code_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss', '.less',
        '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go',
        '.rs', '.swift', '.kt', '.scala', '.clj', '.hs', '.ml', '.r',
        '.sql', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini',
        '.sh', '.bat', '.ps1', '.vim', '.lua', '.perl', '.dart'
    }
    extension = get_file_extension(filename).lower()
    return extension in code_extensions

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text content"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls

def clean_url(url: str) -> str:
    """Clean and normalize URL"""
    url = url.strip()
    
    # Remove common tracking parameters
    tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid', 'gclid']
    
    if '?' in url:
        base_url, query_string = url.split('?', 1)
        params = query_string.split('&')
        cleaned_params = [param for param in params if not any(param.startswith(f"{tp}=") for tp in tracking_params)]
        
        if cleaned_params:
            url = f"{base_url}?{'&'.join(cleaned_params)}"
        else:
            url = base_url
    
    return url

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def parse_model_type(model_input: str) -> str:
    """Parse and validate model type input"""
    model_input = model_input.lower().strip()
    
    model_mapping = {
        'chat': 'chat',
        'c': 'chat',
        'conversation': 'chat',
        'vision': 'vision',
        'v': 'vision',
        'image': 'vision',
        'img': 'vision',
        'visual': 'vision',
        'coding': 'coding',
        'code': 'coding',
        'programming': 'coding',
        'dev': 'coding',
        'development': 'coding',
        'report': 'report',
        'r': 'report',
        'document': 'report',
        'doc': 'report'
    }
    
    return model_mapping.get(model_input, 'chat')  # Default to chat

def validate_temperature(temperature: float) -> float:
    """Validate and clamp temperature value"""
    return max(0.0, min(2.0, temperature))

def validate_max_tokens(max_tokens: int, model_type: str = "chat") -> int:
    """Validate and clamp max_tokens value"""
    # Model-specific limits - updated to 8000 for better responses
    limits = {
        "chat": 8000,  # Updated from 4096 to 8000
        "vision": 8000,  # Updated from 2048 to 8000
        "coding": 8000,  # Kept at 8000
        "report": 8000   # Updated from 4096 to 8000
    }
    
    max_limit = limits.get(model_type, 8000)  # Default to 8000 instead of 4096
    return max(1, min(max_limit, max_tokens))

def create_error_response(error_message: str, error_code: str = "GENERIC_ERROR") -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": True,
        "error_code": error_code,
        "message": error_message,
        "timestamp": format_timestamp()
    }

def create_success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response"""
    response = {
        "success": True,
        "message": message,
        "timestamp": format_timestamp()
    }
    
    if data is not None:
        response["data"] = data
    
    return response

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes"""
    word_count = len(text.split())
    reading_time = max(1, round(word_count / words_per_minute))
    return reading_time

def extract_code_language(filename: str, content: str = None) -> str:
    """Extract programming language from filename or content"""
    extension = get_file_extension(filename).lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.clj': 'clojure',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.r': 'r',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.sh': 'bash',
        '.bat': 'batch',
        '.ps1': 'powershell'
    }
    
    return language_map.get(extension, 'text')

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            break_point = max(last_period, last_newline)
            if break_point > start + chunk_size * 0.5:  # If break point is reasonable
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries with later values taking precedence"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

def safe_filename_from_url(url: str) -> str:
    """Generate safe filename from URL"""
    # Extract path from URL
    from urllib.parse import urlparse, unquote
    
    parsed = urlparse(url)
    path = unquote(parsed.path)
    
    # Get the last part of the path
    filename = os.path.basename(path) or "webpage"
    
    # Remove extension if it's not useful
    if filename.endswith('.html') or filename.endswith('.htm'):
        filename = filename.rsplit('.', 1)[0]
    
    # Sanitize filename
    filename = sanitize_filename(filename)
    
    # Ensure it's not too long
    if len(filename) > 50:
        filename = filename[:50]
    
    return filename or "webpage"
