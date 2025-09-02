import os
from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional, List

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    api_title: str = "Local LLM Backend"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Model Configuration - FIXED to match your downloaded models
    models_config: Dict[str, Any] = {
        "report_generation": {
        "model_name": "Qwen2.5-14B-Instruct-Q6_K.gguf",  # ✅ UPDATED: Higher capacity model
            "model_path": "./models/report_generation/",       # ✅ FIXED: Correct path
            "context_length": 32768,
            "n_gpu_layers": -1,
            "n_threads": 16,
            "temperature": 0.7,  # Good for creative content
            "max_tokens": 8000,  # Updated to 8000 for longer sections
            "top_p": 0.9,  # Good balance
            "top_k": 50,   # Reasonable variety
        "description": "Qwen2.5-14B Q6_K for advanced report generation and detailed responses"
        },
        "vision": {
            "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",  # HuggingFace transformers model
            "model_path": "transformers://",  # Using HuggingFace transformers
            "context_length": 32768,
            "n_gpu_layers": -1,  # Not applicable for transformers
            "n_threads": 16,     # Not applicable for transformers
            "temperature": 0.8,
            "max_tokens": 8000,  # Updated to 8000
            "top_p": 0.95,
            "top_k": 100,
            "description": "Qwen2.5-VL 3B using HuggingFace transformers for advanced image analysis"
        },
        "coding": {
            "model_name": "qwen2.5-coder-7b-instruct-q6_k.gguf",  # ✅ FIXED: Matches your downloaded file
            "model_path": "./models/coding/",                      # ✅ FIXED: Correct path and case
            "context_length": 32768,
            "n_gpu_layers": -1,
            "n_threads": 16,
            "temperature": 0.1,  # Lower temperature for coding
            "max_tokens": 8000,  # Updated to 8000
            "top_p": 0.95,
            "top_k": 50,
            "description": "Qwen2.5-Coder 7B Q6_K for code generation and analysis"
        },
        "embedding": {
            "model_name": "Qwen3-Embedding-0.6B-Q8_0.gguf",  # ✅ FIXED: Matches downloaded file
            "model_path": "./models/embedding/",
            "context_length": 32768,
            "n_gpu_layers": -1,
            "n_threads": 16,
            "embedding_dimension": 1024,
            "batch_size": 512,
            "description": "Qwen3 Embedding 0.6B for chat history and semantic search"
        }
    }
    
    # Model recommendations - Updated to match your current setup
    recommended_models: Dict[str, Dict[str, str]] = {
        "report_generation": {
            "qwen2.5-14b-instruct-q6_k": "✅ Already downloaded: Qwen2.5-14B-Instruct-Q6_K.gguf"
        },
        "coding": {
            "qwen2.5-coder-7b-q6_k": "✅ Already downloaded: qwen2.5-coder-7b-instruct-q6_k.gguf"
        },
        "vision": {
            "qwen2.5-vl-3b-transformers": "✅ Using HuggingFace Transformers: Qwen/Qwen2.5-VL-3B-Instruct (auto-download)"
        },
        "embedding": {
            "qwen3-embedding-0.6b-q8_0": "✅ Already downloaded: Qwen3-Embedding-0.6B-Q8_0.gguf"
        }
    }
    
    # Memory Management (optimized for M1 MacBook)
    max_memory_usage: float = 0.75
    model_cache_size: int = 2
    
    # File Processing
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = [
        ".txt", ".pdf", ".docx", ".doc", ".md", ".rtf",
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff",
        ".py", ".js", ".html", ".css", ".json", ".xml", ".yaml", ".yml",
        ".cpp", ".c", ".java", ".go", ".rs", ".php", ".rb", ".swift"
    ]
    
    # Web Search Settings
    web_search_enabled: bool = True
    max_search_results: int = 10
    search_timeout: int = 20
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # List of user agents for rotation
    user_agents: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
    ]

    # Model providers
    openai_api_key: Optional[str] = None
    
    # Brave Search API Settings
    brave_search_api_key: str = "BSAT_JD5O8SBAw39qbyG0NVnS4Iz3G5"
    brave_search_enabled: bool = True
    brave_rate_limit_per_second: float = 1.0  # 1 request per second as per API limit
    
    # Privacy Settings for Web Search
    privacy_mode: bool = True
    anonymize_queries: bool = True
    log_search_queries: bool = False
    use_proxy: bool = False
    proxy_url: Optional[str] = None
    
    # Search Engine Preferences
    preferred_search_engines: list = ["brave", "bing", "duckduckgo", "startpage"]
    fallback_to_bing: bool = True
    
    # Query Anonymization Settings
    remove_personal_pronouns: bool = True
    sanitize_sensitive_data: bool = True
    anonymize_tracking_params: bool = True
    
    # Deep Research Settings
    max_research_depth: int = 3
    max_sources_per_research: int = 20
    research_timeout: int = 300
    
    # Report Generation
    report_templates_path: str = "./templates/"
    output_path: str = "./temp_files/"
    
    # Advanced Agentic Report Settings
    agentic_report_settings: Dict[str, Any] = {
        "enabled": True,
        "max_iterations": 5,
        "research_depth": 3,
        "quality_threshold": 0.8,
        "auto_refinement": True,
        "multi_perspective_analysis": True,
        "fact_checking": True,
        "citation_verification": True,
        "report_types": ["research", "analysis", "technical", "business"],
        "output_formats": ["pdf", "docx", "html", "markdown"],
        "max_sources_per_section": 10,
        "confidence_scoring": True,
        "peer_review_simulation": False,
        "collaborative_editing": False,
        "real_time_updates": False
    }
    
    # Chart Generation Settings
    chart_generation_settings: Dict[str, Any] = {
        "enabled": True,
        "chart_types": ["bar", "line", "pie", "scatter", "histogram", "box", "heatmap"],
        "default_theme": "modern",
        "color_schemes": ["viridis", "plasma", "inferno", "magma", "blues", "greens"],
        "max_data_points": 10000,
        "auto_scaling": True,
        "interactive_charts": True,
        "export_formats": ["png", "svg", "pdf", "html"],
        "resolution": {"width": 1200, "height": 800},
        "font_settings": {
            "family": "Arial",
            "size": 12,
            "title_size": 16
        },
        "animation": {
            "enabled": True,
            "duration": 1000
        },
        "data_processing": {
            "auto_clean": True,
            "handle_missing": "interpolate",
            "outlier_detection": True
        }
    }
    
    # Paths
    models_base_path: str = "./models/"
    temp_files_path: str = "./temp_files/"
    logs_path: str = "./logs/"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Performance Settings (RTX 3090 optimized)
    llama_cpp_settings: Dict[str, Any] = {
        "use_mlock": True,
        "use_mmap": True,
        "n_batch": 1024,
        "n_ctx": 8192,
        "f16_kv": True,
        "logits_all": False,
        "vocab_only": False,
        "verbose": False
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        protected_namespaces = ()

# Create a global settings instance
settings = Settings()

# Model size estimations for M1 MacBook (in GB)
MODEL_SIZES = {
    "7B_Q4_K_M": 4.1,
    "7B_Q5_K_M": 4.8,
    "7B_Q6_K": 5.5,
    "7B_Q8_0": 7.2,
    "13B_Q4_K_M": 7.9,
    "13B_Q5_K_M": 9.2,
    "13B_Q8_0": 14.1,
    "30B_Q4_K_M": 19.5,
    "70B_Q4_K_M": 42.0
}

# Performance expectations for RTX 3090 (tokens/second)
PERFORMANCE_ESTIMATES = {
    "RTX_3090_24GB": {
        "7B_Q4_K_M": {"pp": 120, "tg": 65},
        "7B_Q5_K_M": {"pp": 100, "tg": 55},
        "7B_Q6_K": {"pp": 85, "tg": 45},
        "7B_Q8_0": {"pp": 80, "tg": 45},
        "13B_Q4_K_M": {"pp": 70, "tg": 40}
    }
}

def get_optimal_model_config(available_memory_gb: float, model_type: str = "chat") -> Dict[str, Any]:
    """Get optimal model configuration based on available memory for RTX 3090"""
    
    if available_memory_gb >= 20:  # RTX 3090 has 24GB VRAM
        return {
            "model_size": "7B-13B",
            "quantization": "Q5_K_M",
            "estimated_memory": "5-9GB",
            "estimated_performance": "40-100 t/s"
        }
    elif available_memory_gb >= 16:
        return {
            "model_size": "7B",
            "quantization": "Q4_K_M",
            "estimated_memory": "4-6GB",
            "estimated_performance": "65-120 t/s"
        }
    else:
        return {
            "model_size": "7B",
            "quantization": "Q4_0",
            "estimated_memory": "3-4GB", 
            "estimated_performance": "80-150 t/s"
        }
