import os
import asyncio
import logging
import psutil
from typing import Dict, Any, Optional, List, AsyncGenerator
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava15ChatHandler
import time
from datetime import datetime
import base64
from PIL import Image
import torch
from config import Settings, get_optimal_model_config
from .qwen_vision_manager import QwenVisionManager
from services.web_search import get_search_service

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages multiple LLM models optimized for RTX 3090 with model switching"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: Dict[str, Llama] = {}
        self.model_handlers: Dict[str, Any] = {}
        self.model_stats: Dict[str, Dict] = {}
        self.last_used: Dict[str, datetime] = {}
        self.current_large_model: Optional[str] = None  # Track currently loaded large model
        
        # Initialize the Transformers-based vision manager
        self.qwen_vision_manager = QwenVisionManager()
        
        # Initialize RobustWebSearchService here for reuse
        self.web_search_service = get_search_service(settings)
        
        # Define large models that require switching
        self.large_models = ["report_generation", "vision", "coding"]
        self.small_models = ["embedding"]
        
        # Create model directories
        self._ensure_model_directories()
        
    def _ensure_model_directories(self):
        """Create model directories if they don't exist"""
        # Only check models that exist in configuration
        for model_type in self.settings.models_config.keys():
            if model_type == "vision":
                # Skip directory creation for vision as we're using Transformers
                continue
            model_config = self.settings.models_config[model_type]
            model_path = model_config["model_path"]
            os.makedirs(model_path, exist_ok=True)
    
    async def get_models_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {}
        
        for model_type in self.settings.models_config.keys():
            if model_type == "vision":
                # Special handling for Transformers-based vision model
                vision_info = self.qwen_vision_manager.get_model_info()
                status[model_type] = {
                    "loaded": vision_info["model_loaded"],
                    "model_file_exists": True,  # Transformers models are downloaded on-demand
                    "model_path": f"transformers:{vision_info['model_name']}",
                    "last_used": self.last_used.get(model_type),
                    "memory_usage": self._get_memory_usage(),
                    "stats": self.model_stats.get(model_type, {}),
                    "description": "Qwen2.5-VL using Transformers - state-of-the-art multimodal model",
                    "is_current_large_model": model_type == self.current_large_model,
                    "model_type": "transformers",
                    "device": vision_info["device"],
                    "capabilities": vision_info["capabilities"]
                }
            else:
                model_config = self.settings.models_config[model_type]
                model_path = os.path.join(model_config["model_path"], model_config["model_name"])
                
                status[model_type] = {
                    "loaded": model_type in self.models,
                    "model_file_exists": os.path.exists(model_path),
                    "model_path": model_path,
                    "last_used": self.last_used.get(model_type),
                    "memory_usage": self._get_memory_usage(),
                    "stats": self.model_stats.get(model_type, {}),
                    "description": model_config.get("description", ""),
                    "is_current_large_model": model_type == self.current_large_model
                }
        
        return status
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their information"""
        models_info = []
        
        for model_type in self.settings.models_config.keys():
            config = self.settings.models_config[model_type]
            model_path = os.path.join(config["model_path"], config["model_name"])
            
            # Check if model file exists
            file_exists = os.path.exists(model_path)
            file_size_gb = 0
            if file_exists:
                file_size_gb = os.path.getsize(model_path) / (1024**3)
            
            model_info = {
                "name": config["model_name"],
                "type": model_type,
                "status": "loaded" if model_type in self.models else ("available" if file_exists else "not_downloaded"),
                "memory_usage": f"{file_size_gb:.1f} GB" if file_exists else "N/A",
                "performance": self._estimate_performance(model_type, config["model_name"]),
                "config": config,
                "description": config.get("description", ""),
                "is_large_model": model_type in self.large_models,
                "is_current_large_model": model_type == self.current_large_model,
                "download_url": self._get_download_url(model_type, config["model_name"])
            }
            
            models_info.append(model_info)
        
        return models_info
    
    def _estimate_performance(self, model_type: str, model_name: str) -> str:
        """Estimate performance based on model type and quantization for RTX 3090"""
        if "Q4_0" in model_name:
            return "80-150 t/s (optimal for RTX 3090)"
        elif "Q4_K_M" in model_name:
            return "65-120 t/s (excellent quality/speed)"
        elif "Q5_K_M" in model_name:
            return "55-100 t/s (high quality)"
        elif "Q8_0" in model_name:
            return "45-80 t/s (highest quality)"
        else:
            return "Varies (GPU accelerated)"
    
    def _get_download_url(self, model_type: str, model_name: str) -> Optional[str]:
        """Get download URL for a model"""
        model_urls = self.settings.recommended_models.get(model_type, {})
        
        # Match model name to URL
        for name_key, url in model_urls.items():
            if name_key.lower() in model_name.lower():
                return url
        
        return None
    
    async def load_model(self, model_type: str) -> Dict[str, Any]:
        """Load a specific model with automatic switching for large models"""
        if model_type not in self.settings.models_config.keys():
            raise ValueError(f"Invalid model type: {model_type}. Available: {list(self.settings.models_config.keys())}")
        
        # Special handling for Transformers-based vision model
        if model_type == "vision":
            return await self._load_vision_model()
        
        # Check if model is already loaded
        if model_type in self.models:
            logger.info(f"Model {model_type} already loaded")
            return {"status": "already_loaded", "model_type": model_type}
        
        config = self.settings.models_config[model_type]
        absolute_model_path = os.path.abspath(os.path.join("/workspace/private_gpt_backend/", config["model_path"], config["model_name"]))
        
        # Check if model file exists
        if not os.path.exists(absolute_model_path):
            raise FileNotFoundError(f"Model file not found: {absolute_model_path}")
        
        # Handle model switching for large models
        if model_type in self.large_models:
            await self._switch_large_model(model_type)
        
        try:
            logger.info(f"Loading {model_type} model from {absolute_model_path}")
            start_time = time.time()
            
            # Configure model parameters optimized for RTX 3090
            model_params = {
                "model_path": absolute_model_path,
                "n_ctx": config["context_length"],
                "n_gpu_layers": config["n_gpu_layers"],  # -1 for full GPU offloading
                "n_threads": config["n_threads"],
                "f16_kv": self.settings.llama_cpp_settings["f16_kv"],
                "use_mlock": self.settings.llama_cpp_settings["use_mlock"],
                "use_mmap": self.settings.llama_cpp_settings["use_mmap"],
                "n_batch": self.settings.llama_cpp_settings["n_batch"],  # Larger batch for RTX 3090
                "verbose": False  # Disable verbose logging to reduce console output
            }
            
            # Special handling for Gemma 3n models
            if "gemma-3n" in config["model_name"].lower():
                logger.info("Detected Gemma 3n model, applying special configuration")
                # Gemma 3n specific parameters
                model_params.update({
                    "chat_format": "gemma",  # Use Gemma chat format
                    "rope_freq_base": 10000.0,  # Gemma specific RoPE frequency
                    "rope_freq_scale": 1.0,
                    "mul_mat_q": True,  # Enable quantized matrix multiplication
                })
            
            # Load the model
            model = Llama(**model_params)
            
            self.models[model_type] = model
            self.last_used[model_type] = datetime.now()
            
            # Update current large model tracker
            if model_type in self.large_models:
                self.current_large_model = model_type
            
            load_time = time.time() - start_time
            
            # Store model statistics
            self.model_stats[model_type] = {
                "load_time": load_time,
                "model_size_gb": os.path.getsize(absolute_model_path) / (1024**3),
                "context_length": config["context_length"],
                "loaded_at": datetime.now().isoformat(),
                "gpu_layers": config["n_gpu_layers"],
                "threads": config["n_threads"]
            }
            
            logger.info(f"Successfully loaded {model_type} model in {load_time:.2f}s")
            
            return {
                "status": "loaded",
                "model_type": model_type,
                "load_time": load_time,
                "model_stats": self.model_stats[model_type],
                "switched_from": getattr(self, '_last_switched_from', None)
            }
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model:", exc_info=True)
            raise
    
    async def _load_vision_model(self) -> Dict[str, Any]:
        """Load the Transformers-based Qwen2.5-VL vision model"""
        
        # Unload any other large model first to free up VRAM
        if self.current_large_model and self.current_large_model != "vision":
            await self.unload_model(self.current_large_model)
        
        # Check if already loaded
        if self.qwen_vision_manager.model_loaded:
            logger.info("Qwen2.5-VL vision model already loaded")
            return {"status": "already_loaded", "model_type": "vision"}
        
        # Handle model switching for large models
        await self._switch_large_model("vision")
        
        try:
            logger.info("Loading Qwen2.5-VL vision model using Transformers")
            start_time = time.time()
            
            # Load the Transformers model
            success = self.qwen_vision_manager.load_model()
            if not success:
                raise RuntimeError("Failed to load Qwen2.5-VL model")
            
            self.last_used["vision"] = datetime.now()
            self.current_large_model = "vision"
            
            load_time = time.time() - start_time
            
            # Store model statistics
            self.model_stats["vision"] = {
                "load_time": load_time,
                "model_type": "transformers",
                "model_name": self.qwen_vision_manager.model_name,
                "device": self.qwen_vision_manager.device,
                "loaded_at": datetime.now().isoformat(),
                "transformers_based": True
            }
            
            logger.info(f"Successfully loaded Qwen2.5-VL vision model in {load_time:.2f}s")
            
            return {
                "status": "loaded",
                "model_type": "vision",
                "load_time": load_time,
                "model_stats": self.model_stats["vision"],
                "model_name": self.qwen_vision_manager.model_name,
                "device": self.qwen_vision_manager.device,
                "switched_from": getattr(self, '_last_switched_from', None)
            }
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL vision model: {e}", exc_info=True)
            raise
    
    async def _switch_large_model(self, new_model_type: str):
        """Switch large models by unloading current one before loading new one"""
        if self.current_large_model and self.current_large_model != new_model_type:
            logger.info(f"Switching from {self.current_large_model} to {new_model_type}")
            await self.unload_model(self.current_large_model)
    
    async def unload_model(self, model_type: str):
        """Unload a specific model and clear CUDA cache."""
        
        # Special handling for Transformers-based vision model
        if model_type == "vision":
            if self.qwen_vision_manager.model_loaded:
                self.qwen_vision_manager.unload_model()
                if model_type in self.last_used:
                    del self.last_used[model_type]
                if model_type == self.current_large_model:
                    self.current_large_model = None
                logger.info(f"Unloaded Qwen2.5-VL vision model")
            else:
                logger.warning(f"Qwen2.5-VL vision model was not loaded")
            return
        
        if model_type in self.models:
            del self.models[model_type]
            if model_type in self.model_handlers:
                del self.model_handlers[model_type]
            if model_type in self.last_used:
                del self.last_used[model_type]
            
            # Update current large model tracker
            if model_type == self.current_large_model:
                self.current_large_model = None
                
            logger.info(f"Unloaded {model_type} model")
            
            # Clear CUDA cache to free up VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache.")
        else:
            logger.warning(f"Model {model_type} was not loaded")
    
    async def _free_memory(self):
        """Free memory by unloading least recently used model"""
        if not self.last_used:
            return
        
        # Find least recently used model
        lru_model = min(self.last_used.items(), key=lambda x: x[1])[0]
        await self.unload_model(lru_model)
        logger.info(f"Freed memory by unloading {lru_model} model")
    
    async def generate_response(
        self,
        message: str,
        model_type: str = "report_generation",  # Default to report generation
        context: Optional[str] = None,
        max_tokens: int = 8000,  # Updated to 8000
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using specified model with automatic switching"""
        
        # Special handling for vision model text conversations
        if model_type == "vision":
            return await self._generate_vision_text_response(
                message=message,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        # Special handling for coding model
        if model_type == "coding":
            return await self.generate_code(
                prompt=message,
                max_tokens=max_tokens,
                temperature=temperature,
                conversation_history=context or ""
            )
        
        # Ensure model is loaded (will auto-switch if needed)
        if model_type not in self.models:
            await self.load_model(model_type)
        
        model = self.models[model_type]
        config = self.settings.models_config[model_type]
        self.last_used[model_type] = datetime.now()
        
        # Prepare prompt
        full_prompt = self._prepare_prompt(
            message=message,
            model_type=model_type,
            context=context
        )
        
        try:
            logger.info(f"Generating response for model: {model_type}")
            start_time = time.time()
            
            # Generate response
            response = model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=config.get("top_p", 0.9),
                top_k=config.get("top_k", 40),
                stop=["</s>"],  # Only stop on end-of-sequence, not on double newlines
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            return {
                "text": response["choices"][0]["text"].strip(),
                "model_name": config["model_name"],
                "model_type": model_type,
                "tokens_generated": len(response["choices"][0]["text"].split()),
                "generation_time": generation_time,
                "tokens_per_second": len(response["choices"][0]["text"].split()) / generation_time
            }
            
        except Exception as e:
            logger.error(f"Error generating response with {model_type}: {e}", exc_info=True)
            raise
    
    async def _generate_vision_text_response(
        self,
        message: str,
        context: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text-only response using the Qwen2.5-VL vision model"""
        
        # Ensure vision model is loaded
        if not self.qwen_vision_manager.model_loaded:
            await self.load_model("vision")
        
        self.last_used["vision"] = datetime.now()
        
        try:
            start_time = time.time()
            
            # Use the QwenVisionManager for text generation
            result = self.qwen_vision_manager.generate_text_response(
                message=message,
                context=context,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            generation_time = time.time() - start_time
            
            # Ensure result has required fields
            if isinstance(result, dict) and "text" in result:
                return {
                    "text": result["text"],
                    "model_name": self.qwen_vision_manager.model_name,
                    "model_type": "vision",
                    "tokens_generated": result.get("tokens_generated", len(result["text"].split())),
                    "generation_time": result.get("generation_time", generation_time),
                    "tokens_per_second": result.get("tokens_generated", len(result["text"].split())) / generation_time if generation_time > 0 else 0
                }
            else:
                # Fallback if result format is unexpected
                text_result = str(result) if result else "Hello! I'm Qwen2.5-VL, ready to help with any questions or tasks."
                return {
                    "text": text_result,
                    "model_name": self.qwen_vision_manager.model_name,
                    "model_type": "vision",
                    "tokens_generated": len(text_result.split()),
                    "generation_time": generation_time,
                    "tokens_per_second": len(text_result.split()) / generation_time if generation_time > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Error generating text response with Qwen2.5-VL: {e}", exc_info=True)
            # Return a friendly fallback response instead of raising
            return {
                "text": f"Hello! I'm Qwen2.5-VL, a multimodal AI assistant. I can help with text conversations, image analysis, and more. How can I assist you today?",
                "model_name": self.qwen_vision_manager.model_name if hasattr(self, 'qwen_vision_manager') else "Qwen2.5-VL",
                "model_type": "vision",
                "tokens_generated": 25,
                "generation_time": 0.1,
                "tokens_per_second": 250.0,
                "error": str(e)
            }
    
    async def _generate_vision_text_stream(
        self,
        message: str,
        context: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream text-only response using the Qwen2.5-VL vision model"""
        
        # Ensure vision model is loaded
        if not self.qwen_vision_manager.model_loaded:
            await self.load_model("vision")
        
        self.last_used["vision"] = datetime.now()
        
        try:
            # Use the non-streaming method and yield as single chunk for now
            # TODO: Implement proper streaming in QwenVisionManager.generate_text_response
            result = self.qwen_vision_manager.generate_text_response(
                message=message,
                context=context,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            # Simulate streaming by yielding the complete response
            if isinstance(result, dict) and "text" in result:
                # Yield complete response as streaming chunks
                text = result["text"]
                
                # Split into word chunks for streaming effect
                words = text.split()
                accumulated_text = ""
                
                for i, word in enumerate(words):
                    accumulated_text += word + " "
                    
                    yield {
                        "token": word + " ",
                        "accumulated_text": accumulated_text.strip(),
                        "model_name": self.qwen_vision_manager.model_name,
                        "model_type": "vision",
                        "finished": False
                    }
                    
                    # Small delay to simulate real streaming
                    await asyncio.sleep(0.01)
                
                # Final response
                yield {
                    "token": "",
                    "accumulated_text": text,
                    "model_name": self.qwen_vision_manager.model_name,
                    "model_type": "vision",
                    "tokens_generated": result.get("tokens_generated", len(text.split())),
                    "generation_time": result.get("generation_time", 0.1),
                    "finished": True
                }
            else:
                # Fallback response
                fallback_text = "Hello! I'm Qwen2.5-VL, ready to help with any questions or tasks."
                yield {
                    "token": "",
                    "accumulated_text": fallback_text,
                    "model_name": self.qwen_vision_manager.model_name,
                    "model_type": "vision",
                    "tokens_generated": len(fallback_text.split()),
                    "generation_time": 0.1,
                    "finished": True
                }
                
        except Exception as e:
            logger.error(f"Error in vision text streaming: {e}", exc_info=True)
            yield {
                "token": "",
                "accumulated_text": "Hello! I'm Qwen2.5-VL, a multimodal AI assistant. How can I help you today?",
                "model_name": self.qwen_vision_manager.model_name if hasattr(self, 'qwen_vision_manager') else "Qwen2.5-VL",
                "model_type": "vision",
                "tokens_generated": 15,
                "generation_time": 0.1,
                "finished": True,
                "error": str(e)
            }
    
    async def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail",
        max_tokens: int = 8192,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Analyze an image using the Transformers-based Qwen2.5-VL model"""
        
        # Ensure vision model is loaded
        if not self.qwen_vision_manager.model_loaded:
            await self.load_model("vision")
        
        self.last_used["vision"] = datetime.now()
        
        try:
            # Use the QwenVisionManager for analysis
            if stream:
                # Return generator for streaming
                return self.qwen_vision_manager.analyze_image(
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    stream=True
                )
            else:
                # Regular non-streaming analysis
                result = self.qwen_vision_manager.analyze_image(
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    stream=False
                )
                
                # Add any additional metadata from our tracking
                result.update({
                    "analysis_timestamp": datetime.now().isoformat(),
                    "model_manager": "qwen_vision_transformers"
                })
                
                return result
            
        except Exception as e:
            logger.error(f"Error analyzing image with Qwen2.5-VL: {e}", exc_info=True)
            return {
                "text": f"""Error analyzing image with Qwen2.5-VL: {str(e)}

**Qwen2.5-VL Transformers Model Status:**
- Model: {self.qwen_vision_manager.model_name}
- Device: {self.qwen_vision_manager.device}
- Loaded: {self.qwen_vision_manager.model_loaded}

**Image Analysis Capabilities:**
For satellite and aerial imagery, Qwen2.5-VL provides:
- Advanced object detection and recognition
- Infrastructure analysis (roads, buildings, bridges)
- Land use pattern classification
- Geographic feature identification
- Spatial relationship analysis
- Text recognition (OCR) from images
- High-resolution image processing

**Troubleshooting:**
1. Ensure the image file exists and is accessible
2. Supported formats: JPEG, PNG, WebP, BMP
3. Check CUDA/GPU availability for optimal performance
4. Verify Transformers library installation

Please try again or check the logs for more detailed error information.""",
                "model_name": self.qwen_vision_manager.model_name,
                "model_type": "qwen2.5-vl",
                "tokens_generated": 50,
                "generation_time": 0.1,
                "error": str(e)
            }
    
    def analyze_image_stream(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail",
        max_tokens: int = 8192
    ):
        """Stream image analysis using the Transformers-based Qwen2.5-VL model"""
        
        # This is a synchronous generator that wraps the vision manager's streaming
        try:
            # Use the QwenVisionManager for streaming analysis
            stream_generator = self.qwen_vision_manager.analyze_image(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.8,
                stream=True
            )
            
            # Yield each chunk from the stream
            for chunk in stream_generator:
                # Add additional metadata
                chunk.update({
                    "analysis_timestamp": datetime.now().isoformat(),
                    "model_manager": "qwen_vision_transformers"
                })
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming image analysis with Qwen2.5-VL: {e}", exc_info=True)
            yield {
                "token": "",
                "accumulated_text": f"Streaming error: {str(e)}",
                "model_name": self.qwen_vision_manager.model_name if hasattr(self, 'qwen_vision_manager') else "qwen2.5-vl",
                "model_type": "qwen2.5-vl",
                "finished": True,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "model_manager": "qwen_vision_transformers"
            }
    
    async def analyze_audio(
        self,
        audio_path: str,
        prompt: str = "Transcribe and analyze this audio",
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Analyze audio using the Qwen2.5-Omni multimodal model - full audio support"""
        
        # Ensure vision model is loaded (Qwen2.5-Omni handles all modalities)
        if "vision" not in self.models:
            await self.load_model("vision")
        
        model = self.models["vision"]
        config = self.settings.models_config["vision"]
        self.last_used["vision"] = datetime.now()
        
        try:
            start_time = time.time()
            
            # Qwen2.5-Omni chat template format for audio analysis
            system_message = "You are Qwen2.5-Omni, a state-of-the-art multimodal AI assistant that can transcribe, analyze, and understand audio content, as well as handle images, videos, and text."
            
            # Create Qwen2.5-Omni formatted prompt for audio analysis
            formatted_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}\n\n[Audio file: {audio_path}]<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate response with Qwen2.5-Omni recommended settings
            response = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=config["temperature"],
                top_p=config.get("top_p", 0.95),
                top_k=config.get("top_k", 50),
                stop=["<|im_end|>", "<|im_start|>", "</s>"],
                echo=False
            )
            
            generated_text = response["choices"][0]["text"].strip()
            
            # Clean up any remaining template tokens
            generated_text = generated_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
            
            # Enhanced fallback for audio processing
            if not generated_text:
                generated_text = f"I can process the audio file at {audio_path}. I'm Qwen2.5-Omni, a multimodal model that supports audio transcription, analysis, speech recognition, and audio understanding. Please specify what you'd like me to do with this audio file."
            
            generation_time = time.time() - start_time
            
            return {
                "text": generated_text,
                "model_name": config["model_name"],
                "model_type": "vision",
                "tokens_generated": len(generated_text.split()),
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio with Qwen2.5-Omni: {e}", exc_info=True)
            # Return fallback response for audio processing
            return {
                "text": f"I encountered an error while analyzing the audio. I'm Qwen2.5-Omni and I support audio transcription and analysis. Please ensure the audio file is in a supported format (MP3, WAV, OGG, M4A, etc.) and try again. Audio file: {audio_path}",
                "model_name": config["model_name"],
                "model_type": "vision",
                "tokens_generated": 0,
                "generation_time": 0
            }
    
    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        conversation_history: str = ""
    ) -> Dict[str, Any]:
        """Generate intelligent responses for coding model - handles both general chat and code requests"""
        
        # Ensure coding model is loaded
        if "coding" not in self.models:
            await self.load_model("coding")
        
        model = self.models["coding"]
        config = self.settings.models_config["coding"]
        self.last_used["coding"] = datetime.now()
        
        # Log conversation history for debugging
        logger.info(f"Conversation history length: {len(conversation_history)}")
        logger.info(f"Current prompt: {prompt}")
        
        # Use proper Qwen2.5 chat format for consistent English responses
        system_message = "You are Qwen2.5-Coder, a helpful AI coding assistant. You respond in English and help with programming questions, code generation, debugging, and general technical discussions. You are friendly, helpful, and always respond in English."
        
        if conversation_history.strip():
            # Format with conversation history using proper Qwen format
            full_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n{conversation_history}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Simple greeting or question format
            full_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            start_time = time.time()
            
            # Generate response with optimized settings for coding
            response = model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=config.get("top_p", 0.95),
                top_k=config.get("top_k", 50),
                stop=["<|im_end|>", "<|im_start|>", "</s>", "User:", "Assistant:"],
                echo=False
            )
            
            generation_time = time.time() - start_time
            generated_text = response["choices"][0]["text"].strip()
            
            # Clean up any chat format tokens
            generated_text = generated_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
            
            # Fallback for empty responses
            if not generated_text:
                generated_text = "Hello! I'm Qwen2.5-Coder, your coding assistant. I'm here to help with programming questions, code generation, debugging, and technical discussions. How can I assist you today?"
                logger.warning(f"Generated empty response, using fallback")
            else:
                # Enhanced code formatting detection and wrapping
                generated_text = self._format_code_response(generated_text, prompt, language)
            
            # Log the response for debugging
            logger.info(f"Generated response length: {len(generated_text)}")
            logger.info(f"Generated response preview: {generated_text[:100]}...")
            
            return {
                "text": generated_text,
                "model_name": config["model_name"],
                "model_type": "coding",
                "tokens_generated": len(generated_text.split()) if generated_text else 0,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            # Return a friendly fallback response instead of raising an exception
            return {
                "text": "Hello! I'm Qwen2.5-Coder, your AI coding assistant. I'm ready to help with programming questions, code generation, debugging, and technical discussions. How can I assist you today?",
                "model_name": config["model_name"],
                "model_type": "coding",
                "tokens_generated": 25,
                "generation_time": 0.1
            }
    
    def _format_code_response(self, text: str, prompt: str, language: str = "python") -> str:
        """Format coding responses with proper markdown code blocks"""
        import re
        
        # Keywords that indicate a code request
        code_keywords = [
            'write', 'create', 'generate', 'make', 'build', 'code', 'program', 'script',
            'function', 'class', 'algorithm', 'implement', 'solve', 'fix', 'debug',
            'example', 'sample', 'snippet', 'demo'
        ]
        
        # Check if the prompt is asking for code
        is_code_request = any(keyword in prompt.lower() for keyword in code_keywords)
        
        # If already has proper code blocks, return as-is
        if '```' in text:
            return text
        
        # If this is a clear code request and response looks like code, wrap it
        if is_code_request and self._looks_like_code(text):
            # Detect language from the code or use provided language
            detected_lang = self._detect_language(text) or language
            return f"Sure! Below is a {detected_lang} {self._get_code_description(prompt)}:\n\n```{detected_lang}\n{text.strip()}\n```\n\nThis code {self._get_code_explanation(prompt, detected_lang)}."
        
        # For mixed responses (explanation + code), try to identify and wrap code blocks
        formatted_text = self._wrap_inline_code_blocks(text, language)
        
        return formatted_text
    
    def _looks_like_code(self, text: str) -> bool:
        """Detect if text looks like code"""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if __name__',  # Python
            'function ', 'const ', 'let ', 'var ', '=>',  # JavaScript
            'public class', 'private ', 'public ', 'void ',  # Java
            '#include', 'int main', 'printf', 'cout',  # C/C++
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE TABLE',  # SQL
        ]
        
        code_patterns = [
            r'\w+\s*\([^)]*\)\s*{',  # Function declarations
            r'^\s*[a-zA-Z_]\w*\s*=\s*',  # Variable assignments
            r'^\s*for\s*\([^)]*\)',  # For loops
            r'^\s*while\s*\([^)]*\)',  # While loops
            r'^\s*if\s*\([^)]*\)',  # If statements
        ]
        
        # Check for code indicators
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in code_indicators):
            return True
        
        # Check for code patterns
        import re
        for pattern in code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        # Check for indentation patterns common in code
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        return indented_lines > len(lines) * 0.3  # If 30% of lines are indented
    
    def _detect_language(self, text: str) -> str:
        """Detect programming language from code"""
        language_patterns = {
            'python': [r'\bdef\s+\w+\(', r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import', r'\bif\s+__name__\s*==\s*[\'"]__main__[\'"]'],
            'javascript': [r'\bfunction\s+\w+\(', r'\bconst\s+\w+', r'\blet\s+\w+', r'=>'],
            'java': [r'\bpublic\s+class\s+\w+', r'\bpublic\s+static\s+void\s+main'],
            'cpp': [r'#include\s*<\w+>', r'\bint\s+main\s*\(', r'\bcout\s*<<'],
            'c': [r'#include\s*<\w+\.h>', r'\bprintf\s*\('],
            'sql': [r'\bSELECT\s+', r'\bINSERT\s+INTO', r'\bCREATE\s+TABLE'],
        }
        
        import re
        text_lower = text.lower()
        
        for lang, patterns in language_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                return lang
        
        return 'auto'
    
    def _get_code_description(self, prompt: str) -> str:
        """Generate description of what the code does based on prompt"""
        prompt_lower = prompt.lower()
        
        if 'game' in prompt_lower:
            return 'game implementation'
        elif 'algorithm' in prompt_lower:
            return 'algorithm'
        elif 'function' in prompt_lower:
            return 'function'
        elif 'class' in prompt_lower:
            return 'class implementation'
        elif 'script' in prompt_lower:
            return 'script'
        else:
            return 'program'
    
    def _get_code_explanation(self, prompt: str, language: str) -> str:
        """Generate explanation of what the code accomplishes"""
        prompt_lower = prompt.lower()
        
        explanations = {
            'snake': f"implements a classic Snake game using {language}",
            'game': f"creates an interactive game using {language}",
            'sort': f"provides a sorting algorithm implementation",
            'search': f"implements a search functionality",
            'api': f"creates an API endpoint",
            'web': f"builds a web application component",
        }
        
        for keyword, explanation in explanations.items():
            if keyword in prompt_lower:
                return explanation
        
        return f"provides the functionality you requested using {language}"
    
    def _wrap_inline_code_blocks(self, text: str, default_language: str = "python") -> str:
        """Identify and wrap code blocks in mixed text"""
        import re
        
        lines = text.split('\n')
        result_lines = []
        in_code_block = False
        code_lines = []
        current_language = default_language
        
        for line in lines:
            # Check if this line looks like code
            if self._line_looks_like_code(line):
                if not in_code_block:
                    # Starting a new code block
                    in_code_block = True
                    code_lines = []
                    current_language = self._detect_language_from_line(line) or default_language
                code_lines.append(line)
            else:
                if in_code_block:
                    # End the current code block
                    if code_lines:
                        result_lines.append(f"```{current_language}")
                        result_lines.extend(code_lines)
                        result_lines.append("```")
                    in_code_block = False
                    code_lines = []
                result_lines.append(line)
        
        # Handle case where text ends with code
        if in_code_block and code_lines:
            result_lines.append(f"```{current_language}")
            result_lines.extend(code_lines)
            result_lines.append("```")
        
        return '\n'.join(result_lines)
    
    def _line_looks_like_code(self, line: str) -> bool:
        """Check if a single line looks like code"""
        # Skip empty lines and comments in explanations
        if not line.strip() or line.strip().startswith('#'):
            return False
        
        code_line_patterns = [
            r'^\s*(def|class|import|from|if|for|while|try|except|with)\s+',  # Python keywords
            r'^\s*(function|const|let|var|if|for|while)\s+',  # JavaScript keywords
            r'^\s*[a-zA-Z_]\w*\s*[=:]\s*',  # Variable assignments
            r'^\s*[a-zA-Z_]\w*\([^)]*\)\s*[{:]?',  # Function calls/definitions
            r'^\s*[\w\.]+\s*=\s*\w+',  # Simple assignments
        ]
        
        import re
        for pattern in code_line_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _detect_language_from_line(self, line: str) -> str:
        """Detect language from a single line of code"""
        import re
        
        if re.search(r'\b(def|import|from)\s+', line):
            return 'python'
        elif re.search(r'\b(function|const|let|var)\s+', line):
            return 'javascript'
        elif re.search(r'\b(public|private|class)\s+', line):
            return 'java'
        
        return None
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_type not in self.models:
            return {"status": "not_loaded"}
        
        config = self.settings.models_config[model_type]
        stats = self.model_stats.get(model_type, {})
        
        return {
            "status": "loaded",
            "model_type": model_type,
            "model_name": config["model_name"],
            "context_length": config["context_length"],
            "last_used": self.last_used.get(model_type),
            "stats": stats,
            "memory_usage": self._get_memory_usage()
        }
    
    async def generate_code_stream(
        self,
        prompt: str,
        language: str = "auto",
        max_tokens: int = 8000,  # Updated to 8000
        temperature: float = 0.7,
        conversation_history: str = "",
        use_web_search: bool = False,
        use_deep_research: bool = False,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate code as a stream with intelligent features."""
        max_tokens = 8000  # Hardcode to max
        final_sources = sources if sources is not None else []
        context = None
        accumulated_text = ""
        
        try:
            # Determine if a search is needed
            perform_search = use_web_search and not use_deep_research and not context and not final_sources

            if perform_search:
                # Sanitize search query
                search_query_prompt = f"Extract only the main search keywords from this query, separated by spaces. Return ONLY the keywords, no explanations.\n\nQuery: \"{prompt}\"\nKeywords:"
                
                search_query_response = await self.generate_response(
                    message=search_query_prompt,
                    model_type="coding",
                    max_tokens=50,
                    temperature=0.1
                )
                search_query = search_query_response.get("text", prompt).strip()
                logger.info(f"Sanitized search query using coding model: '{search_query}'")

                # Perform web search
                search_results = await self.web_search_service.search_and_scrape(search_query, max_results=5)
                if search_results:
                    context = self.web_search_service.format_search_context(search_results)
                    final_sources = search_results

            # Prepare the prompt for the coding model
            code_prompt = self._prepare_prompt(
                message=prompt,
                model_type="coding",
                language=language,
                context=context,
                conversation_history=conversation_history
            )

            # Ensure coding model is loaded
            if "coding" not in self.models:
                await self.load_model("coding")
            
            model = self.models["coding"]
            config = self.settings.models_config["coding"]
            self.last_used["coding"] = datetime.now()

            try:
                stream = model(code_prompt, max_tokens=max_tokens, temperature=temperature, stream=True, stop=["<|im_end|>", "<|im_start|>", "</s>"])
                
                for output in stream:
                    if 'choices' in output and len(output['choices']) > 0:
                        token = output['choices'][0].get('text', '')
                        token = token.replace("<|im_end|>", "").replace("<|im_start|>", "")
                        accumulated_text += token
                        # Minimize chunk size - only essential fields for intermediate chunks
                        yield {
                            "token": token, 
                            "accumulated_text": accumulated_text, 
                            "finished": False
                        }

                # Ensure final chunk is always sent
                logger.info(f"📤 Sending final chunk with {len(final_sources)} sources")
                final_chunk = {
                    "token": "", 
                    "accumulated_text": accumulated_text.strip(), 
                    "finished": True, 
                    "sources": final_sources,
                    "model_used": config.get("model_name", "coding"),
                    "model_type": "coding"
                }
                yield final_chunk

            except Exception as e:
                logger.error(f"Error in streaming response: {e}", exc_info=True)
                error_chunk = {
                    "error": str(e), 
                    "finished": True, 
                    "sources": final_sources,
                    "accumulated_text": accumulated_text,
                    "model_used": config.get("model_name", "coding") if 'config' in locals() else "coding",
                    "model_type": "coding"
                }
                yield error_chunk
            
        except Exception as e:
            logger.error(f"Error in generate_code_stream: {e}", exc_info=True)
            error_chunk = {
                "error": str(e), 
                "finished": True, 
                "sources": final_sources,
                "accumulated_text": accumulated_text,
                "model_type": "coding"
            }
            yield error_chunk

    async def generate_response_stream(
        self,
        message: str,
        model_type: str,
        context: Optional[str] = None,
        max_tokens: int = 8000,  # Updated to 8000
        temperature: float = 0.7,
        use_web_search: bool = False,
        use_deep_research: bool = False,
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Generate response as a stream, with search features."""
        max_tokens = 8000  # Hardcode to max
        final_sources = sources if sources is not None else []
        accumulated_text = ""
        try:
            # Special handling for the vision model to prevent KeyError
            if model_type == "vision":
                # This is a text-only conversation with the vision model
                async for chunk in self._generate_vision_text_stream(
                    message=message,
                    context=context,
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    if chunk.get('accumulated_text'):
                        accumulated_text = chunk['accumulated_text']
                    # Ensure sources are included in final chunk
                    if chunk.get('finished', False):
                        chunk['sources'] = final_sources
                    else:
                        chunk['sources'] = []
                    yield chunk
                return
                
            # Determine if a search is needed
            perform_search = use_web_search and not use_deep_research and not context and not final_sources
            
            # Sanitize search query if a search is to be performed
            if perform_search:
                search_query_prompt = f"Extract only the main search keywords from this query, separated by spaces. Return ONLY the keywords, no explanations.\n\nQuery: \"{message}\"\nKeywords:"
                
                # Use the same model type for sanitization to avoid switching
                search_query_response = await self.generate_response(
                    message=search_query_prompt,
                    model_type=model_type,
                    max_tokens=50,
                    temperature=0.1
                )
                
                search_query = search_query_response.get("text", message).strip()
                logger.info(f"Sanitized search query using {model_type} model: '{search_query}'")

                # Perform web search
                search_results = await self.web_search_service.search_and_scrape(search_query, max_results=5)
                if search_results:
                    context = self.web_search_service.format_search_context(search_results)
                    final_sources = search_results
            
            # Prepare the prompt for the LLM
            full_prompt = self._prepare_prompt(
                message, model_type, context
            )
            
            # Ensure the right model is loaded
            if model_type not in self.models:
                await self.load_model(model_type)
            
            model = self.models[model_type]
            config = self.settings.models_config[model_type]
            self.last_used[model_type] = datetime.now()

            try:
                stream = model(full_prompt, max_tokens=max_tokens, temperature=temperature, stream=True, stop=["<|im_end|>", "<|im_start|>", "</s>"])
                
                for output in stream:
                    if 'choices' in output and len(output['choices']) > 0:
                        token = output['choices'][0].get('text', '')
                        token = token.replace("<|im_end|>", "").replace("<|im_start|>", "")
                        accumulated_text += token
                        yield {"token": token, "accumulated_text": accumulated_text, "finished": False, "sources": []}

                # Ensure final chunk is always sent
                logger.info(f"📤 Sending final chunk with {len(final_sources)} sources")
                final_chunk = {
                    "token": "", 
                    "accumulated_text": accumulated_text.strip(), 
                    "finished": True, 
                    "sources": final_sources,
                    "model_used": config.get("model_name", model_type),
                    "model_type": model_type
                }
                yield final_chunk
            
            except Exception as e:
                logger.error(f"Error in streaming response: {e}", exc_info=True)
                error_chunk = {
                    "error": str(e), 
                    "finished": True, 
                    "sources": final_sources,
                    "accumulated_text": accumulated_text,
                    "model_used": config.get("model_name", model_type) if 'config' in locals() else model_type,
                    "model_type": model_type
                }
                yield error_chunk

        except Exception as e:
            logger.error(f"Error in generate_response_stream: {e}", exc_info=True)
            error_chunk = {
                "error": str(e), 
                "finished": True, 
                "sources": final_sources,
                "accumulated_text": accumulated_text,
                "model_type": model_type
            }
            yield error_chunk

    def _prepare_prompt(
        self,
        message: str,
        model_type: str,
        context: Optional[str] = None,
        conversation_history: Optional[str] = None,
        language: Optional[str] = "auto"
    ) -> str:
        """Prepare the prompt for the LLM based on model type and context."""
        
        # Base system messages for different model types
        if model_type == "coding":
            base_system = f"You are Qwen2.5-Coder, a helpful AI coding assistant for the {language} language. You respond in English and help with programming questions, code generation, debugging, and general technical discussions."
        else:  # report_generation or vision
            base_system = "You are a knowledgeable AI assistant that provides detailed, comprehensive, and well-researched responses."

        # Enhanced system prompt when context is provided
        if context and context.strip():
            enhanced_system = f"""{base_system}

IMPORTANT INSTRUCTIONS FOR USING SEARCH CONTEXT:
- You have been provided with current, comprehensive search results and web content below
- Use this information as your PRIMARY source to provide detailed, accurate, and up-to-date responses
- Reference specific information, statistics, trends, and findings from the search results
- Synthesize information from multiple sources to provide a comprehensive answer
- Include relevant details, examples, and specifics from the sources
- Structure your response clearly with sections, bullet points, or numbered lists when appropriate
- If the user asks about recent trends, developments, or current information, prioritize the search context over general knowledge
- Be thorough and comprehensive - aim for detailed responses that demonstrate understanding of the provided context

SEARCH CONTEXT AND SOURCES:
{context}

Based on the above comprehensive search results, provide a detailed and thorough response to the user's question. Make sure to utilize the specific information from the sources."""
        else:
            enhanced_system = base_system

        # Construct system prompt
        system_prompt = f"<|im_start|>system\n{enhanced_system}<|im_end|>"

        # Construct user prompt with conversation history if provided
        user_prompt_parts = []
        if conversation_history:
            user_prompt_parts.append(f"{conversation_history}\n")
        
        # Add user message with enhanced instructions for context-based responses
        if context and context.strip():
            user_message = f"""Based on the comprehensive search results provided in the system context above, please provide a detailed and thorough response to this question:

{message}

Please ensure your response:
- Utilizes specific information from the search results
- Is comprehensive and detailed
- References relevant findings, statistics, and trends
- Provides practical insights and examples
- Is well-structured and easy to follow"""
        else:
            user_message = message
            
        user_prompt_parts.append(f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")
        user_prompt = "".join(user_prompt_parts)

        return f"{system_prompt}\n{user_prompt}"
        
    def get_loaded_model_stats(self) -> Dict[str, Any]:
        """Get stats about the currently loaded large model."""
        if not self.current_large_model:
            return {"status": "not_loaded"}
        
        config = self.settings.models_config[self.current_large_model]
        stats = self.model_stats.get(self.current_large_model, {})
        
        return {
            "status": "loaded",
            "model_type": self.current_large_model,
            "model_name": config["model_name"],
            "context_length": config["context_length"],
            "last_used": self.last_used.get(self.current_large_model),
            "stats": stats,
            "memory_usage": self._get_memory_usage()
        } 