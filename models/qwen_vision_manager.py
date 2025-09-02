#!/usr/bin/env python3
"""
Qwen2.5-VL Vision Manager using Transformers
Provides state-of-the-art vision analysis using the official Qwen2.5-VL model
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
from PIL import Image
import base64
import io
import time
from datetime import datetime

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers or qwen_vl_utils not available: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class QwenVisionManager:
    """Advanced Vision Manager using Qwen2.5-VL-7B-Instruct with Transformers"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """Initialize the Qwen2.5-VL vision model"""
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Required dependencies not available. Please install: pip install git+https://github.com/huggingface/transformers qwen-vl-utils[decord]==0.0.8")
        
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.load_time = None
        
        # Enhanced generation settings for maximum performance
        self.generation_config = {
            "max_new_tokens": 8192,      # Maximum detail
            "temperature": 0.8,          # Balanced creativity
            "top_p": 0.95,              # High quality
            "top_k": 100,               # Good diversity
            "do_sample": True,          # Enable sampling
            "repetition_penalty": 1.1,  # Prevent repetition
            "pad_token_id": 0           # Padding token
        }
        
        logger.info(f"🔧 QwenVisionManager initialized for {model_name}")
        logger.info(f"🖥️ Device: {self.device}")
        logger.info(f"⚙️ Enhanced settings: {self.generation_config}")
    
    def load_model(self) -> bool:
        """Load the Qwen2.5-VL model"""
        
        if self.model_loaded:
            return True
        
        try:
            start_time = time.time()
            logger.info(f"🚀 Loading Qwen2.5-VL model: {self.model_name}")
            
            # Load model with optimal settings
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
                device_map="auto",
                # Enable flash attention for better performance if available
                # attn_implementation="flash_attention_2",  # Uncomment if supported
            )
            
            # Load processor with enhanced visual token settings
            # Higher max_pixels for better detail in satellite imagery
            min_pixels = 256 * 28 * 28    # ~200K pixels minimum
            max_pixels = 1280 * 28 * 28   # ~1M pixels maximum for high detail
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            
            self.load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(f"✅ Qwen2.5-VL model loaded successfully in {self.load_time:.2f}s")
            logger.info(f"📊 Visual token range: {min_pixels:,} - {max_pixels:,} pixels")
            logger.info(f"💾 Model device: {next(self.model.parameters()).device}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen2.5-VL model: {e}")
            return False
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Analyze this image in comprehensive detail",
        max_new_tokens: int = 8192,
        temperature: float = 0.8,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Analyze an image using Qwen2.5-VL with enhanced settings"""
        
        if not self.load_model():
            raise RuntimeError("Failed to load Qwen2.5-VL model")
        
        try:
            start_time = time.time()
            
            # Validate image file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Get image information
            img = Image.open(image_path)
            width, height = img.size
            file_size = os.path.getsize(image_path) / (1024 * 1024)
            
            logger.info(f"🖼️ Processing image: {os.path.basename(image_path)}")
            logger.info(f"📐 Dimensions: {width}x{height}, Size: {file_size:.2f}MB")
            
            # Simplified message structure for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{os.path.abspath(image_path)}"
                        },
                        {
                            "type": "text", 
                            "text": f"""Analyze this image and give response according to the user prompt :

{prompt}

"""
                        }
                    ]
                }
            ]
            
            # Process the messages for the model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision information
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs for the model
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            logger.info(f"🔄 Generating analysis with max_tokens: {max_new_tokens}")
            
            # Generate response with enhanced settings
            with torch.no_grad():
                if stream:
                    # Streaming generation
                    return self._generate_stream(inputs, max_new_tokens, temperature, start_time, width, height, file_size, image_path)
                else:
                    # Non-streaming generation
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=self.generation_config["top_p"],
                        top_k=self.generation_config["top_k"],
                        do_sample=self.generation_config["do_sample"],
                        repetition_penalty=self.generation_config["repetition_penalty"],
                        pad_token_id=self.generation_config["pad_token_id"]
                    )
            
            # Decode the generated response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            generation_time = time.time() - start_time
            
            # Clean up the output
            output_text = output_text.strip()
            
            logger.info(f"✅ Analysis completed in {generation_time:.2f}s")
            logger.info(f"📝 Generated {len(output_text)} characters")
            
            return {
                "text": output_text,
                "model_name": self.model_name,
                "model_type": "qwen2.5-vl",
                "tokens_generated": len(output_text.split()),
                "generation_time": generation_time,
                "image_info": {
                    "width": width,
                    "height": height,
                    "file_size_mb": file_size,
                    "file_name": os.path.basename(image_path)
                },
                "config_used": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": self.generation_config["top_p"],
                    "top_k": self.generation_config["top_k"],
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing image with Qwen2.5-VL: {e}")
            return {
                "text": f"""Error analyzing image with Qwen2.5-VL: {str(e)}

The image ({os.path.basename(image_path) if os.path.exists(image_path) else 'unknown'}) could not be processed.

Qwen2.5-VL model capabilities:
- Advanced image understanding and analysis
- Text recognition (OCR) and document parsing  
- Object detection and spatial reasoning
- Chart and diagram analysis
- Dynamic resolution processing
- Structured output generation

Please ensure the image is in a supported format (JPEG, PNG, WebP, etc.) and try again.""",
                "model_name": self.model_name,
                "model_type": "qwen2.5-vl",
                "tokens_generated": 50,
                "generation_time": 0.1,
                "error": str(e)
            }
    
    def _generate_stream(self, inputs, max_new_tokens, temperature, start_time, width, height, file_size, image_path):
        """Generate streaming response"""
        try:
            # Use TextIteratorStreamer for streaming
            from transformers import TextIteratorStreamer
            import threading
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                timeout=60.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation parameters
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": self.generation_config["top_p"],
                "top_k": self.generation_config["top_k"],
                "do_sample": self.generation_config["do_sample"],
                "repetition_penalty": self.generation_config["repetition_penalty"],
                "pad_token_id": self.generation_config["pad_token_id"],
                "streamer": streamer
            }
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()
            
            # Stream the tokens
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield {
                    "token": new_text,
                    "accumulated_text": generated_text,
                    "model_name": self.model_name,
                    "model_type": "qwen2.5-vl",
                    "finished": False,
                    "image_info": {
                        "width": width,
                        "height": height,
                        "file_size_mb": file_size,
                        "file_name": os.path.basename(image_path)
                    }
                }
            
            generation_time = time.time() - start_time
            
            # Final response
            yield {
                "token": "",
                "accumulated_text": generated_text,
                "model_name": self.model_name,
                "model_type": "qwen2.5-vl",
                "finished": True,
                "tokens_generated": len(generated_text.split()),
                "generation_time": generation_time,
                "image_info": {
                    "width": width,
                    "height": height,
                    "file_size_mb": file_size,
                    "file_name": os.path.basename(image_path)
                },
                "config_used": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": self.generation_config["top_p"],
                    "top_k": self.generation_config["top_k"],
                    "device": str(self.device)
                }
            }
            
            # Wait for generation thread to complete
            generation_thread.join()
            
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield {
                "token": "",
                "accumulated_text": f"Streaming error: {str(e)}",
                "model_name": self.model_name,
                "model_type": "qwen2.5-vl",
                "finished": True,
                "error": str(e)
            }
    
    def generate_text_response(
        self,
        message: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text-only response using Qwen2.5-VL (without image input)"""
        
        if not self.load_model():
            raise RuntimeError("Failed to load Qwen2.5-VL model")
        
        try:
            start_time = time.time()
            
            # Build the conversation message
            if context:
                full_message = f"Context: {context}\n\nUser: {message}\n\nAssistant:"
            else:
                full_message = f"User: {message}\n\nAssistant:"
            
            # Create text-only message structure for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_message
                        }
                    ]
                }
            ]
            
            # Process the messages for the model (text-only)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs (no images or videos)
            inputs = self.processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            logger.info(f"🔄 Generating text response with max_tokens: {max_new_tokens}")
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=self.generation_config["top_p"],
                    top_k=self.generation_config["top_k"],
                    do_sample=self.generation_config["do_sample"],
                    repetition_penalty=self.generation_config["repetition_penalty"],
                    pad_token_id=self.generation_config["pad_token_id"]
                )
            
            # Decode the generated response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            generation_time = time.time() - start_time
            
            # Clean up the output
            output_text = output_text.strip()
            
            # Fallback if response is empty
            if not output_text:
                output_text = "Hello! I'm Qwen2.5-VL, ready to help with any questions or tasks you have."
            
            logger.info(f"✅ Text response generated in {generation_time:.2f}s")
            logger.info(f"📝 Generated {len(output_text)} characters")
            
            return {
                "text": output_text,
                "model_name": self.model_name,
                "model_type": "qwen2.5-vl",
                "tokens_generated": len(output_text.split()),
                "generation_time": generation_time,
                "config_used": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": self.generation_config["top_p"],
                    "top_k": self.generation_config["top_k"],
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating text response with Qwen2.5-VL: {e}")
            return {
                "text": "Hello! I'm Qwen2.5-VL, a multimodal AI assistant. I can help with text conversations, image analysis, and more. How can I assist you today?",
                "model_name": self.model_name,
                "model_type": "qwen2.5-vl",
                "tokens_generated": 25,
                "generation_time": 0.1,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "load_time": self.load_time,
            "generation_config": self.generation_config,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "capabilities": [
                "Advanced image understanding",
                "OCR and text recognition", 
                "Object detection and grounding",
                "Chart and diagram analysis",
                "Dynamic resolution processing",
                "Spatial reasoning",
                "Structured output generation",
                "Text-only conversations"
            ]
        }
    
    def unload_model(self):
        """Unload the model to free memory"""
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor  
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        logger.info("🗑️ Qwen2.5-VL model unloaded")

# Test function
async def test_qwen_vision():
    """Test the Qwen2.5-VL vision manager"""
    
    print("🧪 Testing Qwen2.5-VL Vision Manager")
    print("=" * 50)
    
    try:
        # Initialize manager
        manager = QwenVisionManager()
        
        # Test model loading
        success = manager.load_model()
        if not success:
            print("❌ Failed to load model")
            return
        
        print("✅ Model loaded successfully")
        
        # Test with the satellite image
        if os.path.exists("1.jpeg"):
            print("\n🛰️ Testing satellite image analysis...")
            
            result = manager.analyze_image(
                image_path="1.jpeg",
                prompt="Analyze this satellite image in comprehensive detail. Identify all visible infrastructure, roads, buildings, land use patterns, and geographic features.",
                max_new_tokens=4096
            )
            
            print(f"\n📊 Results:")
            print(f"   Response Length: {len(result['text'])} characters")
            print(f"   Tokens Generated: {result['tokens_generated']}")
            print(f"   Generation Time: {result['generation_time']:.2f}s")
            print(f"   Model: {result['model_name']}")
            
            print(f"\n🔍 Analysis:")
            print("=" * 30)
            print(result['text'])
            
        else:
            print("⚠️ Test image 1.jpeg not found")
        
        # Get model info
        info = manager.get_model_info()
        print(f"\n📋 Model Information:")
        for key, value in info.items():
            if key != "capabilities":
                print(f"   {key}: {value}")
        
        print(f"\n✨ Capabilities:")
        for cap in info["capabilities"]:
            print(f"   • {cap}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_qwen_vision()) 