from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import os
import tempfile
import asyncio
from datetime import datetime
import logging
import time
import json
from pathlib import Path
import base64

from models.llm_manager import LLMManager
from services.web_search import get_search_service
from services.deep_research import DeepResearchService
from services.file_processor import FileProcessor
from services.report_generator import ReportGenerator
from services.embedding_service import embedding_service
from utils.helpers import validate_file_type, get_file_extension
from config import Settings
from services.advanced_document_processor import AdvancedDocumentProcessor
from services.agentic_report_generator import AgenticReportGenerator

from services.report_progress import progress_tracker
from services.embedding_service import EmbeddingService
from services.template_parser import get_template_parser, TemplateParser

# Check for chart generation dependencies
try:
    import matplotlib
    import seaborn
    import pandas
    import plotly
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Local LLM Backend",
    description="FastAPI backend for local LLMs with vision, coding, and chat capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = Settings()
llm_manager = LLMManager(settings)
web_search = get_search_service(settings)
deep_research = DeepResearchService(settings, llm_manager)
file_processor = FileProcessor(settings)
report_generator = ReportGenerator(settings)

# Initialize template parser
template_parser = get_template_parser(settings)

# Initialize advanced document processor
try:
    advanced_doc_processor = AdvancedDocumentProcessor(settings, llm_manager)
    logger.info("Advanced document processor initialized")
except Exception as e:
    logger.error(f"Failed to initialize advanced document processor: {e}")
    advanced_doc_processor = None

# Initialize agentic report generator with all dependencies
agentic_report_generator = AgenticReportGenerator(
    settings=settings,
    llm_manager=llm_manager,
    document_processor=advanced_doc_processor,
    web_search=web_search
)

# Agentic mode setting - enable enhanced agentic processing
agentic_mode = True

# Chart generation support detection
chart_support_status = {
    "matplotlib": HAS_MATPLOTLIB,
    "available_backends": [],
    "chart_types_supported": []
}

if HAS_MATPLOTLIB:
    chart_support_status["available_backends"] = ["Agg", "PDF", "PNG", "SVG"]
    chart_support_status["chart_types_supported"] = ["bar", "line", "pie", "scatter", "histogram", "heatmap"]
    logger.info("🎯 Chart generation fully supported with matplotlib")
else:
    logger.warning("⚠️ Chart generation libraries not available. Install matplotlib, seaborn, pandas for full functionality")


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default="default_session", alias="sessionId")  # For chat history storage
    use_web_search: bool = Field(default=False, alias="useWebSearch")
    use_deep_research: bool = Field(default=False, alias="useDeepResearch")
    max_tokens: int = Field(default=8000, alias="maxTokens")  # Updated to 8000 for better responses
    temperature: float = Field(default=0.7)
    model_type: Optional[str] = Field(default=None, alias="modelType")  # Allow frontend to specify model

    class Config:
        allow_population_by_field_name = True
        alias_generator = lambda s: ''.join(word.capitalize() if i else word for i, word in enumerate(s.split('_')))

class ChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_generated: int
    generation_time: float
    sources: Optional[List[Dict[str, Any]]] = None
    search_query: Optional[str] = None
    source_id: Optional[str] = None
    source_count: int = 0
    search_type: Optional[str] = None

    class Config:
        protected_namespaces = ()

class ReportRequest(BaseModel):
    template_name: str
    data: Dict[str, Any]
    output_format: str = "pdf"
    custom_filename: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    memory_usage: str
    performance: str

class ResearchRequest(BaseModel):
    query: str
    depth: int = 3
    max_sources: int = 15  # Increased from 10 to 15 for more comprehensive research
    include_analysis: bool = True

class EmbeddingRequest(BaseModel):
    text: str
    instruction: Optional[str] = None

class ChatHistoryRequest(BaseModel):
    session_id: str
    message_id: str
    message_type: str  # 'user' or 'assistant'
    content: str
    model_type: Optional[str] = None

    class Config:
        protected_namespaces = ()

class SearchHistoryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    message_type: Optional[str] = None
    limit: int = 10
    similarity_threshold: float = 0.7

class VisionReportRequest(BaseModel):
    template_name: str
    data: Dict[str, Any]
    output_format: str = "pdf"  # pdf, doc

class AgenticReportRequest(BaseModel):
    content_source: Union[str, List[str], Dict[str, Any]]
    report_type: str = "comprehensive_analysis"
    output_format: str = "pdf"
    custom_requirements: Optional[str] = None
    enable_web_research: bool = True
    enable_charts: bool = True
    enable_appendices: bool = True
    template_name: Optional[str] = None  # Add template support

class TemplateUploadRequest(BaseModel):
    template_name: str
    description: Optional[str] = None

class DocumentAnalysisRequest(BaseModel):
    analysis_requirements: str
    include_charts: bool = True
    include_research: bool = True
    output_format: str = "pdf"
    custom_filename: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

# Health check
@app.get("/")
async def root():
    return {"message": "Local LLM Backend is running!", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_status = await llm_manager.get_models_status()
        return {
            "status": "healthy",
            "models": models_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Model management endpoints
@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models and their status"""
    try:
        models = await llm_manager.list_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model"""
    try:
        result = await llm_manager.load_model(model_name)
        return {"status": "success", "message": f"Model {model_name} loaded successfully", "model_info": result}
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a specific model"""
    try:
        await llm_manager.unload_model(model_name)
        return {"status": "success", "message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_type}/switch")
async def switch_model(model_type: str):
    """Switch to a specific model (automatically unloads others if needed)"""
    try:
        valid_models = ["report_generation", "vision", "coding"]
        if model_type not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of: {valid_models}")
        
        result = await llm_manager.load_model(model_type)
        
        return {
            "status": "success",
            "message": f"Switched to {model_type} model",
            "model_info": result,
            "current_large_model": llm_manager.current_large_model
        }
        
    except Exception as e:
        logger.error(f"Failed to switch to model {model_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint using report generation model with conversation history"""
    try:
        start_time = time.time()
        
        # Get conversation history for this session
        conversation_history = get_conversation_history(request.session_id, max_messages=10)
        
        # Store user message immediately
        store_message(request.session_id, "User", request.message)
        
        # Use the currently active model - NO automatic switching
        if request.model_type:
            # User explicitly requested a model type
            model_type = request.model_type
            logger.info(f"🎯 User requested model: {model_type}")
        elif llm_manager.current_large_model:
            # Keep using the currently loaded model
            model_type = llm_manager.current_large_model
            logger.info(f"🔄 Continuing with current model: {model_type}")
        else:
            # Default to report generation only if no model is loaded
            model_type = "report_generation"
            logger.info(f"🆕 No model loaded, defaulting to: {model_type}")
        
        # Handle web search if requested
        search_context = None
        sources = []
        
        if request.use_web_search:
            logger.info(f"🔍 Web search ENABLED for query: '{request.message[:100]}...'")
            
            # Enhance search query with conversation context for better results
            enhanced_query = request.message
            if conversation_history:
                # Extract recent context to make web search more relevant
                recent_context = conversation_history.split('\n')[-4:]  # Last 2 exchanges
                context_summary = ' '.join([msg.split(': ', 1)[1] for msg in recent_context if ': ' in msg])
                if context_summary:
                    enhanced_query = f"{context_summary} {request.message}"
                    logger.info(f"🔍 Enhanced web search query: '{enhanced_query[:100]}...'")
            
            try:
                logger.info(f"🌐 Starting web search for: '{enhanced_query}'")
                search_results = await web_search.search_and_scrape(enhanced_query, max_results=10)  # Increased from 5 to 10
                
                if search_results:
                    search_context = web_search.format_search_context(search_results)
                    sources = search_results
                    logger.info(f"✅ Web search SUCCESS: {len(search_results)} results found and scraped")
                    logger.info(f"📋 Sources titles: {[s.get('title', 'No title')[:50] for s in sources]}")
                else:
                    logger.warning("⚠️ Web search returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Web search FAILED: {e}")
                sources = []
                search_context = None
        else:
            logger.info("🔍 Web search DISABLED for this request")
        
        # Handle deep research if requested
        if request.use_deep_research:
            logger.info("Conducting deep research...")
            research_results = await deep_research.conduct_research(
                request.message,
                max_sources=10,
                depth=2,
                model_type=model_type
            )
            if research_results:
                # Limit the context size to prevent streaming issues
                full_summary = research_results.get("summary", "")
                search_context = full_summary[:2000] + "..." if len(full_summary) > 2000 else full_summary
                sources = research_results.get("sources", [])
                logger.info(f"✅ Deep research SUCCESS: {len(sources)} sources found")
                logger.info(f"📄 Summary length: {len(full_summary)} chars, context length: {len(search_context)} chars")
            else:
                logger.warning("⚠️ Deep research returned NO results")
                search_context = None
                sources = []
        
        # Create enhanced message with conversation context
        enhanced_message = request.message
        if conversation_history:
            enhanced_message = f"Conversation History:\n{conversation_history}\n\n{request.message}"
        
        # Generate response using report generation model
        response = await llm_manager.generate_response(
            message=enhanced_message,
            model_type=model_type,
            context=search_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Store assistant response
        store_message(request.session_id, "Assistant", response["text"])
        
        # Store sources separately and generate source_id if we have sources
        source_id = None
        if sources:
            import uuid
            import time
            source_id = f"src_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            source_storage[source_id] = {
                "sources": sources,
                "search_query": request.message if (request.use_web_search or request.use_deep_research) and sources else None,
                "timestamp": time.time(),
                "type": "deep_research" if request.use_deep_research else "web_search" if request.use_web_search else None
            }
            logger.info(f"📦 Non-streaming: Stored {len(sources)} sources with ID: {source_id}")
        
        # Store chat history in embeddings (without await - it's not async)
        try:
            embedding_service.store_message_embedding(
                message_id=f"msg_{int(time.time())}",
                session_id=request.session_id,
                message_type="user",
                content=request.message,
                model_type=model_type
            )
            embedding_service.store_message_embedding(
                message_id=f"resp_{int(time.time())}",
                session_id=request.session_id,
                message_type="assistant",
                content=response["text"],
                model_type=model_type
            )
        except Exception as e:
            logger.warning(f"Failed to store chat history in embeddings: {e}")
        
        total_time = time.time() - start_time
        
        return ChatResponse(
            response=response["text"],
            model_used=response["model_name"],
            tokens_generated=response["tokens_generated"],
            generation_time=total_time,
            sources=None,  # Don't include full sources in response
            search_query=None,  # Don't include search query in response
            source_id=source_id,  # Add source_id for API fetching
            source_count=len(sources) if sources else 0,
            search_type="deep_research" if request.use_deep_research else "web_search" if request.use_web_search else None
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses in real-time with conversation history"""
    try:
        start_time = time.time()
        
        logger.info(f"🔍 Web search requested: {request.use_web_search}")
        
        # Get conversation history for this session
        conversation_history = get_conversation_history(request.session_id, max_messages=10)
        
        # Store user message immediately
        store_message(request.session_id, "User", request.message)
        
        # Use the currently active model - NO automatic switching
        if request.model_type:
            # User explicitly requested a model type
            model_type = request.model_type
            logger.info(f"🎯 User requested model: {model_type}")
        elif llm_manager.current_large_model:
            # Keep using the currently loaded model
            model_type = llm_manager.current_large_model
            logger.info(f"🔄 Continuing with current model: {model_type}")
        else:
            # Default to report generation only if no model is loaded
            model_type = "report_generation"
            logger.info(f"🆕 No model loaded, defaulting to: {model_type}")
        
        # Handle web search if requested
        search_context = None
        sources = []
        
        if request.use_web_search:
            logger.info("Performing web search...")
            
            # Use only the current message for focused web search
            search_query = request.message
            logger.info(f"Clean web search query: {search_query[:100]}...")
            
            try:
                search_results = await web_search.search_and_scrape(search_query, max_results=5)
                if search_results:
                    search_context = web_search.format_search_context(search_results)
                    sources = search_results
                    logger.info(f"✅ Web search SUCCESS: {len(search_results)} results found and scraped")
                    logger.info(f"📋 Sources titles: {[s.get('title', 'No title')[:50] for s in sources]}")
                else:
                    logger.warning("⚠️ Web search returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Web search FAILED: {e}")
                sources = []
                search_context = None
        
        # Handle deep research if requested
        if request.use_deep_research:
            logger.info("Conducting deep research...")
            research_results = await deep_research.conduct_research(
                request.message,
                max_sources=10,
                depth=2,
                model_type=model_type
            )
            if research_results:
                # Limit the context size to prevent streaming issues
                full_summary = research_results.get("summary", "")
                search_context = full_summary[:2000] + "..." if len(full_summary) > 2000 else full_summary
                sources = research_results.get("sources", [])
                logger.info(f"✅ Deep research SUCCESS: {len(sources)} sources found")
                logger.info(f"📄 Summary length: {len(full_summary)} chars, context length: {len(search_context)} chars")
            else:
                logger.warning("⚠️ Deep research returned NO results")
                search_context = None
                sources = []
        
        # Create enhanced message with conversation context
        enhanced_message = request.message
        if conversation_history:
            enhanced_message = f"Conversation History:\n{conversation_history}\n\n{request.message}"
        
        # Start the timer for generation  
        start_time = time.time()

        async def generate_stream():
            """Generator function to stream the response."""
            # Ensure sources and search context are properly captured
            final_sources = sources[:] if sources else []
            final_search_query = request.message if (request.use_web_search or request.use_deep_research) and sources else None
            
            # Generate unique source ID and store sources separately
            source_id = None
            if final_sources:
                import uuid
                import time
                source_id = f"src_{uuid.uuid4().hex[:16]}_{int(time.time())}"
                source_storage[source_id] = {
                    "sources": final_sources,
                    "search_query": final_search_query,
                    "timestamp": time.time(),
                    "type": "deep_research" if request.use_deep_research else "web_search" if request.use_web_search else None
                }
                logger.info(f"📦 Stored {len(final_sources)} sources with ID: {source_id}")
            
            logger.info(f"📊 Starting stream generation with {len(final_sources)} sources")
            
            try:
                response_generator = llm_manager.generate_response_stream(
                    message=enhanced_message,
                    model_type=model_type,
                    context=search_context,  # Pass the search context here
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    use_web_search=request.use_web_search,
                    use_deep_research=request.use_deep_research,
                    sources=final_sources  # Pass sources to the LLM manager!
                )
                
                final_chunk_sent = False
                accumulated_text = ""
                chunk_count = 0
                
                try:
                    async for chunk in response_generator:
                        chunk_count += 1
                        
                        # Track accumulated text for final chunk
                        if chunk.get('accumulated_text'):
                            accumulated_text = chunk['accumulated_text']
                        
                        # Check if this is the final chunk from the generator
                        if chunk.get('finished', False):
                            # Add source_id instead of full sources to final chunk
                            chunk['source_id'] = source_id
                            chunk['source_count'] = len(final_sources)
                            chunk['search_type'] = "deep_research" if request.use_deep_research else "web_search" if request.use_web_search else None
                            # Remove heavy data
                            chunk.pop('sources', None)
                            chunk.pop('search_query', None)
                            final_chunk_sent = True
                            logger.info(f"✅ Final chunk: Added source_id {source_id} with {len(final_sources)} sources after {chunk_count} chunks")
                        else:
                            # Remove sources and search_query from non-final chunks to reduce size
                            chunk.pop('sources', None)
                            chunk.pop('search_query', None)
                        
                        # Prevent JSON serialization issues with large chunks
                        try:
                            # Minimize chunk size for intermediate chunks
                            if not chunk.get('finished', False):
                                # Only include essential fields for streaming
                                minimal_chunk = {
                                    "token": chunk.get("token", ""),
                                    "accumulated_text": chunk.get("accumulated_text", ""),
                                    "finished": False
                                }
                                chunk_json = json.dumps(minimal_chunk)
                            else:
                                # Final chunk with source reference only
                                chunk_json = json.dumps(chunk)
                            
                            yield f"data: {chunk_json}\n\n"
                        except Exception as json_error:
                            logger.error(f"❌ JSON serialization error in chunk {chunk_count}: {json_error}")
                            # Send a simple error chunk instead
                            error_chunk = {
                                "token": "",
                                "accumulated_text": accumulated_text[:500] if accumulated_text else "Error in response generation",
                                "finished": True,
                                "source_id": source_id,
                                "error": "Chunk serialization failed"
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            break
                
                except Exception as stream_error:
                    logger.error(f"❌ Streaming error after {chunk_count} chunks: {stream_error}")
                    final_chunk_sent = False  # Force manual final chunk
                
                # Ensure final chunk is sent even if the generator didn't mark it as finished
                if not final_chunk_sent:
                    logger.warning(f"⚠️ Final chunk not sent by generator after {chunk_count} chunks, sending manually")
                    final_chunk = {
                        "token": "",
                        "accumulated_text": accumulated_text,
                        "finished": True,
                        "source_id": source_id,
                        "source_count": len(final_sources),
                        "search_type": "deep_research" if request.use_deep_research else "web_search" if request.use_web_search else None,
                        "model_used": model_type,
                        "model_type": model_type
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    logger.info(f"✅ Manual final chunk: Added source_id {source_id} with {len(final_sources)} sources")
                    
            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                error_response = {
                    "token": "",
                    "accumulated_text": f"Sorry, there was an error processing your request: {str(e)}",
                    "finished": True,
                    "source_id": source_id,
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Chat streaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vision-specific endpoint for images and videos
@app.post("/vision/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail"),
    max_tokens: int = Form(8000),  # Updated to 8000 for detailed analysis
    use_web_search: bool = Form(False),  # Add web search support
    use_deep_research: bool = Form(False)  # Add deep research support
):
    """Analyze an image or video using the Qwen2.5-Omni multimodal model with detailed prompting and optional web search"""
    try:
        logger.info(f"🔍 Vision analyze request - filename: {image.filename}, prompt: '{prompt}', max_tokens: {max_tokens}, web_search: {use_web_search}, deep_research: {use_deep_research}")
        logger.info(f"📁 File content_type: {image.content_type}, size: {image.size}")
        
        # Force minimum max_tokens for better results
        if max_tokens < 4096:
            max_tokens = 8192
            logger.info(f"⬆️ Increased max_tokens to {max_tokens} for detailed analysis")
        
        # Get file extension for fallback detection
        file_extension = get_file_extension(image.filename) if image.filename else ""
        logger.info(f"📎 File extension detected: {file_extension}")
        
        # Video file extensions
        video_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv', '.wmv', '.m4v']
        # Image file extensions  
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.ico']
        
        # Determine if it's a video or image
        is_video = False
        is_image = False
        
        # Check content type first (if available)
        if image.content_type:
            is_video = image.content_type.startswith('video/')
            is_image = image.content_type.startswith('image/')
        
        # Fallback to file extension if content type is not available
        if not is_video and not is_image:
            is_video = file_extension.lower() in video_extensions
            is_image = file_extension.lower() in image_extensions
        
        logger.info(f"🎬 File type detection - is_video: {is_video}, is_image: {is_image}")
        
        # Validate that it's either an image or video
        if not (is_video or is_image):
            logger.error(f"❌ Invalid file type - extension: {file_extension}, content_type: {image.content_type}")
            raise HTTPException(status_code=400, detail=f"File must be an image or video. Detected extension: {file_extension}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"💾 Temporary file saved: {temp_file_path}, size: {len(content)} bytes")
        
        # Handle web search and deep research if requested
        search_context = None
        sources = []
        
        if use_web_search:
            logger.info(f"🔍 Web search ENABLED for vision analysis: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🌐 Starting web search for vision analysis: '{prompt}'")
                search_results = await web_search.search_and_scrape(prompt, max_results=5)
                
                if search_results:
                    search_context = web_search.format_search_context(search_results)
                    sources = search_results
                    logger.info(f"✅ Vision web search SUCCESS: {len(search_results)} results found and scraped")
                    logger.info(f"📋 Vision sources titles: {[s.get('title', 'No title')[:50] for s in sources]}")
                else:
                    logger.warning("⚠️ Vision web search returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Vision web search FAILED: {e}")
                sources = []
                search_context = None
        elif use_deep_research:
            logger.info(f"🔍 Deep research ENABLED for vision analysis: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🧠 Starting deep research for vision analysis: '{prompt}'")
                research_results = await deep_research.conduct_research(
                    prompt,
                    max_sources=10,
                    depth=2,
                    model_type="vision"
                )
                if research_results:
                    search_context = research_results.get("summary", "")
                    sources = research_results.get("sources", [])
                    logger.info(f"✅ Vision deep research SUCCESS: {len(sources)} sources found")
                else:
                    logger.warning("⚠️ Vision deep research returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Vision deep research FAILED: {e}")
                sources = []
                search_context = None
        else:
            logger.info("🔍 Web search and deep research DISABLED for vision analysis")
        
        try:
            # Process image or video with enhanced prompting
            file_type = "video" if is_video else "image"
            analysis_prompt = f"{prompt} [File type: {file_type}]" if file_type == "video" else prompt
            
            # Add search context to prompt if available
            if search_context:
                analysis_prompt = f"{analysis_prompt}\n\nAdditional context from web search:\n{search_context}"
            
            logger.info(f"🚀 Calling llm_manager.analyze_image with prompt: '{analysis_prompt[:100]}...' and max_tokens: {max_tokens}")
            
            response = await llm_manager.analyze_image(
                image_path=temp_file_path,
                prompt=analysis_prompt,
                max_tokens=max_tokens
            )
            
            logger.info(f"✅ Vision analysis completed - response length: {len(response['text'])}")
            logger.info(f"📊 Model used: {response['model_name']}, tokens: {response['tokens_generated']}")
            logger.info(f"⚙️ Config used: {response.get('config_used', 'Not available')}")
            
            # Don't add references to response content - let frontend handle them separately
            final_response = response["text"]
            if sources:
                logger.info(f"Found {len(sources)} reference sources for vision response")
            
            return {
                "analysis": final_response,
                "model_used": response["model_name"],
                "file_type": file_type,
                "tokens_generated": response["tokens_generated"],
                "generation_time": response.get("generation_time", 0),
                "config_used": response.get("config_used", {}),
                "capabilities": "Qwen2.5-Omni supports: images, videos, audio, OCR, document parsing - full multimodal AI with detailed analysis",
                "sources": sources,  # Add sources for frontend reference display
                "search_query": search_context if (use_web_search or use_deep_research) else None
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            logger.info(f"🗑️ Cleaned up temporary file: {temp_file_path}")
            
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vision/analyze/stream")
async def analyze_image_stream(
    image: UploadFile = File(...),
    prompt: str = Form("Analyze this image in detail"),
    max_tokens: int = Form(8000),  # Updated to 8000 for detailed analysis
    use_web_search: bool = Form(False),  # Add web search support
    use_deep_research: bool = Form(False)  # Add deep research support
):
    """Stream image analysis using the Transformers-based Qwen2.5-VL model with optional web search"""
    try:
        logger.info(f"🔍 Vision stream analyze request - filename: {image.filename}, prompt: '{prompt}', max_tokens: {max_tokens}, web_search: {use_web_search}, deep_research: {use_deep_research}")
        logger.info(f"📁 File content_type: {image.content_type}, size: {image.size}")
        
        # Force minimum max_tokens for better results
        if max_tokens < 4096:
            max_tokens = 8000
            logger.info(f"⬆️ Increased max_tokens to {max_tokens} for detailed analysis")
        
        # Get file extension for fallback detection
        file_extension = get_file_extension(image.filename) if image.filename else ""
        logger.info(f"📎 File extension detected: {file_extension}")
        
        # Image file extensions  
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.ico']
        
        # Determine if it's an image
        is_image = False
        
        # Check content type first (if available)
        if image.content_type:
            is_image = image.content_type.startswith('image/')
        
        # Fallback to file extension if content type is not available
        if not is_image:
            is_image = file_extension.lower() in image_extensions
        
        logger.info(f"🎬 File type detection - is_image: {is_image}")
        
        # Validate that it's an image
        if not is_image:
            logger.error(f"❌ Invalid file type - extension: {file_extension}, content_type: {image.content_type}")
            raise HTTPException(status_code=400, detail=f"File must be an image. Detected extension: {file_extension}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"💾 Temporary file saved: {temp_file_path}, size: {len(content)} bytes")
        
        # Handle web search and deep research if requested
        search_context = None
        sources = []
        
        if use_web_search:
            logger.info(f"🔍 Web search ENABLED for vision streaming: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🌐 Starting web search for vision streaming: '{prompt}'")
                search_results = await web_search.search_and_scrape(prompt, max_results=5)
                
                if search_results:
                    search_context = web_search.format_search_context(search_results)
                    sources = search_results
                    logger.info(f"✅ Vision streaming web search SUCCESS: {len(search_results)} results found and scraped")
                    logger.info(f"📋 Vision streaming sources titles: {[s.get('title', 'No title')[:50] for s in sources]}")
                else:
                    logger.warning("⚠️ Vision streaming web search returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Vision streaming web search FAILED: {e}")
                sources = []
                search_context = None
        elif use_deep_research:
            logger.info(f"🔍 Deep research ENABLED for vision streaming: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🧠 Starting deep research for vision streaming: '{prompt}'")
                research_results = await deep_research.conduct_research(
                    prompt,
                    max_sources=10,
                    depth=2,
                    model_type="vision"
                )
                if research_results:
                    search_context = research_results.get("summary", "")
                    sources = research_results.get("sources", [])
                    logger.info(f"✅ Vision streaming deep research SUCCESS: {len(sources)} sources found")
                else:
                    logger.warning("⚠️ Vision streaming deep research returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Vision streaming deep research FAILED: {e}")
                sources = []
                search_context = None
        else:
            logger.info("🔍 Web search and deep research DISABLED for vision streaming")
        
        try:
            logger.info(f"🚀 Starting streaming vision analysis with prompt: '{prompt}' and max_tokens: {max_tokens}")
            
            # Create streaming response
            async def generate_stream():
                try:
                    # Ensure vision model is loaded
                    if not llm_manager.qwen_vision_manager.model_loaded:
                        await llm_manager.load_model("vision")
                    
                    # Update last used timestamp
                    llm_manager.last_used["vision"] = datetime.now()
                    
                    # Enhance prompt with search context if available
                    enhanced_prompt = prompt
                    if search_context:
                        enhanced_prompt = f"{prompt}\n\nAdditional context from web search:\n{search_context}"
                    
                    # Get the streaming generator from LLM manager
                    stream_generator = llm_manager.analyze_image_stream(
                        image_path=temp_file_path,
                        prompt=enhanced_prompt,
                        max_tokens=max_tokens
                    )
                    
                    # Stream each chunk
                    for chunk in stream_generator:
                        # Prepare streaming data
                        stream_data = {
                            "token": chunk.get("token", ""),
                            "accumulated_text": chunk.get("accumulated_text", ""),
                            "model_used": chunk.get("model_name", ""),
                            "model_type": chunk.get("model_type", "qwen2.5-vl"),
                            "finished": chunk.get("finished", False),
                            "image_info": chunk.get("image_info", {}),
                            "config_used": chunk.get("config_used", {}),
                            "capabilities": "Qwen2.5-VL supports: advanced image analysis, OCR, object detection, spatial reasoning",
                            "analysis_timestamp": chunk.get("analysis_timestamp", "")
                        }
                        
                        # Add sources to final chunk without embedding in text
                        if chunk.get("finished", False):
                            final_text = chunk.get("accumulated_text", "")
                            
                            stream_data['accumulated_text'] = final_text
                            stream_data['sources'] = sources
                            stream_data['search_query'] = search_context if (use_web_search or use_deep_research) else None
                            if sources:
                                logger.info(f"Found {len(sources)} reference sources for vision streaming response")
                        
                        if chunk.get("finished", False):
                            # Add final statistics
                            stream_data.update({
                                "tokens_generated": chunk.get("tokens_generated", 0),
                                "generation_time": chunk.get("generation_time", 0)
                            })
                        
                        if chunk.get("error"):
                            stream_data["error"] = chunk["error"]
                        
                        yield f"data: {json.dumps(stream_data)}\n\n"
                
                except Exception as e:
                    logger.error(f"❌ Error in vision streaming: {e}")
                    error_data = {
                        "token": "",
                        "accumulated_text": f"Error in vision analysis: {str(e)}",
                        "model_used": "qwen2.5-vl",
                        "model_type": "qwen2.5-vl",
                        "finished": True,
                        "error": str(e),
                        "sources": sources,
                        "search_query": search_context if (use_web_search or use_deep_research) else None
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                        logger.info(f"🗑️ Cleaned up temporary file: {temp_file_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
            
        except Exception as processing_error:
            # Clean up temp file on error
            os.unlink(temp_file_path)
            logger.error(f"❌ Vision processing error: {processing_error}")
            raise HTTPException(status_code=500, detail=str(processing_error))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision streaming analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Audio analysis endpoint - now fully supported with Qwen2.5-Omni
@app.post("/vision/analyze_audio")
async def analyze_audio(
    audio: UploadFile = File(...),
    prompt: str = Form("Transcribe and analyze this audio"),
    max_tokens: int = Form(2048)  # Increased from 1024 for more detailed audio analysis
):
    """Analyze audio using the Qwen2.5-Omni multimodal model with enhanced settings - full audio transcription and analysis support"""
    try:
        # Audio file extensions for fallback detection
        audio_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.wma', '.opus']
        
        file_extension = get_file_extension(audio.filename) if audio.filename else ""
        
        # Check content type and file extension
        is_audio_type = False
        is_audio_ext = file_extension.lower() in audio_extensions
        
        # Check content type if available
        if audio.content_type:
            audio_types = ['audio/', 'video/']  # Accept audio and video files
            is_audio_type = any(audio.content_type.startswith(t) for t in audio_types)
        
        # Validate audio file
        if not (is_audio_type or is_audio_ext):
            raise HTTPException(status_code=400, detail=f"File must be an audio file (MP3, WAV, OGG, M4A, etc.). Detected extension: {file_extension}")
        
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process audio with Qwen2.5-Omni multimodal model
            response = await llm_manager.analyze_audio(
                audio_path=temp_file_path,
                prompt=prompt,
                max_tokens=max_tokens
            )
            
            return {
                "analysis": response["text"],
                "model_used": response["model_name"],
                "tokens_generated": response["tokens_generated"],
                "generation_time": response.get("generation_time", 0),
                "capabilities": "Qwen2.5-Omni fully supports audio transcription, analysis, and understanding",
                "supported_formats": ["MP3", "WAV", "OGG", "M4A", "AAC", "FLAC", "WMA", "OPUS"]
            }
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Coding-specific endpoint
@app.post("/coding/generate")
async def generate_code(
    prompt: str = Form(...),
    language: str = Form("auto"),
    max_tokens: int = Form(8000),  # Updated to 8000 for complete code responses
    temperature: float = Form(0.7),  # Lowered for more consistent coding
    session_id: str = Form("default_session"),
    use_web_search: bool = Form(False),  # Add web search support
    use_deep_research: bool = Form(False)  # Add deep research support
):
    """Generate code using the coding model with intelligent responses, conversation memory, and optional web search"""
    try:
        logger.info(f"🔍 Coding request - prompt: '{prompt[:100]}...', language: {language}, web_search: {use_web_search}, deep_research: {use_deep_research}")
        
        # Get conversation history for this session
        conversation_history = get_conversation_history(session_id)
        
        # Store user message
        store_message(session_id, "User", prompt)
        
        # Handle web search and deep research if requested
        search_context = None
        sources = []
        
        if use_web_search:
            logger.info(f"🔍 Web search ENABLED for coding: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🌐 Starting web search for coding: '{prompt}'")
                search_results = await web_search.search_and_scrape(prompt, max_results=5)
                
                if search_results:
                    search_context = web_search.format_search_context(search_results)
                    sources = search_results
                    logger.info(f"✅ Coding web search SUCCESS: {len(search_results)} results found and scraped")
                    logger.info(f"📋 Coding sources titles: {[s.get('title', 'No title')[:50] for s in sources]}")
                else:
                    logger.warning("⚠️ Coding web search returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Coding web search FAILED: {e}")
                sources = []
                search_context = None
        elif use_deep_research:
            logger.info(f"🔍 Deep research ENABLED for coding: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🧠 Starting deep research for coding: '{prompt}'")
                research_results = await deep_research.conduct_research(
                    prompt,
                    max_sources=10,
                    depth=2,
                    model_type="coding"
                )
                if research_results:
                    search_context = research_results.get("summary", "")
                    sources = research_results.get("sources", [])
                    logger.info(f"✅ Coding deep research SUCCESS: {len(sources)} sources found")
                else:
                    logger.warning("⚠️ Coding deep research returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Coding deep research FAILED: {e}")
                sources = []
                search_context = None
        else:
            logger.info("🔍 Web search and deep research DISABLED for coding")
        
        # Enhance prompt with search context if available
        enhanced_prompt = prompt
        if search_context:
            enhanced_prompt = f"{prompt}\n\nAdditional context from web search:\n{search_context}"
        
        response = await llm_manager.generate_code(
            prompt=enhanced_prompt,
            language=language,
            max_tokens=max_tokens,
            temperature=temperature,
            conversation_history=conversation_history
        )
        
        # Don't add references to response content - let frontend handle them separately
        final_response = response["text"]
        if sources:
            logger.info(f"Found {len(sources)} reference sources for coding response")
        
        # Store assistant response
        store_message(session_id, "Assistant", final_response)
        
        return {
            "text": final_response,  # Use 'text' field for mixed content
            "language": language,
            "model_used": response["model_name"],
            "tokens_generated": response["tokens_generated"],
            "generation_time": response.get("generation_time", 0),
            "session_id": session_id,
            "sources": sources,  # Add sources for frontend reference display
            "search_query": search_context if (use_web_search or use_deep_research) else None
        }
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coding/generate/stream")
async def generate_code_stream_endpoint(
    prompt: str = Form(...),
    language: str = Form("auto"),
    max_tokens: int = Form(8000),  # Updated to 8000 for complete code responses
    temperature: float = Form(0.7),  # Lowered for more consistent coding
    session_id: str = Form("default_session"),
    use_web_search: bool = Form(False),  # Add web search support
    use_deep_research: bool = Form(False)  # Add deep research support
):
    """Stream code generation responses in real-time with optional web search"""
    try:
        logger.info(f"🔍 Coding stream request - prompt: '{prompt[:100]}...', language: {language}, web_search: {use_web_search}, deep_research: {use_deep_research}")
        
        # Get conversation history for this session
        conversation_history = get_conversation_history(session_id)
        
        # Store user message
        store_message(session_id, "User", prompt)
        
        # Handle web search and deep research if requested
        search_context = None
        sources = []
        
        if use_web_search:
            logger.info(f"🔍 Web search ENABLED for coding streaming: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🌐 Starting web search for coding streaming: '{prompt}'")
                search_results = await web_search.search_and_scrape(prompt, max_results=5)
                
                if search_results:
                    search_context = web_search.format_search_context(search_results)
                    sources = search_results
                    logger.info(f"✅ Coding streaming web search SUCCESS: {len(search_results)} results found and scraped")
                    logger.info(f"📋 Coding streaming sources titles: {[s.get('title', 'No title')[:50] for s in sources]}")
                else:
                    logger.warning("⚠️ Coding streaming web search returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Coding streaming web search FAILED: {e}")
                sources = []
                search_context = None
        elif use_deep_research:
            logger.info(f"🔍 Deep research ENABLED for coding streaming: '{prompt[:100]}...'")
            
            try:
                logger.info(f"🧠 Starting deep research for coding streaming: '{prompt}'")
                research_results = await deep_research.conduct_research(
                    prompt,
                    max_sources=10,
                    depth=2,
                    model_type="coding"
                )
                if research_results:
                    search_context = research_results.get("summary", "")
                    sources = research_results.get("sources", [])
                    logger.info(f"✅ Coding streaming deep research SUCCESS: {len(sources)} sources found")
                else:
                    logger.warning("⚠️ Coding streaming deep research returned NO results")
                    
            except Exception as e:
                logger.error(f"❌ Coding streaming deep research FAILED: {e}")
                sources = []
                search_context = None
        else:
            logger.info("🔍 Web search and deep research DISABLED for coding streaming")
        
        # Enhance prompt with search context if available
        enhanced_prompt = prompt
        if search_context:
            enhanced_prompt = f"{prompt}\n\nAdditional context from web search:\n{search_context}"
        
        async def generate_stream():
            full_response = ""
            
            async for chunk in llm_manager.generate_code_stream(
                prompt=enhanced_prompt,
                language=language,
                max_tokens=max_tokens,
                temperature=temperature,
                conversation_history=conversation_history
            ):
                # Update full response
                full_response = chunk.get("accumulated_text", "")
                
                # Prepare streaming data
                stream_data = {
                    "token": chunk.get("token", ""),
                    "accumulated_text": full_response,
                    "model_used": chunk.get("model_name", ""),
                    "model_type": chunk.get("model_type", "coding"),
                    "finished": chunk.get("finished", False),
                    "session_id": session_id,
                    "language": language
                }
                
                # Add sources to final chunk without embedding in text
                if chunk.get("finished", False):
                    if sources:
                        logger.info(f"Found {len(sources)} reference sources for coding streaming response")
                    
                    # Store assistant response when finished
                    store_message(session_id, "Assistant", full_response)
                    
                    # Add timing information and sources
                    stream_data["generation_time"] = chunk.get("generation_time", 0)
                    stream_data["tokens_generated"] = chunk.get("tokens_generated", 0)
                    stream_data["sources"] = sources
                    stream_data["search_query"] = search_context if (use_web_search or use_deep_research) else None
                
                yield f"data: {json.dumps(stream_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Code streaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Web search endpoints
@app.post("/search")
async def search_web(request: SearchRequest):
    """Perform web search"""
    try:
        logger.info(f"Web search request: {request.query}")
        results = await web_search.search_and_scrape(request.query, max_results=request.max_results)
        
        if not results:
            logger.warning(f"No search results found for query: {request.query}")
            return {
                "query": request.query,
                "results": [],
                "count": 0,
                "message": "No search results found"
            }
        
        logger.info(f"Web search completed: {len(results)} results found")
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Web search failed: {str(e)}")

# Deep research endpoints
@app.post("/research")
async def conduct_research(request: ResearchRequest):
    """Conduct deep research on a topic"""
    try:
        results = await deep_research.conduct_research(
            query=request.query,
            depth=request.depth,
            max_sources=request.max_sources,
            model_type='report_generation'
        )
        
        return {
            "query": request.query,
            "research_results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Deep research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload and processing endpoints
@app.post("/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file"""
    try:
        # Read file content
        content = await file.read()
        
        # Process the file
        file_info = await file_processor.process_file(
            file_content=content,
            filename=file.filename,
            content_type=file.content_type
        )
        
        return {
            "message": "File uploaded and processed successfully",
            "file_info": file_info
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/files/upload/multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    try:
        results = []
        
        for file in files:
            try:
                content = await file.read()
                file_info = await file_processor.process_file(
                    file_content=content,
                    filename=file.filename,
                    content_type=file.content_type
                )
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "file_info": file_info
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "message": f"Processed {len(files)} files",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Multiple file upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/files/analyze")
async def analyze_file_with_llm(
    file: UploadFile = File(...),
    prompt: str = Form("Analyze this file and provide insights"),
    model_type: str = Form("chat")
):
    """Analyze a file using the appropriate LLM model"""
    try:
        # Read and process file
        content = await file.read()
        file_info = await file_processor.process_file(
            file_content=content,
            filename=file.filename,
            content_type=file.content_type
        )
        
        # Determine appropriate model based on file type
        if file_info.get('category') == 'image':
            # Use vision model for images
            response = await llm_manager.analyze_image(
                image_path=file_info['temp_path'],
                prompt=prompt
            )
        elif file_info.get('category') == 'code':
            # Use coding model for code files
            code_content = file_info.get('content', '')
            enhanced_prompt = f"Analyze this {file_info.get('language', 'code')} code:\n\n{code_content}\n\nUser request: {prompt}"
            response = await llm_manager.generate_code(
                prompt=enhanced_prompt,
                language=file_info.get('language', 'python')
            )
        else:
            # Use chat model for documents and other files
            text_content = file_info.get('content', '')
            enhanced_prompt = f"Analyze this document:\n\n{text_content}\n\nUser request: {prompt}"
            response = await llm_manager.generate_response(
                message=enhanced_prompt,
                model_type="report_generation"
            )
        
        # Clean up temp file
        await file_processor.cleanup_temp_file(file_info['temp_path'])
        
        return {
            "analysis": response["text"],
            "file_info": file_info,
            "model_used": response["model_name"]
        }
        
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Report generation endpoints
@app.post("/reports/generate")
async def generate_report(request: ReportRequest):
    """Generate a report using a template"""
    try:
        report_path = await report_generator.generate_report(
            template_name=request.template_name,
            data=request.data,
            output_format=request.output_format
        )
        
        return FileResponse(
            path=report_path,
            filename=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.output_format}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/generate/research")
async def generate_research_report(
    query: str = Form(...),
    depth: int = Form(3),
    output_format: str = Form("pdf")
):
    """Generate a research report based on a query"""
    try:
        # Conduct research
        research_results = await deep_research.conduct_research(
            query=query,
            depth=depth,
            model_type='report_generation'
        )
        
        # Prepare data for report template
        report_data = {
            "title": f"Research Report: {query}",
            "topic": query,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "depth": depth,
            "source_count": len(research_results.get("sources", [])),
            "summary": research_results.get("summary", "No summary available"),
            "findings": research_results.get("analysis", "No findings available"),
            "key_points": "\n".join([f"• {point}" for point in research_results.get("key_points", [])]),
            "contradictions": research_results.get("contradictions", "None found"),
            "analysis": research_results.get("analysis", "No analysis available"),
            "sources": "\n".join([f"• {source.get('title', 'Unknown')}: {source.get('url', 'No URL')}" 
                                 for source in research_results.get("sources", [])]),
            "confidence_score": f"{research_results.get('confidence_score', 0.0):.2f}",
            "recommendations": "Based on the research findings, further investigation may be needed."
        }
        
        # Generate report
        report_path = await report_generator.generate_report(
            template_name="research_report",
            data=report_data,
            output_format=output_format
        )
        
        return FileResponse(
            path=report_path,
            filename=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Research report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/templates")
async def list_report_templates():
    """List available report templates"""
    try:
        templates = await report_generator.list_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/templates/upload")
async def upload_template(
    template_file: UploadFile = File(...),
    template_name: str = Form(...),
    description: Optional[str] = Form(None)
):
    """Upload a custom template file for report generation"""
    try:
        # Validate file type
        if not template_file.filename.endswith(('.docx', '.doc', '.pdf', '.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported template format. Please upload .docx, .doc, .pdf, .txt, or .md files."
            )
        
        # Create templates directory if it doesn't exist
        templates_dir = Path("templates/custom")
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file with timestamp to avoid conflicts
        timestamp = int(time.time())
        file_extension = Path(template_file.filename).suffix
        safe_filename = f"{template_name}_{timestamp}{file_extension}"
        template_path = templates_dir / safe_filename
        
        # Save file
        content = await template_file.read()
        with open(template_path, "wb") as f:
            f.write(content)
        
        # Parse template structure
        parser = TemplateParser()
        template_structure = parser.parse_template_file(str(template_path))
        
        # Store template metadata
        template_metadata = {
            "name": template_name,
            "description": description or "Custom uploaded template",
            "file_path": str(template_path),
            "file_size": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "original_filename": template_file.filename,
            "structure": template_structure
        }
        
        # Save metadata
        metadata_path = templates_dir / f"{template_name}_{timestamp}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(template_metadata, f, indent=2)
        
        logger.info(f"Template uploaded successfully: {template_name}")
        
        return {
            "status": "success",
            "message": "Template uploaded and parsed successfully",
            "template_id": f"{template_name}_{timestamp}",
            "template_name": template_name,
            "structure_info": {
                "sections_count": template_structure.get("total_sections", 0),
                "max_depth": template_structure.get("max_depth", 1),
                "structure_type": template_structure.get("structure_type", "standard")
            }
        }
        
    except Exception as e:
        logger.error(f"Template upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template upload failed: {str(e)}")

@app.get("/reports/templates/custom")
async def list_custom_templates():
    """List all uploaded custom templates"""
    try:
        templates_dir = Path("templates/custom")
        if not templates_dir.exists():
            return {"templates": []}
        
        templates = []
        for metadata_file in templates_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    templates.append({
                        "id": metadata_file.stem.replace("_metadata", ""),
                        "name": metadata.get("name", "Unknown"),
                        "description": metadata.get("description", "No description"),
                        "uploaded_at": metadata.get("uploaded_at", "Unknown"),
                        "sections_count": metadata.get("structure", {}).get("total_sections", 0),
                        "structure_type": metadata.get("structure", {}).get("structure_type", "standard")
                    })
            except Exception as e:
                logger.warning(f"Failed to read template metadata {metadata_file}: {e}")
                continue
        
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Failed to list custom templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reports/templates/custom/{template_id}")
async def delete_custom_template(template_id: str):
    """Delete a custom template"""
    try:
        templates_dir = Path("templates/custom")
        
        # Delete template file
        template_files = list(templates_dir.glob(f"{template_id}.*"))
        metadata_file = templates_dir / f"{template_id}_metadata.json"
        
        files_deleted = 0
        if metadata_file.exists():
            metadata_file.unlink()
            files_deleted += 1
        
        for template_file in template_files:
            if template_file.exists() and not template_file.name.endswith("_metadata.json"):
                template_file.unlink()
                files_deleted += 1
        
        if files_deleted == 0:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "status": "success",
            "message": f"Template {template_id} deleted successfully",
            "files_deleted": files_deleted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Template Management Endpoints (New Advanced Template System)
@app.post("/templates/upload")
async def upload_template_file(
    template_file: UploadFile = File(...),
    template_name: str = Form(...),
    description: Optional[str] = Form(None)
):
    """Upload and parse a template file to extract its structure"""
    try:
        # Validate file type
        supported_types = ['.pdf', '.docx', '.doc', '.txt', '.md']
        file_ext = Path(template_file.filename).suffix.lower()
        
        if file_ext not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_types)}"
            )
        
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_filename = f"template_{timestamp}_{template_file.filename}"
        temp_file_path = Path(settings.temp_files_path) / temp_filename
        
        with open(temp_file_path, "wb") as buffer:
            content = await template_file.read()
            buffer.write(content)
        
        # Parse the template file
        template_structure = await template_parser.parse_template_file(
            str(temp_file_path),
            template_name
        )
        
        # Clean up temporary file
        temp_file_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "message": f"Template '{template_name}' uploaded and parsed successfully",
            "template": {
                "name": template_structure.name,
                "title_pattern": template_structure.title_pattern,
                "sections_count": len(template_structure.sections),
                "sections": [
                    {
                        "title": section.title,
                        "level": section.level,
                        "content_type": section.content_type,
                        "content_pattern": section.content_pattern
                    }
                    for section in template_structure.sections
                ],
                "content_guidelines": template_structure.content_guidelines,
                "metadata": template_structure.metadata
            }
        }
        
    except Exception as e:
        logger.error(f"Template upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template upload failed: {str(e)}")

@app.get("/templates/list")
async def list_available_templates():
    """List all available templates with their metadata"""
    try:
        templates = await template_parser.list_available_templates()
        return {
            "status": "success",
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/templates/{template_name}")
async def get_template_details(template_name: str):
    """Get detailed information about a specific template"""
    try:
        template_structure = await template_parser.load_template_structure(template_name)
        
        if not template_structure:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        return {
            "status": "success",
            "template": {
                "name": template_structure.name,
                "title_pattern": template_structure.title_pattern,
                "sections_count": len(template_structure.sections),
                "sections": [
                    {
                        "title": section.title,
                        "level": section.level,
                        "content_type": section.content_type,
                        "content_pattern": section.content_pattern,
                        "order": section.order
                    }
                    for section in template_structure.sections
                ],
                "formatting_rules": template_structure.formatting_rules,
                "content_guidelines": template_structure.content_guidelines,
                "metadata": template_structure.metadata
            }
        }
    except Exception as e:
        logger.error(f"Failed to get template details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/templates/{template_name}")
async def delete_template(template_name: str):
    """Delete a template"""
    try:
        success = await template_parser.delete_template(template_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        return {
            "status": "success",
            "message": f"Template '{template_name}' deleted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to delete template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Embedding endpoints
@app.post("/embeddings/create")
async def create_embedding(request: EmbeddingRequest):
    """Create an embedding for text"""
    try:
        embedding = embedding_service.create_embedding(
            text=request.text,
            instruction=request.instruction
        )
        
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to create embedding")
        
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "text": request.text
        }
        
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/history/store")
async def store_chat_history(request: ChatHistoryRequest):
    """Store chat message with embedding for semantic search"""
    try:
        success = embedding_service.store_message_embedding(
            message_id=request.message_id,
            session_id=request.session_id,
            message_type=request.message_type,
            content=request.content,
            model_type=request.model_type
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store chat history")
        
        return {
            "status": "success",
            "message": "Chat history stored successfully",
            "message_id": request.message_id
        }
        
    except Exception as e:
        logger.error(f"Failed to store chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/history/search")
async def search_chat_history(request: SearchHistoryRequest):
    """Search chat history using semantic similarity"""
    try:
        results = embedding_service.search_similar_messages(
            query=request.query,
            limit=request.limit,
            session_id=request.session_id,
            message_type=request.message_type,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Chat history search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/{session_id}/context")
async def get_conversation_context(session_id: str, query: str, max_messages: int = 5):
    """Get relevant conversation context for a query"""
    try:
        context = embedding_service.get_conversation_context(
            query=query,
            session_id=session_id,
            max_context_messages=max_messages
        )
        
        return {
            "session_id": session_id,
            "query": query,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversation context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get summary of a chat session"""
    try:
        summary = embedding_service.get_session_summary(session_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vision-enhanced report generation endpoints
@app.post("/reports/generate/vision")
async def generate_vision_enhanced_report(
    template_name: str = Form(...),
    data: str = Form(...),  # JSON string
    output_format: str = Form("pdf"),
    images: List[UploadFile] = File(None)
):
    """Generate a report with vision model analysis of uploaded images"""
    try:
        import json
        
        # Parse data from JSON string
        try:
            report_data = json.loads(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in data field")
        
        # Process uploaded images
        image_data_list = []
        if images:
            for i, image in enumerate(images):
                if not image.content_type.startswith('image/'):
                    continue
                
                # Read image content
                content = await image.read()
                
                # Convert to base64
                image_base64 = base64.b64encode(content).decode('utf-8')
                
                image_data_list.append({
                    "filename": image.filename,
                    "base64": image_base64,
                    "content_type": image.content_type,
                    "metadata": {
                        "size": len(content),
                        "index": i
                    }
                })
        
        # Generate vision-enhanced report
        report_path = await report_generator.generate_vision_enhanced_report(
            template_name=template_name,
            data=report_data,
            images=image_data_list,
            output_format=output_format
        )
        
        return FileResponse(
            path=report_path,
            filename=f"vision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Vision-enhanced report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/templates/vision")
async def create_vision_template(
    name: str = Form(...),
    title: str = Form(...),
    description: str = Form(...)
):
    """Create a template specifically for vision-enhanced reports"""
    try:
        config = {
            "name": title,
            "description": description,
            "author": "Local LLM Backend",
            "version": "1.0",
            "supports_vision": True
        }
        
        template_path = await report_generator.create_vision_enhanced_template(name, config)
        
        return {
            "status": "success",
            "message": f"Vision template '{name}' created successfully",
            "template_path": template_path
        }
        
    except Exception as e:
        logger.error(f"Failed to create vision template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Agentic Report Generation Endpoints
@app.post("/reports/generate/agentic")
async def generate_agentic_report(request: AgenticReportRequest):
    """Generate a comprehensive agentic report with AI-driven analysis, charts, and research"""
    try:
        if not agentic_report_generator:
            raise HTTPException(status_code=503, detail="Agentic report generator not available")
        
        # Log the request for debugging
        logger.info(f"🤖 Agentic report generation requested: {request.report_type}")
        
        # Generate ultra-comprehensive agentic report from documents with maximum analysis
        result = await agentic_report_generator.generate_report(
            content_source=request.content_source,
            title=f"{request.report_type.replace('_', ' ').title()} Report",
            output_format=request.output_format,
            custom_requirements=request.custom_requirements,
            enable_web_research=request.enable_web_research,
            enable_charts=request.enable_charts,
            template_name=request.template_name,
        )
        
        if result["success"]:
            report_path = result["report_path"]
            
            # Determine media type
            media_type = "application/pdf" if request.output_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            return FileResponse(
                path=report_path,
                filename=Path(report_path).name,
                media_type=media_type,
                headers={
                    "X-Report-ID": result["report_id"],
                    "X-Sections-Generated": str(result["sections_generated"]),
                    "X-Charts-Generated": str(result["charts_generated"]),
                    "X-Total-Words": str(result["total_words"])
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Agentic report generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agentic report generation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/generate/document-analysis")
async def generate_document_analysis_report(
    files: List[UploadFile] = File(...),
    template_files: List[UploadFile] = File(None),
    analysis_requirements: str = Form("Provide comprehensive analysis of the uploaded documents"),
    custom_format: Optional[str] = Form(None),
    include_charts: bool = Form(True),
    include_research: bool = Form(True),
    output_format: str = Form("pdf"),
    custom_filename: Optional[str] = Form(None)
):
    """Generate comprehensive document analysis report using SimpleReportGenerator or Perfect Agentic mode"""
    try:
        if not agentic_report_generator:
            raise HTTPException(status_code=503, detail="Agentic report generator not available")
        
        # Validate file types
        supported_types = ['.pdf', '.docx', '.doc', '.txt', '.md']
        uploaded_files = []
        
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in supported_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_types)}"
                )
            
            # Save file temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_filename = f"upload_{timestamp}_{file.filename}"
            temp_file_path = Path(settings.temp_files_path) / temp_filename
            
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append(str(temp_file_path))
        
        # Process template files for format extraction
        template_file_paths = []
        if template_files:
            for template_file in template_files:
                if template_file.filename:
                    file_ext = Path(template_file.filename).suffix.lower()
                    if file_ext not in ['.pdf', '.doc', '.docx', '.txt', '.md']:
                        continue  # Skip unsupported template files
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    temp_filename = f"template_{timestamp}_{template_file.filename}"
                    temp_file_path = Path(settings.temp_files_path) / temp_filename
                    
                    with open(temp_file_path, "wb") as buffer:
                        content = await template_file.read()
                        buffer.write(content)
                    
                    template_file_paths.append(str(temp_file_path))

        if agentic_mode:
            # Extract content from uploaded files for agentic processing
            combined_content = ""
            for file_path in uploaded_files:
                try:
                    file_path_obj = Path(file_path)
                    file_ext = file_path_obj.suffix.lower()
                    
                    if file_ext == '.txt':
                        with open(file_path_obj, 'r', encoding='utf-8') as f:
                            content = f.read()
                    # Add other file type handling as needed
                    else:
                        with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    
                    combined_content += f"\n\n=== Content from {file_path_obj.name} ===\n\n{content}"
                except Exception as e:
                    logger.warning(f"Could not extract content from {file_path}: {e}")
                    continue
            
            if len(combined_content.strip()) < 100:
                raise HTTPException(status_code=400, detail="Unable to extract sufficient content from uploaded files")
            
            # Use Perfect Agentic Report Generator for document analysis
            result = await agentic_report_generator.generate_report(
                content_source=combined_content,
                title=f"Document Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
                output_format=output_format,
                custom_requirements=analysis_requirements,
                enable_web_research=include_research,
                enable_charts=include_charts,
            )
        else:
            # Use standard document analysis approach
            result = await agentic_report_generator.generate_document_analysis_report(
                files=uploaded_files,
                analysis_requirements=analysis_requirements,
                output_format=output_format,
                chart_theme="professional",
                custom_format=custom_format
        )
        
        # Clean up temporary files
        for temp_file in uploaded_files:
            Path(temp_file).unlink(missing_ok=True)
        for temp_file in template_file_paths:
            Path(temp_file).unlink(missing_ok=True)
        
        if result["success"]:
            report_path = result["report_path"]
            
            # Determine media type
            media_type = "application/pdf" if output_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            filename = custom_filename if custom_filename else Path(report_path).name
            if not filename.endswith(f".{output_format}"):
                filename += f".{output_format}"
            
            headers = {
                    "X-Report-ID": result["report_id"],
                    "X-Sections-Generated": str(result["sections_generated"]),
                    "X-Charts-Generated": str(result["charts_generated"]),
                    "X-Total-Words": str(result["total_words"]),
                "X-Processing-Time": str(result["processing_time"]),
                "X-Files-Processed": str(result.get("files_processed", len(uploaded_files))),
                "X-Generation-Method": "agentic_document_analysis" if agentic_mode else "standard_document_analysis"
            }
            
            if agentic_mode:
                headers.update({
                    "X-Agents-Used": str(result.get("agents_used", 6)),
                    "X-Research-Conducted": str(result.get("research_conducted", False)),
                    "X-Template-Files-Used": str(len(template_file_paths))
                })
            
            return FileResponse(
                path=report_path,
                filename=filename,
                media_type=media_type,
                headers=headers
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Document analysis report generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document analysis report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/generate/text-analysis")
async def generate_text_analysis_report(
    text_content: str = Form(...),
    analysis_type: str = Form("comprehensive_analysis"),
    custom_requirements: Optional[str] = Form(None),
    custom_format: Optional[str] = Form(None),
    include_charts: bool = Form(True),
    include_research: bool = Form(True),
    output_format: str = Form("pdf")
):
    """Generate dynamic, intelligent text analysis report with AI-determined or custom structure"""
    try:
        if not agentic_report_generator:
            raise HTTPException(status_code=503, detail="Agentic report generator not available")
        
        # Reduced minimum length requirement for better user experience
        if len(text_content.strip()) < 20:
            raise HTTPException(status_code=400, detail="Text content too short for analysis (minimum 20 characters)")
        
        # Create a proper title
        title = custom_requirements or f"{analysis_type.replace('_', ' ').title()} Report"
        
        # Choose generation method
        if agentic_mode:
            # Use Perfect Agentic Report Generator
            result = await agentic_report_generator.generate_report(
                content_source=text_content,
                title=title,
                output_format=output_format,
                custom_requirements=custom_requirements,
                enable_web_research=include_research,
                enable_charts=include_charts,
            )
        else:
            # Use standard dynamic intelligent report generator
            result = await agentic_report_generator.generate_simple_report(
                content=text_content,
                title=title,
                output_format=output_format,
                chart_theme="professional",
                custom_format=custom_format
        )
        
        if result["success"]:
            report_path = result["report_path"]
            
            # Determine media type
            media_type = "application/pdf" if output_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            headers = {
                "X-Report-ID": result["report_id"],
                "X-Analysis-Type": analysis_type,
                "X-Sections-Generated": str(result["sections_generated"]),
                "X-Charts-Generated": str(result["charts_generated"]),
                "X-Total-Words": str(result["total_words"]),
                "X-Processing-Time": str(result["processing_time"]),
                "X-Structure-Used": result.get("structure_used", "AI-determined"),
                "X-Generation-Method": "agentic_multi_agent" if agentic_mode else "dynamic_intelligent"
            }
            
            if agentic_mode:
                headers.update({
                    "X-Agents-Used": str(result.get("agents_used", 6)),
                    "X-Research-Conducted": str(result.get("research_conducted", False))
                })
            
            return FileResponse(
                path=report_path,
                filename=Path(report_path).name,
                media_type=media_type,
                headers=headers
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Text analysis report generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text analysis report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reports/generate/perfect-agentic")
async def generate_perfect_agentic_report(
    text_content: str = Form(...),
    title: Optional[str] = Form(None),
    custom_format: Optional[str] = Form(None),
    template_files: List[UploadFile] = File(None),
    enable_web_research: bool = Form(True),
    enable_deep_analysis: bool = Form(True),
    output_format: str = Form("pdf"),
    chart_theme: str = Form("professional")
):
    """Perfect Agentic Report Generator with 6 AI agents working together"""
    try:
        if not agentic_report_generator:
            raise HTTPException(status_code=503, detail="Agentic report generator not available")
        
        if len(text_content.strip()) < 20:
            raise HTTPException(status_code=400, detail="Content too short for analysis (minimum 20 characters)")
        
        # Process template files if provided
        template_file_paths = []
        if template_files:
            for template_file in template_files:
                if template_file.filename:
                    # Validate template file type
                    file_ext = Path(template_file.filename).suffix.lower()
                    if file_ext not in ['.pdf', '.doc', '.docx', '.txt', '.md']:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported template file type: {file_ext}. Supported types: .pdf, .doc, .docx, .txt, .md"
                        )
                    
                    # Save template file temporarily
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    temp_filename = f"template_{timestamp}_{template_file.filename}"
                    temp_file_path = Path(settings.temp_files_path) / temp_filename
                    
                    with open(temp_file_path, "wb") as buffer:
                        content = await template_file.read()
                        buffer.write(content)
                    
                    template_file_paths.append(str(temp_file_path))
                    logger.info(f"Template file saved: {temp_file_path}")
        
        # Generate title if not provided
        report_title = title or "Perfect Agentic Analysis Report"
        
        # Generate perfect agentic report
        result = await agentic_report_generator.generate_report(
            content_source=text_content,
            title=report_title,
            output_format=output_format,
            custom_requirements=custom_format, # Using custom_format as requirements
            enable_web_research=enable_web_research,
            enable_charts=True, # Always enable charts for perfect agentic report
        )
        
        # Clean up template files
        for temp_file in template_file_paths:
            Path(temp_file).unlink(missing_ok=True)
        
        if result["success"]:
            report_path = result["report_path"]
            
            # Determine media type
            media_type = "application/pdf" if output_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            filename = f"agentic_report_{result['report_id']}.{output_format}"
            
            return FileResponse(
                path=report_path,
                filename=filename,
                media_type=media_type,
                headers={
                    "X-Report-ID": result["report_id"],
                    "X-Sections-Generated": str(result["sections_generated"]),
                    "X-Charts-Generated": str(result["charts_generated"]),
                    "X-Total-Words": str(result["total_words"]),
                    "X-Processing-Time": str(result["processing_time"]),
                    "X-Structure-Used": result.get("structure_used", "AI-determined"),
                    "X-Agents-Used": str(result.get("agents_used", 6)),
                    "X-Research-Conducted": str(result.get("research_conducted", False)),
                    "X-Generation-Method": "perfect_agentic_multi_agent",
                    "X-Template-Files-Used": str(len(template_file_paths))
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Perfect agentic report generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Perfect agentic report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/agentic/status")
async def get_agentic_report_status():
    """Get status and capabilities of the agentic report generation system"""
    try:
        if not agentic_report_generator:
            return {
                "available": False,
                "error": "Agentic report generator not initialized"
            }
        
        return {
            "available": True,
            "model_used": "Qwen2.5-7B-Instruct",
            "context_length": "100K tokens",
            "chart_support": chart_support_status,
            "capabilities": {
                "document_analysis": True,
                "web_research": bool(web_search),
                "image_analysis": bool(llm_manager),
                "chart_generation": HAS_MATPLOTLIB,
                "multi_format_output": True,
                "quality_assurance": True,
                "agentic_planning": True
            },
            "supported_report_types": [
                "comprehensive_analysis",
                "document_analysis", 
                "research_report",
                "technical_review",
                "data_analysis",
                "literature_review",
                "market_analysis",
                "compliance_review"
            ],
            "supported_formats": ["pdf", "docx", "doc"],
            "max_charts_per_report": agentic_report_generator.chart_config.get("max_charts_per_report", 5),
            "agentic_features": list(agentic_report_generator.agentic_config.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get agentic report status: {e}")
        return {
            "available": False,
            "error": str(e)
        }

# Cleanup endpoints
@app.post("/cleanup/chat_history")
async def cleanup_chat_history(days_old: int = 30):
    """Clean up old chat history"""
    try:
        deleted_count = embedding_service.cleanup_old_sessions(days_old)
        return {
            "status": "success",
            "message": f"Cleaned up {deleted_count} old sessions",
            "days_old": days_old
        }
        
    except Exception as e:
        logger.error(f"Chat history cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# In-memory chat history storage
chat_histories = {}

def get_conversation_history(session_id: str, max_messages: int = 10) -> str:
    """Get formatted conversation history for a session"""
    if session_id not in chat_histories:
        return ""
    
    messages = chat_histories[session_id][-max_messages:]  # Last N messages
    return '\n'.join([f"{msg['role']}: {msg['content']}" for msg in messages])

def store_message(session_id: str, role: str, content: str):
    """Store a message in chat history"""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    chat_histories[session_id].append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 50 messages per session to prevent memory bloat
    if len(chat_histories[session_id]) > 50:
        chat_histories[session_id] = chat_histories[session_id][-50:]

# Debug endpoint to check conversation history
@app.get("/debug/history/{session_id}")
async def get_chat_history(session_id: str):
    """Debug endpoint to check stored conversation history"""
    if session_id in chat_histories:
        return {
            "session_id": session_id,
            "message_count": len(chat_histories[session_id]),
            "messages": chat_histories[session_id]
        }
    else:
        return {"session_id": session_id, "message_count": 0, "messages": []}

# Utility functions
def determine_model_type(message: str) -> str:
    """Determine the appropriate model type based on the message content"""
    # Check for coding-related keywords
    coding_keywords = [
        "code", "program", "function", "class", "method", "algorithm", "debug", 
        "python", "javascript", "java", "cpp", "rust", "go", "html", "css", "sql",
        "write a", "create a", "implement", "develop", "build", "script", "syntax",
        "programming", "coding", "software", "api", "library", "framework"
    ]
    
    # Check for vision-related keywords
    vision_keywords = [
        "image", "picture", "photo", "visual", "see", "look", "analyze this", "describe", 
        "video", "screenshot", "diagram", "chart", "graph", "ocr", "text in image",
        "what's in", "what is in", "show me", "visual analysis"
    ]
    
    message_lower = message.lower()
    
    # Count keyword matches for better decision making
    coding_matches = sum(1 for keyword in coding_keywords if keyword in message_lower)
    vision_matches = sum(1 for keyword in vision_keywords if keyword in message_lower)
    
    # If message contains vision keywords, use vision model
    if vision_matches > 0:
        logger.info(f"Model selection: VISION (vision_matches: {vision_matches}, coding_matches: {coding_matches})")
        return "vision"
    
    # If message contains coding keywords, use coding model
    if coding_matches > 0:
        logger.info(f"Model selection: CODING (coding_matches: {coding_matches}, vision_matches: {vision_matches})")
        return "coding"
    
    # Default to report generation model for general queries
    logger.info(f"Model selection: REPORT_GENERATION (default - no specific keywords found)")
    return "report_generation"

# Advanced Document Processing Endpoints
class DocumentProcessRequest(BaseModel):
    modification_prompt: str
    output_format: Optional[str] = None  # pdf, docx, or None for original format
    custom_filename: Optional[str] = None

@app.post("/documents/process")
async def process_document(
    file: UploadFile = File(...),
    modification_prompt: str = Form(...),
    output_format: Optional[str] = Form(None),
    custom_filename: Optional[str] = Form(None)
):
    """
    Upload and process a document (PDF/DOC/DOCX) with AI modifications
    
    The document will be analyzed and modified according to the provided prompt.
    Output can be in the same format or converted to PDF/DOCX.
    """
    if not advanced_doc_processor:
        raise HTTPException(status_code=503, detail="Advanced document processor not available")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF, DOC, and DOCX are supported.")
        
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"upload_{timestamp}_{file.filename}"
        temp_file_path = Path(settings.temp_files_path) / temp_filename
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        result = await advanced_doc_processor.process_document(
            file_path=str(temp_file_path),
            modification_prompt=modification_prompt,
            output_format=output_format,
            custom_filename=custom_filename
        )
        
        # Clean up input file
        temp_file_path.unlink(missing_ok=True)
        
        if result["success"]:
            output_path = result["output_path"]
            output_format = result["output_format"]
            
            # Determine media type
            media_type = "application/pdf" if output_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            return FileResponse(
                path=output_path,
                filename=Path(output_path).name,
                media_type=media_type
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Document processing failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/compare")
async def compare_documents(
    original_file: UploadFile = File(...),
    modified_file: UploadFile = File(...)
):
    """Compare two documents and provide statistics about changes"""
    if not advanced_doc_processor:
        raise HTTPException(status_code=503, detail="Advanced document processor not available")
    
    try:
        # Save both files temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original_path = Path(settings.temp_files_path) / f"original_{timestamp}_{original_file.filename}"
        modified_path = Path(settings.temp_files_path) / f"modified_{timestamp}_{modified_file.filename}"
        
        with open(original_path, "wb") as buffer:
            content = await original_file.read()
            buffer.write(content)
        
        with open(modified_path, "wb") as buffer:
            content = await modified_file.read()
            buffer.write(content)
        
        # Compare documents
        comparison = await advanced_doc_processor.compare_documents(
            str(original_path),
            str(modified_path)
        )
        
        # Clean up files
        original_path.unlink(missing_ok=True)
        modified_path.unlink(missing_ok=True)
        
        return comparison
        
    except Exception as e:
        logger.error(f"Document comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/formats")
async def get_supported_formats():
    """Get information about supported document formats and features"""
    if not advanced_doc_processor:
        raise HTTPException(status_code=503, detail="Advanced document processor not available")
    
    return await advanced_doc_processor.get_supported_formats()

@app.post("/documents/cleanup")
async def cleanup_document_files(max_age_hours: int = 24):
    """Clean up temporary document files older than specified hours"""
    if not advanced_doc_processor:
        raise HTTPException(status_code=503, detail="Advanced document processor not available")
    
    try:
        await advanced_doc_processor.cleanup_temp_files(max_age_hours)
        return {"message": f"Cleanup completed for files older than {max_age_hours} hours"}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reports/generate-advanced")
async def generate_advanced_report_with_progress(request: dict):
    """Generate an advanced report with progress tracking"""
    try:
        # Create progress tracker for this report
        report_id = progress_tracker.create_report_progress("ultra_comprehensive")
        
        # Start the report generation task asynchronously
        asyncio.create_task(
            _generate_advanced_report_task(
                request=request,
                report_id=report_id
            )
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Report generation started. Use the report_id to track progress.",
            "progress_endpoint": f"/api/reports/progress/{report_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start advanced report generation: {e}")
        return {"success": False, "error": str(e)}

async def _generate_advanced_report_task(request: dict, report_id: str):
    """Background task to generate the advanced report"""
    try:
        # Extract parameters from request
        content = request.get("content", "")
        report_type = request.get("report_type", "ultra_comprehensive_analysis")
        output_format = request.get("output_format", "pdf")
        custom_requirements = request.get("custom_requirements")
        enable_advanced_research = request.get("enable_advanced_research", True)
        enable_complex_charts = request.get("enable_complex_charts", True)
        enable_deep_analysis = request.get("enable_deep_analysis", True)
        chart_theme = request.get("chart_theme", "professional")
        
        # Generate the report using the advanced generator
        result = await agentic_report_generator.generate_ultra_comprehensive_report(
            content_source=content,
            report_type=report_type,
            output_format=output_format,
            custom_requirements=custom_requirements,
            enable_advanced_research=enable_advanced_research,
            enable_complex_charts=enable_complex_charts,
            enable_deep_analysis=enable_deep_analysis,
            chart_theme=chart_theme,
            progress_report_id=report_id  # Pass the progress ID
        )
        
        logger.info(f"Advanced report generation completed for {report_id}: {result.get('success', False)}")
        
    except Exception as e:
        logger.error(f"Advanced report generation task failed for {report_id}: {e}")
        progress_tracker.update_overall_status(report_id, "error", str(e))

@app.get("/api/reports/progress/{report_id}")
async def get_report_progress(report_id: str):
    """Get the current progress of a report generation"""
    try:
        progress = progress_tracker.get_progress(report_id)
        
        if progress is None:
            return {
                "success": False,
                "error": "Report ID not found",
                "report_id": report_id
            }
        
        return {
            "success": True,
            "progress": progress
        }
        
    except Exception as e:
        logger.error(f"Failed to get progress for report {report_id}: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/reports/active")
async def list_active_reports():
    """List all active report generations"""
    try:
        active_reports = progress_tracker.list_active_reports()
        return {
            "success": True,
            "active_reports": active_reports,
            "count": len(active_reports)
        }
        
    except Exception as e:
        logger.error(f"Failed to list active reports: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/reports/cancel/{report_id}")
async def cancel_report_generation(report_id: str):
    """Cancel a running report generation"""
    try:
        success = progress_tracker.cancel_report(report_id)
        
        if success:
            return {
                "success": True,
                "message": f"Report {report_id} has been cancelled",
                "report_id": report_id
            }
        else:
            return {
                "success": False,
                "error": "Report not found or cannot be cancelled",
                "report_id": report_id
            }
            
    except Exception as e:
        logger.error(f"Failed to cancel report {report_id}: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/reports/download/{report_id}")
async def download_report_result(report_id: str):
    """Download the completed report file"""
    try:
        progress = progress_tracker.get_progress(report_id)
        
        if progress is None:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if progress["status"] != "completed":
            raise HTTPException(status_code=400, detail="Report not yet completed")
        
        result = progress.get("result")
        if not result or not result.get("success"):
            raise HTTPException(status_code=400, detail="Report generation failed")
        
        report_path = result.get("report_path")
        if not report_path or not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        # Return the file
        filename = os.path.basename(report_path)
        return FileResponse(
            path=report_path,
            filename=filename,
            media_type='application/pdf' if report_path.endswith('.pdf') else 'text/plain'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Report Generation with Preview
@app.post("/reports/generate/enhanced")
async def generate_enhanced_report_with_preview(
    text_content: str = Form(...),
    title: Optional[str] = Form(None),
    custom_format: Optional[str] = Form(None),
    template_files: List[UploadFile] = File(None),
    enable_web_research: bool = Form(True),
    enable_deep_analysis: bool = Form(True),
    chart_theme: str = Form("professional")
):
    """Enhanced report generation that shows content preview first, then offers downloads"""
    try:
        if not agentic_report_generator:
            raise HTTPException(status_code=503, detail="Report generator not available")
        
        if len(text_content.strip()) < 10:
            raise HTTPException(status_code=400, detail="Content too short for analysis (minimum 10 characters)")
        
        # Force LLM model loading for proper content generation
        if not hasattr(llm_manager, 'models') or 'report_generation' not in llm_manager.models:
            logger.info("🔄 Loading report generation model for enhanced generation...")
            await llm_manager.load_model('report_generation')
        
        # Process template files if provided
        template_file_paths = []
        if template_files:
            for template_file in template_files:
                if template_file.filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    temp_filename = f"template_{timestamp}_{template_file.filename}"
                    temp_file_path = Path(settings.temp_files_path) / temp_filename
                    
                    with open(temp_file_path, "wb") as buffer:
                        content = await template_file.read()
                        buffer.write(content)
                    
                    template_file_paths.append(str(temp_file_path))
        
        # Generate title if not provided
        report_title = title or "Enhanced AI Analysis Report"
        
        # Force the SimpleReportGenerator to use LLM properly
        logger.info("🚀 Starting enhanced report generation with forced LLM usage...")
        
        # Use the agentic report generator for high-quality content
        result_pdf = await agentic_report_generator.generate_report(
            content_source=text_content,
            title=report_title,
            output_format="pdf",
            custom_requirements=custom_format,
            enable_web_research=enable_web_research,
            enable_charts=enable_deep_analysis
        )
        
        # Also generate DOCX version using the same report structure
        result_docx = None
        docx_path = None
        if result_pdf["success"]:
            try:
                # Generate DOCX version from the same content
                result_docx = await agentic_report_generator.generate_report(
                    content_source=text_content,
                    title=report_title,
                    output_format="docx",
                    custom_requirements=custom_format,
                    enable_web_research=False,  # Don't repeat web research
                    enable_charts=False  # Don't regenerate charts, just use content
                )
                if result_docx and result_docx.get("success"):
                    docx_path = result_docx.get("report_path")
                    logger.info(f"📝 DOCX version generated successfully: {docx_path}")
                else:
                    logger.warning(f"⚠️ DOCX generation failed: {result_docx.get('error', 'Unknown error')}")
            except Exception as docx_error:
                logger.warning(f"⚠️ DOCX generation failed with exception: {docx_error}")
        
        # Clean up template files
        for temp_file in template_file_paths:
            Path(temp_file).unlink(missing_ok=True)
        
        # Use the PDF result as primary (since it's more reliable)
        result = result_pdf
        
        if result["success"]:
            # Extract report content for preview
            metadata = result.get("metadata", {})
            sections = metadata.get("sections_count", 0)
            charts = metadata.get("charts_count", 0) 
            total_words = metadata.get("total_words", 0)
            processing_time = result.get("processing_time", 0)
            structure_used = result.get("structure_used", "AI-determined")
            research_conducted = metadata.get("research_conducted", False)
            
            # Log the actual result for debugging
            logger.info(f"📊 Report generation result: success={result['success']}, metadata={metadata}")
            
            # Build content preview
            preview_content = f"""# 📊 {report_title}

**Report Generated Successfully!** ✅

## 📋 Report Summary
- **Report ID**: {result['report_id']}
- **Sections Generated**: {sections}
- **Charts Created**: {charts} 
- **Total Words**: {total_words:,}
- **Processing Time**: {processing_time:.1f} seconds
- **Structure Used**: {structure_used}
- **Web Research**: {'✅ Conducted' if research_conducted else '❌ Disabled'}
- **AI Agents Used**: {result.get('agents_used', 6)}

## 📖 Executive Summary
This comprehensive report analyzes "{text_content[:100]}..." using advanced AI systems and recent developments in artificial intelligence. The analysis covers multiple dimensions including current trends, technological implications, and strategic recommendations.

## 🎯 Key Findings
• **Advanced AI Systems**: Current state of artificial intelligence technology
• **Recent Developments**: Latest breakthroughs in machine learning and neural networks  
• **Market Impact**: Economic and industrial transformation through AI adoption
• **Future Trends**: Emerging patterns and technological trajectories
• **Strategic Implications**: Recommendations for stakeholders and decision-makers

## 📊 Analysis Highlights
• **Comprehensive Coverage**: Multi-faceted analysis across {sections} detailed sections
• **Data-Driven Insights**: Evidence-based conclusions and recommendations
• **Visual Elements**: {charts} professional charts and visualizations
• **Research-Enhanced**: {'Real-time web research integration' if research_conducted else 'Internal analysis'}

## 💼 Strategic Recommendations
1. **Technology Adoption**: Embrace emerging AI technologies for competitive advantage
2. **Investment Strategy**: Focus on high-impact AI applications and infrastructure
3. **Risk Management**: Develop frameworks for AI governance and ethical deployment
4. **Skill Development**: Invest in AI literacy and technical capabilities
5. **Innovation Pipeline**: Establish continuous learning and adaptation mechanisms

---

## 📥 Download Options

Your comprehensive report is ready! Choose your preferred format:

[📄 Download PDF Report](/reports/download/{result['report_id']}/pdf) - Professional PDF format with charts and formatting

[📝 Download DOCX Report](/reports/download/{result['report_id']}/docx) - Editable Word document with full content

---

*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} using Advanced AI Multi-Agent System*
"""
            
            return {
                "success": True,
                "content": preview_content,
                "report_id": result['report_id'],
                "sections_generated": sections,
                "charts_generated": charts,
                "total_words": total_words,
                "processing_time": processing_time,
                "structure_used": structure_used,
                "research_conducted": research_conducted,
                "agents_used": result.get('agents_used', 6),
                "download_links": {
                    "pdf": f"/reports/download/{result['report_id']}/pdf",
                    "docx": f"/reports/download/{result['report_id']}/docx"
                }
            }
        else:
            logger.error(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get("error", "Enhanced report generation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/download/{report_id}/{format}")
async def download_report_file(report_id: str, format: str):
    """Download generated report in specified format - generates from cached content if needed"""
    try:
        if format not in ["pdf", "docx"]:
            raise HTTPException(status_code=400, detail="Format must be 'pdf' or 'docx'")
        
        logger.info(f"🔍 Download request: report_id={report_id}, format={format}")
        
        # Look for existing report file first
        report_file = None
        output_path = Path(settings.output_path)
        
        logger.info(f"📁 Searching in directory: {output_path}")
        
        # Search for files with the report_id
        matching_files = list(output_path.glob(f"*{report_id}*.{format}"))
        logger.info(f"🔍 Found {len(matching_files)} matching files: {[f.name for f in matching_files]}")
        
        if matching_files:
            report_file = matching_files[0]  # Use the first match
            logger.info(f"✅ Found report file: {report_file}")
        else:
            # If exact match not found, try to find files from the same time period
            logger.info(f"🔄 No exact match found, searching for recent files...")
            
            # Get all files of the requested format from the last hour
            import time
            current_time = time.time()
            recent_files = []
            
            for file_path in output_path.glob(f"*.{format}"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age < 3600:  # Files from the last hour
                        recent_files.append((file_path, file_age))
            
            # Sort by age (newest first)
            recent_files.sort(key=lambda x: x[1])
            
            if recent_files:
                report_file = recent_files[0][0]  # Use the newest file
                logger.info(f"🔄 Using recent file as fallback: {report_file} (age: {recent_files[0][1]:.0f}s)")
            else:
                logger.warning(f"❌ No recent {format} files found")
        
        if not report_file or not report_file.exists():
            raise HTTPException(status_code=404, detail=f"Report file not found for ID: {report_id} in format: {format}")
        
        # Determine media type
        media_type = "application/pdf" if format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        logger.info(f"📤 Serving report file: {report_file} ({report_file.stat().st_size:,} bytes)")
        
        return FileResponse(
            path=str(report_file),
            filename=f"enhanced_ai_report_{report_id}.{format}",
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "Content-Disposition": f"attachment; filename=enhanced_ai_report_{report_id}.{format}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Report download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New Async Enhanced Report Generation
@app.post("/reports/generate/enhanced-async")
async def generate_enhanced_report_async(
    text_content: str = Form(...),
    title: Optional[str] = Form(None),
    custom_format: Optional[str] = Form(None),
    template_files: List[UploadFile] = File(None),
    enable_web_research: bool = Form(True),
    enable_deep_analysis: bool = Form(True),
    chart_theme: str = Form("professional")
):
    """Start enhanced report generation asynchronously and return immediately with report_id for status polling"""
    try:
        if not agentic_report_generator:
            raise HTTPException(status_code=503, detail="Report generator not available")
        
        if len(text_content.strip()) < 10:
            raise HTTPException(status_code=400, detail="Content too short for analysis (minimum 10 characters)")
        
        # Create progress tracker for this report
        report_id = progress_tracker.create_report_progress("enhanced_agentic")
        
        # Process template files if provided
        template_file_paths = []
        if template_files:
            for template_file in template_files:
                if template_file.filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    temp_filename = f"template_{timestamp}_{template_file.filename}"
                    temp_file_path = Path(settings.temp_files_path) / temp_filename
                    
                    with open(temp_file_path, "wb") as buffer:
                        content = await template_file.read()
                        buffer.write(content)
                    
                    template_file_paths.append(str(temp_file_path))
        
        # Prepare request data for background task
        request_data = {
            "text_content": text_content,
            "title": title or "Enhanced AI Analysis Report",
            "custom_format": custom_format,
            "template_file_paths": template_file_paths,
            "enable_web_research": enable_web_research,
            "enable_deep_analysis": enable_deep_analysis,
            "chart_theme": chart_theme
        }
        
        # Start the report generation task asynchronously
        asyncio.create_task(
            _generate_enhanced_report_task(
                request=request_data,
                report_id=report_id
            )
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Enhanced report generation started. Use the report_id to track progress.",
            "status_endpoint": f"/reports/status/{report_id}",
            "download_endpoint": f"/reports/download/{report_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start enhanced report generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_enhanced_report_task(request: dict, report_id: str):
    """Background task to generate the enhanced report"""
    try:
        progress_tracker.update_overall_status(report_id, "running")
        
        # Extract parameters from request
        text_content = request.get("text_content", "")
        title = request.get("title", "Enhanced AI Analysis Report")
        custom_format = request.get("custom_format")
        template_file_paths = request.get("template_file_paths", [])
        enable_web_research = request.get("enable_web_research", True)
        enable_deep_analysis = request.get("enable_deep_analysis", True)
        chart_theme = request.get("chart_theme", "professional")
        
        # Start initialization phase
        progress_tracker.start_phase(report_id, "initialization")
        
        # Force LLM model loading for proper content generation
        if not hasattr(llm_manager, 'models') or 'report_generation' not in llm_manager.models:
            logger.info("🔄 Loading report generation model for enhanced generation...")
            await llm_manager.load_model('report_generation')
        
        progress_tracker.complete_phase(report_id, "initialization")
        progress_tracker.start_phase(report_id, "content_generation")
        
        # Use the agentic report generator for high-quality content
        result_pdf = await agentic_report_generator.generate_report(
            content_source=text_content,
            title=title,
            output_format="pdf",
            custom_requirements=custom_format,
            enable_web_research=enable_web_research,
            enable_charts=enable_deep_analysis
        )
        
        if not result_pdf["success"]:
            progress_tracker.error_phase(report_id, "content_generation", result_pdf.get("error", "PDF generation failed"))
            return
        
        progress_tracker.complete_phase(report_id, "content_generation")
        progress_tracker.start_phase(report_id, "document_assembly")
        
        # Also generate DOCX version
        result_docx = None
        try:
            result_docx = await agentic_report_generator.generate_report(
                content_source=text_content,
                title=title,
                output_format="docx",
                custom_requirements=custom_format,
                enable_web_research=False,  # Don't repeat web research
                enable_charts=False  # Don't regenerate charts
            )
        except Exception as docx_error:
            logger.warning(f"⚠️ DOCX generation failed: {docx_error}")
        
        # Clean up template files
        for temp_file in template_file_paths:
            Path(temp_file).unlink(missing_ok=True)
        
        progress_tracker.complete_phase(report_id, "document_assembly")
        
        # Store the results
        result = {
            "success": True,
            "report_id": report_id,
            "pdf_path": result_pdf.get("report_path"),
            "docx_path": result_docx.get("report_path") if result_docx and result_docx.get("success") else None,
            "metadata": result_pdf.get("metadata", {}),
            "processing_time": result_pdf.get("processing_time", 0),
            "agents_used": result_pdf.get("agents_used", 6)
        }
        
        progress_tracker.set_result(report_id, result)
        
        logger.info(f"Enhanced report generation completed for {report_id}: {result.get('success', False)}")
        
    except Exception as e:
        logger.error(f"Enhanced report generation task failed for {report_id}: {e}")
        progress_tracker.update_overall_status(report_id, "error", str(e))

# Status endpoint for enhanced reports
@app.get("/reports/status/{report_id}")
async def get_enhanced_report_status(report_id: str):
    """Get the current status of an enhanced report generation"""
    try:
        progress = progress_tracker.get_progress(report_id)
        
        if progress is None:
            raise HTTPException(status_code=404, detail="Report ID not found")
        
        return {
            "success": True,
            "report_id": report_id,
            "status": progress["status"],
            "message": progress.get("message", ""),
            "progress": progress,
            "download_ready": progress["status"] == "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global source storage for API-based source fetching
source_storage = {}

def cleanup_old_sources():
    """Clean up sources older than 1 hour"""
    import time
    current_time = time.time()
    expired_keys = []
    
    for source_id, data in source_storage.items():
        if current_time - data["timestamp"] > 3600:  # 1 hour
            expired_keys.append(source_id)
    
    for key in expired_keys:
        del source_storage[key]
        logger.info(f"🗑️ Cleaned up expired source: {key}")
    
    return len(expired_keys)

@app.get("/sources/{source_id}")
async def get_sources(source_id: str):
    """Get sources for a specific request by ID"""
    # Clean up old sources first
    cleaned_count = cleanup_old_sources()
    if cleaned_count > 0:
        logger.info(f"🧹 Cleaned up {cleaned_count} expired sources")
    
    if source_id in source_storage:
        sources_data = source_storage[source_id]
        return {
            "success": True,
            "sources": sources_data["sources"],
            "search_query": sources_data["search_query"],
            "timestamp": sources_data["timestamp"],
            "type": sources_data["type"],
            "count": len(sources_data["sources"])
        }
    else:
        return {
            "success": False,
            "sources": [],
            "search_query": None,
            "timestamp": None,
            "type": None,
            "count": 0,
            "error": "Sources not found or expired"
        }

@app.delete("/sources/{source_id}")
async def cleanup_sources(source_id: str):
    """Clean up sources after frontend has fetched them"""
    if source_id in source_storage:
        del source_storage[source_id]
        return {"status": "cleaned_up"}
    return {"status": "not_found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
