"""
Embedding Service for Chat History and Semantic Search

This service uses the Qwen3-Embedding model to create embeddings of chat messages
and enable semantic search through conversation history.
"""

import os
import json
import numpy as np
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
import sqlite3
import json
import numpy as np
from llama_cpp import Llama
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating embeddings and semantic search"""
    
    def __init__(self):
        self.model = None
        self.db_path = Path("./temp_files/chat_embeddings.db")
        
        # Check if embedding config exists
        if "embedding" not in settings.models_config:
            logger.info("🚫 Embedding service disabled in configuration")
            self.embedding_enabled = False
            self.embedding_dim = 1024  # Default fallback for Qwen3
        else:
            self.embedding_dim = settings.models_config["embedding"]["embedding_dimension"]
            self.embedding_enabled = True  # Flag to disable if loading fails
            logger.info(f"📊 Embedding service initialized with {settings.models_config['embedding']['model_name']}")
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing embeddings"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    message_type TEXT,  -- 'user' or 'assistant'
                    content TEXT,
                    timestamp TIMESTAMP,
                    model_type TEXT,
                    embedding BLOB,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
                )
            """)
            
            # Create index for faster searches
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                ON chat_messages (session_id, timestamp)
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize embedding database: {e}")
            self.embedding_enabled = False
    
    def load_model(self) -> bool:
        """Load the embedding model with GPU-only or disable entirely"""
        try:
            if not self.embedding_enabled:
                return False
                
            if self.model is not None:
                return True
            
            if "embedding" not in settings.models_config:
                logger.info("🚫 Embedding configuration not found - service disabled")
                self.embedding_enabled = False
                return False
            
            config = settings.models_config["embedding"]
            model_path = Path(config["model_path"]) / config["model_name"]
            
            if not model_path.exists():
                logger.warning(f"Embedding model not found: {model_path}")
                self.embedding_enabled = False
                return False
            
            logger.info(f"Loading embedding model: {model_path}")
            
            # Try GPU-only loading
            try:
                self.model = Llama(
                    model_path=str(model_path),
                        n_ctx=8192,  # Use reasonable context
                        n_gpu_layers=-1,  # Force all layers on GPU
                        n_threads=8,
                        embedding=True,
                    verbose=False
                )
                logger.info("✅ Embedding model loaded successfully (GPU mode)")
                return True
            except Exception as e:
                logger.warning(f"Failed to load embedding model on GPU: {e}")
                logger.info("💡 Disabling embedding service - system will work without semantic search")
                self.embedding_enabled = False
                return False
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_enabled = False
            return False
    
    def create_embedding(self, text: str, instruction: str = None) -> Optional[np.ndarray]:
        """Create embedding for text with optional instruction"""
        try:
            if not self.embedding_enabled:
                return None
                
            if not self.load_model():
                return None
            
            # Truncate text if too long
            max_length = 1000  # Conservative limit
            if len(text) > max_length:
                text = text[:max_length]
            
            # Add instruction if provided (recommended for better performance)
            if instruction:
                combined_text = f"{instruction}\n{text}"
                if len(combined_text) > max_length:
                    # Prioritize the main text over instruction if needed
                    combined_text = text[:max_length]
            else:
                combined_text = text
            
            # Create embedding
            embedding = self.model.create_embedding(combined_text)
            
            if isinstance(embedding, dict) and 'data' in embedding:
                # Extract embedding vector
                embedding_vector = np.array(embedding['data'][0]['embedding'], dtype=np.float32)
                return embedding_vector
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to create embedding (disabling service): {e}")
            self.embedding_enabled = False
            return None
    
    def store_chat_session(self, session_id: str, title: str, metadata: Dict = None) -> bool:
        """Store or update chat session information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata or {})
            
            # Insert or update session
            cursor.execute("""
                INSERT OR REPLACE INTO chat_sessions 
                (id, title, created_at, updated_at, metadata)
                VALUES (?, ?, 
                    COALESCE((SELECT created_at FROM chat_sessions WHERE id = ?), ?),
                    ?, ?)
            """, (session_id, title, session_id, now, now, metadata_json))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chat session: {e}")
            return False
    
    def store_message_embedding(self, 
                              message_id: str,
                              session_id: str,
                              message_type: str,
                              content: str,
                              model_type: str = None) -> bool:
        """Store message with its embedding"""
        try:
            # Always store the message even if embeddings fail
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            embedding_blob = None
            
            # Try to create embedding if service is enabled
            if self.embedding_enabled:
                # Create appropriate instruction based on message type
                if message_type == "user":
                    instruction = "Represent this user query for semantic search:"
                else:
                    instruction = "Represent this assistant response for semantic search:"
                
                # Create embedding
                embedding = self.create_embedding(content, instruction)
                if embedding is not None:
                    # Convert embedding to binary format
                    embedding_blob = embedding.tobytes()
            
            cursor.execute("""
                INSERT OR REPLACE INTO chat_messages 
                (id, session_id, message_type, content, timestamp, model_type, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id, session_id, message_type, content,
                datetime.now().isoformat(), model_type, embedding_blob
            ))
            
            conn.commit()
            conn.close()
            
            if embedding_blob is not None:
                logger.info(f"Stored message {message_id} with embedding")
            else:
                logger.info(f"Stored message {message_id} without embedding (service disabled)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            return False
    
    def search_similar_messages(self, 
                              query: str, 
                              limit: int = 10,
                              session_id: str = None,
                              message_type: str = None,
                              similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar messages using semantic similarity"""
        try:
            if not self.embedding_enabled:
                # Fallback to simple text search
                return self._fallback_text_search(query, limit, session_id, message_type)
            
            # Create embedding for query
            query_instruction = "Represent this query for semantic search:"
            query_embedding = self.create_embedding(query, query_instruction)
            if query_embedding is None:
                # Fallback to text search
                return self._fallback_text_search(query, limit, session_id, message_type)
            
            # Retrieve stored messages
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query conditions
            conditions = ["embedding IS NOT NULL"]  # Only search messages with embeddings
            params = []
            
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            
            if message_type:
                conditions.append("message_type = ?")
                params.append(message_type)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            cursor.execute(f"""
                SELECT id, session_id, message_type, content, timestamp, model_type, embedding
                FROM chat_messages
                {where_clause}
                ORDER BY timestamp DESC
            """, params)
            
            results = []
            for row in cursor.fetchall():
                msg_id, sess_id, msg_type, content, timestamp, model_type, embedding_blob = row
                
                # Convert embedding back from binary
                stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= similarity_threshold:
                    results.append({
                        "id": msg_id,
                        "session_id": sess_id,
                        "message_type": msg_type,
                        "content": content,
                        "timestamp": timestamp,
                        "model_type": model_type,
                        "similarity": float(similarity)
                    })
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            conn.close()
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search similar messages: {e}")
            # Fallback to simple text search
            return self._fallback_text_search(query, limit, session_id, message_type)
    
    def _fallback_text_search(self, 
                            query: str, 
                            limit: int = 10,
                            session_id: str = None,
                            message_type: str = None) -> List[Dict[str, Any]]:
        """Fallback text search when embeddings are not available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query conditions
            conditions = ["content LIKE ?"]
            params = [f"%{query}%"]
            
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            
            if message_type:
                conditions.append("message_type = ?")
                params.append(message_type)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            cursor.execute(f"""
                SELECT id, session_id, message_type, content, timestamp, model_type
                FROM chat_messages
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit])
            
            results = []
            for row in cursor.fetchall():
                msg_id, sess_id, msg_type, content, timestamp, model_type = row
                results.append({
                    "id": msg_id,
                    "session_id": sess_id,
                    "message_type": msg_type,
                    "content": content,
                    "timestamp": timestamp,
                    "model_type": model_type,
                    "similarity": 0.5  # Default similarity for text search
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform fallback text search: {e}")
            return []
    
    def get_conversation_context(self, 
                               query: str,
                               session_id: str = None,
                               max_context_messages: int = 5) -> str:
        """Get relevant conversation context for a query"""
        try:
            # Search for similar messages
            similar_messages = self.search_similar_messages(
                query=query,
                limit=max_context_messages,
                session_id=session_id,
                similarity_threshold=0.6 if self.embedding_enabled else 0.0
            )
            
            if not similar_messages:
                return ""
            
            # Build context string
            context_parts = []
            for msg in similar_messages:
                role = "User" if msg["message_type"] == "user" else "Assistant"
                content = msg["content"][:500]  # Truncate long messages
                timestamp = msg["timestamp"][:19]  # Remove microseconds
                
                if self.embedding_enabled and "similarity" in msg:
                    similarity = msg["similarity"]
                    context_parts.append(
                        f"[{timestamp}] {role} (similarity: {similarity:.2f}): {content}"
                    )
                else:
                    context_parts.append(
                        f"[{timestamp}] {role}: {content}"
                    )
            
            context = "\n".join(context_parts)
            search_type = "semantic" if self.embedding_enabled else "text"
            return f"Relevant conversation context ({search_type} search):\n{context}\n"
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return ""
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a chat session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute("""
                SELECT title, created_at, updated_at, metadata
                FROM chat_sessions WHERE id = ?
            """, (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return {}
            
            title, created_at, updated_at, metadata_json = session_row
            
            # Get message count and types
            cursor.execute("""
                SELECT message_type, COUNT(*) as count
                FROM chat_messages WHERE session_id = ?
                GROUP BY message_type
            """, (session_id,))
            
            message_counts = dict(cursor.fetchall())
            
            # Get total messages
            cursor.execute("""
                SELECT COUNT(*) FROM chat_messages WHERE session_id = ?
            """, (session_id,))
            
            total_messages = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "session_id": session_id,
                "title": title,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": json.loads(metadata_json or "{}"),
                "total_messages": total_messages,
                "message_counts": message_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {}
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now().replace(day=datetime.now().day - days_old).isoformat()
            
            # Delete old messages first
            cursor.execute("""
                DELETE FROM chat_messages 
                WHERE session_id IN (
                    SELECT id FROM chat_sessions WHERE updated_at < ?
                )
            """, (cutoff_date,))
            
            # Delete old sessions
            cursor.execute("""
                DELETE FROM chat_sessions WHERE updated_at < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} old sessions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0

# Global embedding service instance
embedding_service = EmbeddingService() 