import logging
from typing import List, Dict, Optional
from app.core.config import settings

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

VECTOR_AVAILABLE = CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE

logger = logging.getLogger(__name__)

class VectorService:
    """Service for managing vector embeddings and similarity search"""

    def __init__(self):
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Vector search will be disabled.")
            self.available = False
            return

        try:
            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)

            # Initialize embedding model if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_real_embeddings = True
                logger.info("Using SentenceTransformers for embeddings")
            else:
                self.embedding_model = None
                self.use_real_embeddings = False
                logger.warning("SentenceTransformers not available, using simple text-based search")

            # Create collections
            self.frames_collection = self.client.get_or_create_collection(
                name="video_frames",
                metadata={"description": "Video frame embeddings"}
            )

            self.transcript_collection = self.client.get_or_create_collection(
                name="video_transcripts",
                metadata={"description": "Video transcript embeddings"}
            )

            self.available = True
            logger.info("Vector service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector service: {e}")
            self.available = False

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.use_real_embeddings and self.embedding_model:
            return self.embedding_model.encode(text).tolist()
        else:
            # Simple hash-based embedding as fallback
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # Convert hex to list of floats (simple approach)
            embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(32, len(hash_hex)), 2)]
            # Pad to fixed size
            while len(embedding) < 16:
                embedding.append(0.0)
            return embedding[:16]

    async def add_frame_embedding(self,
                                frame_id: str,
                                description: str,
                                metadata: Dict) -> bool:
        """Add frame embedding to vector database"""
        try:
            # Generate embedding
            embedding = self._generate_embedding(description)

            # Add to collection
            self.frames_collection.add(
                embeddings=[embedding],
                documents=[description],
                metadatas=[metadata],
                ids=[frame_id]
            )

            return True

        except Exception as e:
            logger.error(f"Error adding frame embedding: {e}")
            return False

    async def add_transcript_embedding(self,
                                     video_id: str,
                                     transcript: str,
                                     metadata: Dict) -> bool:
        """Add transcript embedding to vector database"""
        try:
            # Split transcript into chunks
            chunks = self._split_transcript(transcript)

            embeddings = []
            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                embedding = self._generate_embedding(chunk)
                embeddings.append(embedding)
                documents.append(chunk)

                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                metadatas.append(chunk_metadata)

                ids.append(f"{video_id}_chunk_{i}")

            # Add to collection
            self.transcript_collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            return True

        except Exception as e:
            logger.error(f"Error adding transcript embedding: {e}")
            return False

    async def search_frames(self,
                          query: str,
                          video_id: Optional[int] = None,
                          limit: int = 10) -> List[Dict]:
        """Search for similar frames"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Prepare where clause
            where_clause = {}
            if video_id:
                where_clause['video_id'] = video_id

            # Search
            results = self.frames_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause if where_clause else None
            )

            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'description': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching frames: {e}")
            return []

    async def search_transcript(self,
                              query: str,
                              video_id: Optional[int] = None,
                              limit: int = 5) -> List[Dict]:
        """Search for relevant transcript segments"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Prepare where clause
            where_clause = {}
            if video_id:
                where_clause['video_id'] = video_id

            # Search
            results = self.transcript_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause if where_clause else None
            )

            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching transcript: {e}")
            return []

    def _split_transcript(self, transcript: str, chunk_size: int = 500) -> List[str]:
        """Split transcript into chunks for embedding"""
        words = transcript.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    async def delete_video_embeddings(self, video_id: int):
        """Delete all embeddings for a video"""
        try:
            # Delete frame embeddings
            self.frames_collection.delete(
                where={"video_id": video_id}
            )

            # Delete transcript embeddings
            self.transcript_collection.delete(
                where={"video_id": video_id}
            )

            logger.info(f"Deleted embeddings for video {video_id}")

        except Exception as e:
            logger.error(f"Error deleting embeddings for video {video_id}: {e}")

    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector collections"""
        try:
            frames_count = self.frames_collection.count()
            transcript_count = self.transcript_collection.count()

            return {
                "frames_count": frames_count,
                "transcript_chunks_count": transcript_count,
                "total_embeddings": frames_count + transcript_count
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "frames_count": 0,
                "transcript_chunks_count": 0,
                "total_embeddings": 0
            }
