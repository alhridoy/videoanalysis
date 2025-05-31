"""
Enhanced Vector Service with Direct Image Embeddings and Hybrid Search
Implements the suggested improvements for production-ready multimodal search
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import torch
import clip
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with multiple embedding types"""
    frame_id: str
    timestamp: float
    confidence: float
    description: str
    frame_path: str
    image_similarity: float
    text_similarity: float
    keyword_matches: List[str]
    metadata: Dict[str, Any]

class EnhancedVectorService:
    """Enhanced vector service with direct image embeddings and hybrid search"""
    
    def __init__(self):
        self.client = None
        self.frames_collection = None
        self.available = False
        
        # Multiple embedding models
        self.clip_model = None
        self.clip_preprocess = None
        self.text_model = None
        
        # Hybrid search components
        self.keyword_index = {}  # Simple keyword index
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize ChromaDB and embedding models"""
        try:
            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create enhanced collection with metadata
            self.frames_collection = self.client.get_or_create_collection(
                name="enhanced_video_frames",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize CLIP for direct image embeddings
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                logger.info(f"CLIP model loaded on {device}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
            
            # Initialize text embedding model
            try:
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Text embedding model loaded")
            except Exception as e:
                logger.warning(f"Failed to load text model: {e}")
            
            self.available = True
            logger.info("Enhanced Vector Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Vector Service: {e}")
            self.available = False
    
    def _generate_image_embedding(self, image_path: str) -> Optional[List[float]]:
        """Generate direct image embedding using CLIP"""
        if not self.clip_model:
            return None
            
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten().tolist()
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None
    
    def _generate_text_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        if self.text_model:
            return self.text_model.encode(text).tolist()
        else:
            # Fallback to simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(32, len(hash_hex)), 2)]
            while len(embedding) < 16:
                embedding.append(0.0)
            return embedding[:16]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for hybrid search"""
        # Simple keyword extraction
        text = text.lower()
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def _build_keyword_index(self, frame_id: str, description: str):
        """Build keyword index for hybrid search"""
        keywords = self._extract_keywords(description)
        
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(frame_id)
    
    async def add_enhanced_frame_embedding(self,
                                         frame_id: str,
                                         description: str,
                                         image_path: str,
                                         metadata: Dict) -> bool:
        """Add frame with both image and text embeddings"""
        try:
            # Generate both image and text embeddings
            image_embedding = self._generate_image_embedding(image_path)
            text_embedding = self._generate_text_embedding(description)
            
            # Combine embeddings (concatenate or use primary embedding)
            if image_embedding and text_embedding:
                # Use image embedding as primary, store text embedding in metadata
                primary_embedding = image_embedding
                enhanced_metadata = metadata.copy()
                enhanced_metadata['text_embedding'] = text_embedding
                enhanced_metadata['has_image_embedding'] = True
            elif text_embedding:
                # Fallback to text embedding only
                primary_embedding = text_embedding
                enhanced_metadata = metadata.copy()
                enhanced_metadata['has_image_embedding'] = False
            else:
                logger.error(f"Failed to generate any embedding for frame {frame_id}")
                return False
            
            # Store keywords for hybrid search
            enhanced_metadata['keywords'] = self._extract_keywords(description)
            
            # Add to ChromaDB
            self.frames_collection.add(
                embeddings=[primary_embedding],
                documents=[description],
                metadatas=[enhanced_metadata],
                ids=[frame_id]
            )
            
            # Build keyword index
            self._build_keyword_index(frame_id, description)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding enhanced frame embedding: {e}")
            return False
    
    async def hybrid_search(self,
                          query: str,
                          video_id: Optional[int] = None,
                          limit: int = 10,
                          keyword_weight: float = 0.3,
                          vector_weight: float = 0.7) -> List[SearchResult]:
        """Perform hybrid search combining keyword and vector search"""
        try:
            # Step 1: Keyword pre-filtering
            query_keywords = self._extract_keywords(query)
            keyword_candidates = set()
            
            for keyword in query_keywords:
                if keyword in self.keyword_index:
                    keyword_candidates.update(self.keyword_index[keyword])
            
            # Step 2: Vector search
            query_embedding = self._generate_text_embedding(query)
            
            # Prepare where clause for video filtering
            where_clause = {}
            if video_id:
                where_clause['video_id'] = video_id
            
            # Perform vector search
            vector_results = self.frames_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more results for hybrid ranking
                where=where_clause if where_clause else None,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )
            
            # Step 3: Hybrid scoring and ranking
            enhanced_results = []
            
            for i, (frame_id, document, metadata, distance) in enumerate(zip(
                vector_results['ids'][0],
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['distances'][0]
            )):
                # Calculate vector similarity (convert distance to similarity)
                vector_similarity = max(0, 1 - distance)
                
                # Calculate keyword similarity
                frame_keywords = set(metadata.get('keywords', []))
                query_keywords_set = set(query_keywords)
                
                if query_keywords_set and frame_keywords:
                    keyword_similarity = len(query_keywords_set.intersection(frame_keywords)) / len(query_keywords_set.union(frame_keywords))
                else:
                    keyword_similarity = 0
                
                # Calculate hybrid score
                hybrid_score = (vector_weight * vector_similarity) + (keyword_weight * keyword_similarity)
                
                # Bonus for keyword pre-filtering matches
                if frame_id in keyword_candidates:
                    hybrid_score += 0.1
                
                enhanced_results.append(SearchResult(
                    frame_id=frame_id,
                    timestamp=metadata.get('timestamp', 0),
                    confidence=min(100, hybrid_score * 100),
                    description=document,
                    frame_path=metadata.get('frame_path', ''),
                    image_similarity=vector_similarity if metadata.get('has_image_embedding') else 0,
                    text_similarity=vector_similarity,
                    keyword_matches=list(query_keywords_set.intersection(frame_keywords)),
                    metadata=metadata
                ))
            
            # Sort by hybrid score and return top results
            enhanced_results.sort(key=lambda x: x.confidence, reverse=True)
            return enhanced_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        try:
            collection_count = self.frames_collection.count() if self.frames_collection else 0
            keyword_count = len(self.keyword_index)
            
            return {
                "available": self.available,
                "total_frames": collection_count,
                "keyword_index_size": keyword_count,
                "has_clip_model": self.clip_model is not None,
                "has_text_model": self.text_model is not None
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"available": False, "error": str(e)}
