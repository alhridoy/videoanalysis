"""
Dual-Embedding Service with MiniLM text + CLIP image weighted fusion
Implements production-ready retrieval accuracy improvements
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
from dataclasses import dataclass
import cv2
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class DualEmbeddingResult:
    """Result with dual embeddings and weighted fusion"""
    frame_id: str
    timestamp: float
    confidence: float
    description: str
    frame_path: str
    text_similarity: float
    image_similarity: float
    fusion_score: float
    scene_change_score: float
    is_keyframe: bool
    metadata: Dict[str, Any]

class DualEmbeddingService:
    """Production-ready dual embedding service with weighted fusion"""
    
    def __init__(self):
        self.client = None
        self.text_collection = None
        self.image_collection = None
        self.fusion_collection = None
        self.available = False
        
        # Dual embedding models
        self.clip_model = None
        self.clip_preprocess = None
        self.text_model = None
        
        # Scene change detection
        self.scene_threshold = 0.3  # Threshold for scene change detection
        self.keyframe_percentage = 0.1  # Select ~10% of frames as keyframes
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize ChromaDB and dual embedding models"""
        try:
            # Initialize ChromaDB with separate collections
            self.client = chromadb.PersistentClient(
                path="./chroma_db_dual",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create separate collections for different embedding types
            self.text_collection = self.client.get_or_create_collection(
                name="text_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.image_collection = self.client.get_or_create_collection(
                name="image_embeddings", 
                metadata={"hnsw:space": "cosine"}
            )
            
            self.fusion_collection = self.client.get_or_create_collection(
                name="fusion_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize CLIP for image embeddings
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                logger.info(f"CLIP model loaded on {device}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
            
            # Initialize MiniLM for text embeddings
            try:
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("MiniLM text model loaded")
            except Exception as e:
                logger.warning(f"Failed to load MiniLM model: {e}")
            
            self.available = True
            logger.info("Dual Embedding Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Dual Embedding Service: {e}")
            self.available = False
    
    def _generate_clip_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate CLIP image embedding"""
        if not self.clip_model:
            return None
            
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error generating CLIP embedding: {e}")
            return None
    
    def _generate_minilm_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate MiniLM text embedding"""
        if not self.text_model:
            return None
            
        try:
            embedding = self.text_model.encode(text)
            return embedding / np.linalg.norm(embedding)  # Normalize
        except Exception as e:
            logger.error(f"Error generating MiniLM embedding: {e}")
            return None
    
    def _create_weighted_fusion(self, text_embedding: np.ndarray, 
                               image_embedding: np.ndarray,
                               text_weight: float = 0.6,
                               image_weight: float = 0.4) -> np.ndarray:
        """Create weighted fusion of text and image embeddings"""
        try:
            # Ensure embeddings are same dimension through projection or padding
            if text_embedding.shape[0] != image_embedding.shape[0]:
                # Project to common dimension (use smaller dimension)
                target_dim = min(text_embedding.shape[0], image_embedding.shape[0])
                text_embedding = text_embedding[:target_dim]
                image_embedding = image_embedding[:target_dim]
            
            # Weighted fusion
            fusion_embedding = (text_weight * text_embedding + 
                              image_weight * image_embedding)
            
            # Normalize the fusion
            fusion_embedding = fusion_embedding / np.linalg.norm(fusion_embedding)
            
            return fusion_embedding
            
        except Exception as e:
            logger.error(f"Error creating weighted fusion: {e}")
            return text_embedding  # Fallback to text embedding
    
    def _detect_scene_change(self, current_frame_path: str, 
                           previous_frame_path: Optional[str]) -> float:
        """Detect scene change between consecutive frames"""
        if not previous_frame_path:
            return 1.0  # First frame is always a scene change
        
        try:
            # Load frames
            current_frame = cv2.imread(current_frame_path)
            previous_frame = cv2.imread(previous_frame_path)
            
            if current_frame is None or previous_frame is None:
                return 0.5  # Default moderate score
            
            # Resize for faster processing
            current_frame = cv2.resize(current_frame, (224, 224))
            previous_frame = cv2.resize(previous_frame, (224, 224))
            
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram difference
            hist_current = cv2.calcHist([current_gray], [0], None, [256], [0, 256])
            hist_previous = cv2.calcHist([previous_gray], [0], None, [256], [0, 256])
            
            # Normalize histograms
            hist_current = hist_current / hist_current.sum()
            hist_previous = hist_previous / hist_previous.sum()
            
            # Calculate correlation (higher = more similar)
            correlation = cv2.compareHist(hist_current, hist_previous, cv2.HISTCMP_CORREL)
            
            # Convert to scene change score (lower correlation = higher scene change)
            scene_change_score = 1.0 - correlation
            
            return max(0.0, min(1.0, scene_change_score))
            
        except Exception as e:
            logger.error(f"Error detecting scene change: {e}")
            return 0.5
    
    def _select_keyframes(self, frames_data: List[Dict]) -> List[Dict]:
        """Select ~10% of frames as keyframes based on scene changes"""
        if not frames_data:
            return []
        
        # Calculate scene change scores
        for i, frame_data in enumerate(frames_data):
            if i == 0:
                frame_data['scene_change_score'] = 1.0
                frame_data['is_keyframe'] = True
            else:
                scene_score = self._detect_scene_change(
                    frame_data['frame_path'],
                    frames_data[i-1]['frame_path']
                )
                frame_data['scene_change_score'] = scene_score
        
        # Sort by scene change score and select top 10%
        sorted_frames = sorted(frames_data, key=lambda x: x['scene_change_score'], reverse=True)
        keyframe_count = max(1, int(len(frames_data) * self.keyframe_percentage))
        
        # Mark keyframes
        keyframe_ids = set()
        for i in range(keyframe_count):
            sorted_frames[i]['is_keyframe'] = True
            keyframe_ids.add(sorted_frames[i]['frame_id'])
        
        # Mark non-keyframes
        for frame_data in frames_data:
            if frame_data['frame_id'] not in keyframe_ids:
                frame_data['is_keyframe'] = False
        
        return [f for f in frames_data if f['is_keyframe']]
    
    async def add_dual_embedding(self, frame_id: str, description: str, 
                               image_path: str, metadata: Dict,
                               text_weight: float = 0.6,
                               image_weight: float = 0.4) -> bool:
        """Add frame with dual embeddings and weighted fusion"""
        try:
            # Generate both embeddings
            text_embedding = self._generate_minilm_embedding(description)
            image_embedding = self._generate_clip_embedding(image_path)
            
            if text_embedding is None and image_embedding is None:
                logger.error(f"Failed to generate any embedding for frame {frame_id}")
                return False
            
            # Handle missing embeddings
            if text_embedding is None:
                text_embedding = np.zeros_like(image_embedding)
                text_weight = 0.0
                image_weight = 1.0
            elif image_embedding is None:
                image_embedding = np.zeros_like(text_embedding)
                text_weight = 1.0
                image_weight = 0.0
            
            # Create weighted fusion
            fusion_embedding = self._create_weighted_fusion(
                text_embedding, image_embedding, text_weight, image_weight
            )
            
            # Enhanced metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                'text_weight': text_weight,
                'image_weight': image_weight,
                'has_text_embedding': text_embedding is not None,
                'has_image_embedding': image_embedding is not None,
                'embedding_type': 'dual_fusion'
            })
            
            # Store in separate collections
            self.text_collection.add(
                embeddings=[text_embedding.tolist()],
                documents=[description],
                metadatas=[enhanced_metadata],
                ids=[f"{frame_id}_text"]
            )
            
            self.image_collection.add(
                embeddings=[image_embedding.tolist()],
                documents=[description],
                metadatas=[enhanced_metadata],
                ids=[f"{frame_id}_image"]
            )
            
            self.fusion_collection.add(
                embeddings=[fusion_embedding.tolist()],
                documents=[description],
                metadatas=[enhanced_metadata],
                ids=[frame_id]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding dual embedding: {e}")
            return False
    
    async def dual_search(self, query: str, video_id: Optional[int] = None,
                         limit: int = 10, text_weight: float = 0.6,
                         image_weight: float = 0.4) -> List[DualEmbeddingResult]:
        """Perform search with dual embeddings and weighted fusion"""
        try:
            # Generate query embeddings
            query_text_embedding = self._generate_minilm_embedding(query)
            
            # For image queries, we could generate CLIP text embedding
            # CLIP can encode text queries for image search
            query_image_embedding = None
            if self.clip_model:
                try:
                    text_tokens = clip.tokenize([query])
                    with torch.no_grad():
                        query_image_embedding = self.clip_model.encode_text(text_tokens)
                        query_image_embedding = query_image_embedding / query_image_embedding.norm(dim=-1, keepdim=True)
                        query_image_embedding = query_image_embedding.cpu().numpy().flatten()
                except Exception as e:
                    logger.warning(f"Failed to generate CLIP text embedding: {e}")
            
            if query_text_embedding is None and query_image_embedding is None:
                return []
            
            # Create fusion query embedding
            if query_text_embedding is not None and query_image_embedding is not None:
                query_fusion = self._create_weighted_fusion(
                    query_text_embedding, query_image_embedding, text_weight, image_weight
                )
            elif query_text_embedding is not None:
                query_fusion = query_text_embedding
            else:
                query_fusion = query_image_embedding
            
            # Prepare where clause
            where_clause = {}
            if video_id:
                where_clause['video_id'] = video_id
            
            # Search fusion collection
            fusion_results = self.fusion_collection.query(
                query_embeddings=[query_fusion.tolist()],
                n_results=limit,
                where=where_clause if where_clause else None,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )
            
            # Convert to DualEmbeddingResult
            results = []
            for i, (frame_id, document, metadata, distance) in enumerate(zip(
                fusion_results['ids'][0],
                fusion_results['documents'][0], 
                fusion_results['metadatas'][0],
                fusion_results['distances'][0]
            )):
                # Calculate similarities
                fusion_similarity = max(0, 1 - distance)
                
                # Get individual similarities if available
                text_similarity = metadata.get('text_similarity', fusion_similarity)
                image_similarity = metadata.get('image_similarity', fusion_similarity)
                
                results.append(DualEmbeddingResult(
                    frame_id=frame_id,
                    timestamp=metadata.get('timestamp', 0),
                    confidence=min(100, fusion_similarity * 100),
                    description=document,
                    frame_path=metadata.get('frame_path', ''),
                    text_similarity=text_similarity,
                    image_similarity=image_similarity,
                    fusion_score=fusion_similarity,
                    scene_change_score=metadata.get('scene_change_score', 0.5),
                    is_keyframe=metadata.get('is_keyframe', False),
                    metadata=metadata
                ))
            
            # Sort by fusion score
            results.sort(key=lambda x: x.fusion_score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error in dual search: {e}")
            return []
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        try:
            text_count = self.text_collection.count() if self.text_collection else 0
            image_count = self.image_collection.count() if self.image_collection else 0
            fusion_count = self.fusion_collection.count() if self.fusion_collection else 0
            
            return {
                "available": self.available,
                "text_embeddings": text_count,
                "image_embeddings": image_count,
                "fusion_embeddings": fusion_count,
                "has_clip_model": self.clip_model is not None,
                "has_minilm_model": self.text_model is not None,
                "keyframe_percentage": self.keyframe_percentage,
                "scene_threshold": self.scene_threshold,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"available": False, "error": str(e)}
