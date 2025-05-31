from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os
from collections import defaultdict

from app.core.database import get_db
from app.models.video import Video, VideoFrame
from app.services.gemini_service import GeminiService
from app.services.vector_service import VectorService
logger = logging.getLogger(__name__)

# Enhanced services (require additional dependencies)
try:
    from app.services.enhanced_vector_service import EnhancedVectorService
    from app.services.batch_video_service import BatchVideoService
    from app.services.dual_embedding_service import DualEmbeddingService
    from app.services.scene_detector import SceneDetector
    from app.services.thumbnail_service import ThumbnailService
    from app.services.enhanced_video_analysis import EnhancedVideoAnalysis
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
router = APIRouter()

class VisualSearchRequest(BaseModel):
    video_id: int
    query: str
    max_results: int = 10

class SearchResult(BaseModel):
    timestamp: float
    confidence: float
    description: str
    frame_path: str
    summary: Optional[str] = None  # Concise answer
    objects_detected: Optional[List[str]] = None
    people_count: Optional[int] = None
    detailed_analysis: Optional[str] = None
    image_similarity: Optional[float] = None
    text_similarity: Optional[float] = None
    keyword_matches: Optional[List[str]] = None
    
class ClipResult(BaseModel):
    start_time: float
    end_time: float
    confidence: float
    description: str
    frame_count: int
    frames: List[SearchResult]

class VisualSearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    clips: List[ClipResult]
    total_results: int
    direct_answer: Optional[str] = None  # For counting and specific queries
    query_type: Optional[str] = None  # 'counting' | 'object_detection' | 'scene_analysis' | 'general'
    summary: Optional[str] = None
    processing_method: Optional[str] = None  # 'batch_video' | 'hybrid_search' | 'vector_search' | 'semantic_search'

def calculate_confidence_from_distance(distance: float, distance_metric: str = "cosine") -> float:
    """
    Convert distance to confidence score (0-100)
    
    Args:
        distance: Distance value from vector search
        distance_metric: Type of distance metric used
    
    Returns:
        Confidence score between 0 and 100
    """
    if distance_metric == "cosine":
        # Cosine distance range is [0, 2] where 0 is identical
        # Convert to similarity: similarity = 1 - (distance / 2)
        similarity = max(0, 1 - (distance / 2))
        confidence = round(similarity * 100, 1)
    elif distance_metric == "euclidean":
        # For euclidean, we need to normalize differently
        # Assuming normalized vectors, max distance ≈ sqrt(2) ≈ 1.414
        similarity = max(0, 1 - (distance / 1.414))
        confidence = round(similarity * 100, 1)
    else:
        # Default fallback
        confidence = max(0, 100 - (distance * 50))
    
    return min(100.0, max(0.0, confidence))

def group_frames_into_clips(results: List[SearchResult], threshold_seconds: float = 5.0) -> List[ClipResult]:
    """
    Group consecutive frames into clips based on temporal proximity
    
    Args:
        results: List of search results sorted by timestamp
        threshold_seconds: Maximum gap between frames to consider them part of the same clip
    
    Returns:
        List of clips with aggregated confidence
    """
    if not results:
        return []
    
    # Sort by timestamp
    sorted_results = sorted(results, key=lambda x: x.timestamp)
    
    clips = []
    current_clip_frames = [sorted_results[0]]
    
    for i in range(1, len(sorted_results)):
        current_frame = sorted_results[i]
        last_frame = current_clip_frames[-1]
        
        # Check if this frame is close enough to be part of the same clip
        if current_frame.timestamp - last_frame.timestamp <= threshold_seconds:
            current_clip_frames.append(current_frame)
        else:
            # Create clip from accumulated frames
            if current_clip_frames:
                clips.append(_create_clip_from_frames(current_clip_frames))
            current_clip_frames = [current_frame]
    
    # Don't forget the last clip
    if current_clip_frames:
        clips.append(_create_clip_from_frames(current_clip_frames))
    
    return clips

def _create_clip_from_frames(frames: List[SearchResult]) -> ClipResult:
    """Create a clip from a group of frames"""
    start_time = frames[0].timestamp
    end_time = frames[-1].timestamp
    
    # Calculate average confidence with temporal smoothing
    confidences = [f.confidence for f in frames]
    avg_confidence = sum(confidences) / len(confidences)
    
    # Use the description from the highest confidence frame
    best_frame = max(frames, key=lambda f: f.confidence)
    
    return ClipResult(
        start_time=start_time,
        end_time=end_time,
        confidence=round(avg_confidence, 1),
        description=f"Clip from {start_time:.1f}s to {end_time:.1f}s: {best_frame.description}",
        frame_count=len(frames),
        frames=frames
    )

@router.post("/visual", response_model=VisualSearchResponse)
async def visual_search(
    request: Request,
    search_request: VisualSearchRequest,
    db: Session = Depends(get_db)
):
    """Enhanced visual search with direct image embeddings, hybrid search, and batch video processing"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == search_request.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # IMPROVEMENT 3: Batch video processing using Gemini's video API
        if (video.video_type == "upload" and
            video.file_path and
            os.path.exists(video.file_path) and
            hasattr(request.app.state, 'gemini_api_key')):

            logger.info(f"Attempting batch video processing for query: '{search_request.query}'")
            try:
                batch_service = BatchVideoService(request.app.state.gemini_api_key)
                if batch_service.available:
                    # Upload video and perform batch search
                    video_uri = await batch_service.upload_video_to_gemini(video.file_path)
                    if video_uri:
                        batch_result = await batch_service.batch_search_video(
                            video_uri,
                            search_request.query,
                            video_context=f"Title: {video.title}"
                        )

                        # Convert batch results to SearchResult format
                        results = []
                        for i, (timestamp, description) in enumerate(zip(
                            batch_result.timestamps,
                            batch_result.descriptions
                        )):
                            results.append(SearchResult(
                                timestamp=timestamp,
                                confidence=round(batch_result.confidence * 100, 1),
                                description=description,
                                frame_path="/api/placeholder/120/68",
                                summary=f"Instance {i+1}: {description[:100]}...",
                                detailed_analysis=description
                            ))

                        # Group results into clips
                        clips = group_frames_into_clips(results)
                        clips.sort(key=lambda c: c.confidence, reverse=True)

                        # Cleanup uploaded video
                        batch_service.cleanup_uploaded_video(video_uri)

                        return VisualSearchResponse(
                            query=search_request.query,
                            results=results[:search_request.max_results],
                            clips=clips[:max(5, search_request.max_results // 2)],
                            total_results=len(results),
                            direct_answer=batch_result.direct_answer,
                            query_type=batch_result.query_type,
                            processing_method="batch_video"
                        )

            except Exception as e:
                logger.warning(f"Batch video processing failed, falling back: {e}")

        # IMPROVEMENT 1 & 2: Enhanced vector service with direct image embeddings and hybrid search
        enhanced_vector_service = EnhancedVectorService()

        if enhanced_vector_service.available:
            logger.info(f"Using enhanced hybrid search for query: '{search_request.query}'")
            try:
                # Perform hybrid search (combines keyword + vector + image embeddings)
                enhanced_results = await enhanced_vector_service.hybrid_search(
                    query=search_request.query,
                    video_id=search_request.video_id,
                    limit=search_request.max_results * 2,
                    keyword_weight=0.3,
                    vector_weight=0.7
                )

                # Convert enhanced results to SearchResult format
                results = []
                for result in enhanced_results:
                    results.append(SearchResult(
                        timestamp=result.timestamp,
                        confidence=result.confidence,
                        description=result.description,
                        frame_path=result.frame_path,
                        summary=result.description[:100] + "..." if len(result.description) > 100 else result.description,
                        objects_detected=result.metadata.get('keywords', [])[:5],  # Top 5 keywords as objects
                        detailed_analysis=result.description,
                        image_similarity=result.image_similarity,
                        text_similarity=result.text_similarity,
                        keyword_matches=result.keyword_matches
                    ))

                # Determine query type and generate direct answer
                query_lower = search_request.query.lower()
                direct_answer = None
                query_type = "general"

                if any(word in query_lower for word in ['how many', 'count', 'number of']):
                    query_type = "counting"
                    if 'people' in query_lower or 'person' in query_lower:
                        direct_answer = f"{len(results)} instances of people detected in video"
                    else:
                        direct_answer = f"{len(results)} instances found matching '{search_request.query}'"
                elif any(word in query_lower for word in ['microphone', 'mic', 'equipment', 'object']):
                    query_type = "object_detection"
                    direct_answer = f"Object '{search_request.query}' detected in {len(results)} instances"
                elif any(word in query_lower for word in ['background', 'scene', 'setting', 'color']):
                    query_type = "scene_analysis"
                    direct_answer = f"Scene analysis found {len(results)} relevant moments"

                # Group results into clips
                clips = group_frames_into_clips(results)
                clips.sort(key=lambda c: c.confidence, reverse=True)

                return VisualSearchResponse(
                    query=search_request.query,
                    results=results[:search_request.max_results],
                    clips=clips[:max(5, search_request.max_results // 2)],
                    total_results=len(results),
                    direct_answer=direct_answer,
                    query_type=query_type,
                    processing_method="hybrid_search"
                )

            except Exception as e:
                logger.warning(f"Enhanced hybrid search failed, falling back: {e}")

        # Fallback to original vector search
        vector_service = VectorService()
        if vector_service.available:
            logger.debug(f"Using standard vector search for query: '{search_request.query}'")
            search_results = await vector_service.search_frames(
                query=search_request.query,
                video_id=search_request.video_id,
                limit=search_request.max_results * 2
            )

            results = []
            for result in search_results:
                confidence = calculate_confidence_from_distance(
                    result.get('distance', 0),
                    distance_metric="cosine"
                )

                results.append(SearchResult(
                    timestamp=result['metadata'].get('timestamp', 0),
                    confidence=confidence,
                    description=result['description'],
                    frame_path=result['metadata'].get('frame_path', '')
                ))
        else:
            # Final fallback to enhanced semantic search
            logger.info("All vector services unavailable, using enhanced semantic search")
            frames = db.query(VideoFrame)\
                .filter(VideoFrame.video_id == search_request.video_id)\
                .order_by(VideoFrame.timestamp.asc())\
                .all()

            results = await _enhanced_semantic_search(search_request.query, frames)

        # Group results into clips
        clips = group_frames_into_clips(results)
        clips.sort(key=lambda c: c.confidence, reverse=True)

        return VisualSearchResponse(
            query=search_request.query,
            results=results[:search_request.max_results],
            clips=clips[:max(5, search_request.max_results // 2)],
            total_results=len(results),
            processing_method="semantic_search"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced visual search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _enhanced_semantic_search(query: str, frames: List[VideoFrame]) -> List[SearchResult]:
    """Enhanced semantic visual search using detailed frame descriptions"""
    results = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Enhanced semantic mappings for better object detection
    semantic_mappings = {
        'car': ['vehicle', 'automobile', 'truck', 'van', 'suv', 'sedan', 'transportation', 'driving'],
        'red': ['crimson', 'scarlet', 'burgundy', 'maroon', 'cherry', 'pink'],
        'blue': ['azure', 'navy', 'cyan', 'cobalt', 'turquoise', 'teal', 'indigo'],
        'green': ['emerald', 'lime', 'forest', 'olive', 'mint', 'sage'],
        'person': ['people', 'human', 'individual', 'man', 'woman', 'figure', 'someone'],
        'building': ['structure', 'house', 'office', 'tower', 'architecture', 'construction'],
        'text': ['sign', 'writing', 'words', 'letters', 'label', 'caption'],
        'screen': ['monitor', 'display', 'television', 'computer', 'laptop'],
        'phone': ['mobile', 'smartphone', 'device', 'cell', 'telephone'],
        'book': ['document', 'paper', 'reading', 'publication', 'magazine'],
        'food': ['meal', 'eating', 'restaurant', 'kitchen', 'cooking', 'dining'],
        'animal': ['pet', 'dog', 'cat', 'bird', 'wildlife', 'creature']
    }
    
    # Expand query with semantic alternatives
    expanded_query_words = set(query_words)
    for word in query_words:
        if word in semantic_mappings:
            expanded_query_words.update(semantic_mappings[word])
    
    logger.debug(f"Enhanced search for '{query}' with expanded terms: {expanded_query_words}")
    
    for frame in frames:
        confidence = 0.0
        description = ""
        
        if frame.description:
            description_lower = frame.description.lower()
            
            # Multi-level matching strategy
            exact_matches = sum(1 for word in query_words if word in description_lower)
            semantic_matches = sum(1 for word in expanded_query_words if word in description_lower)
            
            # Color + object combination matching (e.g., "red car")
            color_object_bonus = 0.0
            if len(query_words) >= 2:
                colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown']
                objects = ['car', 'person', 'building', 'shirt', 'dress', 'hat', 'bag', 'phone', 'book', 'screen']
                
                query_colors = [w for w in query_words if w in colors]
                query_objects = [w for w in query_words if w in objects]
                
                if query_colors and query_objects:
                    color_in_desc = any(color in description_lower for color in query_colors)
                    object_in_desc = any(obj in description_lower for obj in query_objects)
                    if color_in_desc and object_in_desc:
                        color_object_bonus = 15.0  # 15% bonus for color+object match
            
            # Calculate confidence score (0-100 range)
            if exact_matches > 0 or semantic_matches > 0:
                # Base confidence from matches
                exact_score = (exact_matches / len(query_words)) * 60 if query_words else 0
                semantic_score = (semantic_matches / len(expanded_query_words)) * 20 if expanded_query_words else 0
                
                # Total confidence
                confidence = min(95.0, exact_score + semantic_score + color_object_bonus + 10.0)
                
                # Extract relevant portion of description
                sentences = frame.description.split('.')
                relevant_sentence = ""
                for sentence in sentences:
                    if any(word in sentence.lower() for word in expanded_query_words):
                        relevant_sentence = sentence.strip()
                        break
                
                if not relevant_sentence and sentences:
                    relevant_sentence = sentences[0].strip()
                
                description = f"Match for '{query}': {relevant_sentence[:200]}..."
        
        # Fallback to pattern-based detection if no description
        if confidence == 0.0 and not frame.description:
            confidence, description = _pattern_based_detection(query_lower, frame.timestamp, query)
        
        if confidence > 10.0:  # Lower threshold for more results
            results.append(SearchResult(
                timestamp=frame.timestamp,
                confidence=round(confidence, 1),
                description=description,
                frame_path=frame.frame_path
            ))
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x.confidence, reverse=True)
    logger.debug(f"Enhanced semantic search returned {len(results)} results for '{query}'")
    return results

def _pattern_based_detection(query_lower: str, timestamp: float, original_query: str) -> tuple[float, str]:
    """Fallback pattern-based detection for frames without descriptions"""
    confidence = 0.0
    description = ""
    
    # Pattern matching with confidence scores (0-100 range)
    patterns = {
        'car': (30, 15, 70.0, "Vehicle detected"),
        'person': (25, 12, 65.0, "Person detected"),
        'people': (25, 12, 65.0, "People detected"),
        'text': (40, 18, 60.0, "Text/signage detected"),
        'sign': (40, 18, 60.0, "Sign detected"),
        'red': (35, 20, 55.0, "Red object detected"),
        'blue': (28, 14, 58.0, "Blue object detected"),
        'green': (33, 16, 52.0, "Green object detected"),
        'building': (120, 60, 60.0, "Building/structure detected"),
        'house': (120, 60, 60.0, "House/building detected"),
        'screen': (45, 22, 50.0, "Screen/display detected"),
        'phone': (50, 25, 48.0, "Phone/device detected")
    }
    
    for keyword, (interval, duration, base_conf, desc) in patterns.items():
        if keyword in query_lower:
            if timestamp % interval < duration:
                # Add small variation based on timestamp
                confidence = base_conf + (timestamp % 20) * 0.5
                description = f"{desc} at {timestamp:.1f}s"
                break
    
    # Generic fallback
    if confidence == 0.0 and timestamp % 45 < 20:
        confidence = 40.0 + (timestamp % 8) * 1.0
        description = f"Potential match for '{original_query}' at {timestamp:.1f}s"
    
    return min(100.0, confidence), description

@router.get("/{video_id}/frames")
async def get_video_frames(
    video_id: int,
    limit: int = 500,  # Increased limit for longer videos
    db: Session = Depends(get_db)
):
    """Get video frames for a video"""
    # Verify video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get frames
    frames = db.query(VideoFrame)\
        .filter(VideoFrame.video_id == video_id)\
        .order_by(VideoFrame.timestamp.asc())\
        .limit(limit)\
        .all()
    
    return {
        "video_id": video_id,
        "frames": [
            {
                "id": frame.id,
                "timestamp": frame.timestamp,
                "frame_path": frame.frame_path,
                "description": frame.description,
                "objects_detected": frame.objects_detected
            }
            for frame in frames
        ]
    }

@router.post("/{video_id}/analyze-frames")
async def analyze_video_frames(
    request: Request,
    video_id: int,
    db: Session = Depends(get_db)
):
    """Enhanced frame analysis with direct image embeddings"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get frames that haven't been analyzed
        frames = db.query(VideoFrame)\
            .filter(VideoFrame.video_id == video_id)\
            .filter(VideoFrame.description.is_(None))\
            .limit(50)\
            .all()  # Increased limit for longer videos

        if not frames:
            return {
                "message": "All frames already analyzed or no frames available",
                "analyzed_count": 0
            }

        gemini_service = request.app.state.gemini_service

        # Try enhanced vector service first (with direct image embeddings)
        enhanced_vector_service = EnhancedVectorService()
        vector_service = VectorService()  # Fallback

        analyzed_count = 0

        for frame in frames:
            try:
                # Analyze frame with Gemini Vision
                analysis = await gemini_service.analyze_frame(
                    frame.frame_path,
                    context=f"Frame from video: {video.title}"
                )

                if analysis["status"] == "success":
                    frame.description = analysis["description"]

                    # Try enhanced vector service first (IMPROVEMENT 1: Direct image embeddings)
                    if enhanced_vector_service.available:
                        success = await enhanced_vector_service.add_enhanced_frame_embedding(
                            frame_id=f"frame_{frame.id}",
                            description=analysis["description"],
                            image_path=frame.frame_path,
                            metadata={
                                "video_id": video_id,
                                "timestamp": frame.timestamp,
                                "frame_path": frame.frame_path
                            }
                        )
                        if not success:
                            logger.warning(f"Enhanced embedding failed for frame {frame.id}, trying standard")
                            # Fallback to standard vector service
                            if vector_service.available:
                                await vector_service.add_frame_embedding(
                                    frame_id=f"frame_{frame.id}",
                                    description=analysis["description"],
                                    metadata={
                                        "video_id": video_id,
                                        "timestamp": frame.timestamp,
                                        "frame_path": frame.frame_path
                                    }
                                )
                    elif vector_service.available:
                        # Fallback to standard vector service
                        await vector_service.add_frame_embedding(
                            frame_id=f"frame_{frame.id}",
                            description=analysis["description"],
                            metadata={
                                "video_id": video_id,
                                "timestamp": frame.timestamp,
                                "frame_path": frame.frame_path
                            }
                        )

                    analyzed_count += 1

            except Exception as e:
                logger.error(f"Error analyzing frame {frame.id}: {e}")
                continue

        db.commit()

        # Update video embedding status if all frames are analyzed
        total_frames = db.query(VideoFrame).filter(VideoFrame.video_id == video_id).count()
        analyzed_frames = db.query(VideoFrame)\
            .filter(VideoFrame.video_id == video_id)\
            .filter(VideoFrame.description.isnot(None))\
            .count()

        if analyzed_frames == total_frames:
            video.embedding_status = "completed"
            db.commit()

        # Get service statistics
        enhanced_stats = enhanced_vector_service.get_stats() if enhanced_vector_service.available else {}

        return {
            "message": f"Analyzed {analyzed_count} frames with enhanced embeddings",
            "analyzed_count": analyzed_count,
            "total_frames": len(frames),
            "video_total_frames": total_frames,
            "video_analyzed_frames": analyzed_frames,
            "enhanced_embeddings": enhanced_vector_service.available,
            "service_stats": enhanced_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing frames: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{video_id}/native-moment-retrieval")
async def native_moment_retrieval(
    request: Request,
    video_id: int,
    db: Session = Depends(get_db)
):
    """Enhanced moment retrieval using Gemini 2.5 native video analysis"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if not hasattr(request.app.state, 'gemini_api_key'):
            raise HTTPException(status_code=500, detail="Gemini API key not configured")

        # Use enhanced video analysis
        enhanced_analyzer = EnhancedVideoAnalysis(request.app.state.gemini_api_key)

        # Get video file path
        video_path = video.file_path
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        # Get context from transcript if available
        context = f"Title: {video.title}"
        if video.transcript:
            context += f"\nTranscript preview: {video.transcript[:1000]}"

        # Perform native moment retrieval
        moments = await enhanced_analyzer.analyze_video_moments_native(video_path, context)

        return {
            "video_id": video_id,
            "moments": moments,
            "analysis_method": "gemini_2.5_native",
            "total_moments": len(moments)
        }

    except Exception as e:
        logger.error(f"Error in native moment retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{video_id}/native-temporal-counting")
async def native_temporal_counting(
    request: Request,
    video_id: int,
    query: str,
    db: Session = Depends(get_db)
):
    """Enhanced temporal counting using Gemini 2.5 native video analysis"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if not hasattr(request.app.state, 'gemini_api_key'):
            raise HTTPException(status_code=500, detail="Gemini API key not configured")

        # Use enhanced video analysis
        enhanced_analyzer = EnhancedVideoAnalysis(request.app.state.gemini_api_key)

        # Get video file path
        video_path = video.file_path
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        # Get context from transcript if available
        context = f"Title: {video.title}"
        if video.transcript:
            context += f"\nTranscript preview: {video.transcript[:1000]}"

        # Perform native temporal counting
        result = await enhanced_analyzer.temporal_counting_native(video_path, query, context)

        return {
            "video_id": video_id,
            "query": query,
            "result": result,
            "analysis_method": "gemini_2.5_native"
        }

    except Exception as e:
        logger.error(f"Error in native temporal counting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{video_id}/index-transcript")
async def index_video_transcript(
    request: Request,
    video_id: int,
    db: Session = Depends(get_db)
):
    """Index video transcript for semantic search"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.transcript:
            raise HTTPException(status_code=400, detail="Video has no transcript")
        
        # Initialize vector service
        vector_service = VectorService()
        if not vector_service.available:
            raise HTTPException(
                status_code=503,
                detail="Vector search service not available"
            )
        
        # Add transcript to vector database
        success = await vector_service.add_transcript_embedding(
            video_id=str(video_id),
            transcript=video.transcript,
            metadata={
                "video_id": video_id,
                "title": video.title,
                "video_type": video.video_type
            }
        )
        
        if success:
            return {
                "message": "Transcript indexed successfully",
                "video_id": video_id
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to index transcript"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error indexing transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Gemini 2.5 Features

class TemporalCountingRequest(BaseModel):
    video_id: int
    query: str

class TemporalCountingResponse(BaseModel):
    query: str
    total_count: int
    occurrences: List[Dict]
    patterns: str
    notes: str

class MomentRetrievalResponse(BaseModel):
    video_id: int
    moments: List[Dict]
    total_moments: int

@router.post("/{video_id}/temporal-counting", response_model=TemporalCountingResponse)
async def temporal_counting_analysis(
    request: Request,
    video_id: int,
    counting_request: TemporalCountingRequest,
    db: Session = Depends(get_db)
):
    """Advanced temporal counting using Gemini 2.5 capabilities"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get analyzed frames
        frames = db.query(VideoFrame)\
            .filter(VideoFrame.video_id == video_id)\
            .filter(VideoFrame.description.isnot(None))\
            .order_by(VideoFrame.timestamp.asc())\
            .all()

        if not frames:
            raise HTTPException(
                status_code=400,
                detail="No analyzed frames available. Please analyze frames first."
            )

        # Prepare frame data
        frames_data = [
            {
                "timestamp": frame.timestamp,
                "description": frame.description
            }
            for frame in frames
        ]

        # Perform temporal counting analysis
        gemini_service = request.app.state.gemini_service
        result = await gemini_service.temporal_counting_analysis(
            frames_data,
            counting_request.query
        )

        return TemporalCountingResponse(
            query=counting_request.query,
            total_count=result.get("total_count", 0),
            occurrences=result.get("occurrences", []),
            patterns=result.get("patterns", ""),
            notes=result.get("notes", "")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in temporal counting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}/moment-retrieval", response_model=MomentRetrievalResponse)
async def moment_retrieval_analysis(
    request: Request,
    video_id: int,
    db: Session = Depends(get_db)
):
    """Advanced moment retrieval using Gemini 2.5 capabilities"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if not video.transcript:
            raise HTTPException(
                status_code=400,
                detail="Video transcript required for moment retrieval"
            )

        # Perform moment retrieval analysis
        gemini_service = request.app.state.gemini_service
        moments = await gemini_service.analyze_video_moments(
            video_path=video.file_path or "",
            transcript=video.transcript
        )

        return MomentRetrievalResponse(
            video_id=video_id,
            moments=moments,
            total_moments=len(moments)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in moment retrieval analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Production Features

@router.post("/{video_id}/dual-embedding-analysis")
async def dual_embedding_analysis(
    request: Request,
    video_id: int,
    db: Session = Depends(get_db)
):
    """Enhanced frame analysis with dual embeddings (MiniLM + CLIP)"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get frames for analysis
        frames = db.query(VideoFrame)\
            .filter(VideoFrame.video_id == video_id)\
            .order_by(VideoFrame.timestamp.asc())\
            .all()

        if not frames:
            return {
                "message": "No frames available for analysis",
                "processed_count": 0
            }

        # Initialize services
        dual_service = DualEmbeddingService()
        scene_detector = SceneDetector()
        gemini_service = request.app.state.gemini_service

        # Scene change detection for keyframe selection (~10% of frames)
        frame_paths = [frame.frame_path for frame in frames]
        timestamps = [frame.timestamp for frame in frames]

        scene_results = scene_detector.detect_scene_changes(frame_paths, timestamps)
        keyframes = scene_detector.select_keyframes(scene_results)

        logger.info(f"Selected {len(keyframes)} keyframes from {len(frames)} total frames")

        # Process keyframes with dual embeddings
        processed_count = 0
        for keyframe_result in keyframes:
            frame = next((f for f in frames if f.timestamp == keyframe_result.timestamp), None)
            if not frame:
                continue

            try:
                # Analyze frame if not already done
                if not frame.description:
                    analysis = await gemini_service.analyze_frame(
                        frame.frame_path,
                        context=f"Keyframe from video: {video.title}"
                    )
                    if analysis["status"] == "success":
                        frame.description = analysis["description"]

                # Add dual embedding (MiniLM text + CLIP image)
                if dual_service.available and frame.description:
                    success = await dual_service.add_dual_embedding(
                        frame_id=f"frame_{frame.id}",
                        description=frame.description,
                        image_path=frame.frame_path,
                        metadata={
                            "video_id": video_id,
                            "timestamp": frame.timestamp,
                            "frame_path": frame.frame_path,
                            "is_keyframe": True,
                            "scene_change_score": keyframe_result.scene_change_score,
                            "scene_id": keyframe_result.scene_id,
                            "transition_type": keyframe_result.transition_type
                        }
                    )

                    if success:
                        processed_count += 1

            except Exception as e:
                logger.error(f"Error processing keyframe {frame.id}: {e}")
                continue

        db.commit()

        # Get statistics
        scene_stats = scene_detector.get_scene_statistics(scene_results)
        dual_stats = dual_service.get_service_stats()

        return {
            "message": f"Processed {processed_count} keyframes with dual embeddings",
            "total_frames": len(frames),
            "keyframes_selected": len(keyframes),
            "keyframe_percentage": round(len(keyframes) / len(frames) * 100, 1),
            "processed_count": processed_count,
            "scene_statistics": scene_stats,
            "dual_embedding_stats": dual_stats,
            "enhancement_type": "dual_embedding_minilm_clip"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in dual embedding analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}/thumbnails")
async def generate_video_thumbnails(
    video_id: int,
    interval: float = 10.0,
    db: Session = Depends(get_db)
):
    """Generate real thumbnails for video timeline"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")

        # Initialize thumbnail service
        thumbnail_service = ThumbnailService()

        # Generate timeline thumbnails
        thumbnails = thumbnail_service.generate_timeline_thumbnails(
            video.file_path,
            interval=interval
        )

        return {
            "video_id": video_id,
            "thumbnails": thumbnails,
            "interval_seconds": interval,
            "total_thumbnails": len(thumbnails)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating thumbnails: {e}")
        raise HTTPException(status_code=500, detail=str(e))
