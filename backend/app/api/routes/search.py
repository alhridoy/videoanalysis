from fastapi import APIRouter, HTTPException, Depends, Request, Query
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
    from app.services.native_video_search import (
        NativeVideoSearchService,
        search_objects,
        count_occurrences,
        find_color_objects,
        find_text,
        find_scenes
    )
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
    use_native: bool = True,  # New query parameter to control search type
    db: Session = Depends(get_db)
):
    """Enhanced visual search with native video search, direct image embeddings, hybrid search, and batch video processing"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == search_request.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # DIAGNOSTIC LOGGING
        logger.info(f"🔍 VISUAL SEARCH DEBUG for query: '{search_request.query}'")
        logger.info(f"Video ID: {video.id}, Title: {video.title}")
        logger.info(f"Video Type: {video.video_type}")
        logger.info(f"File Path: {video.file_path}")
        logger.info(f"File exists: {os.path.exists(video.file_path) if video.file_path else False}")
        logger.info(f"Enhanced features available: {ENHANCED_FEATURES_AVAILABLE}")
        logger.info(f"Use native: {use_native}")

        # DIRECT VISUAL SEARCH - Simple implementation that works (try this first)
        if video.file_path and os.path.exists(video.file_path):
            logger.info(f"🎯 Trying direct visual search for: {search_request.query}")
            direct_results = await _direct_visual_search(
                video.file_path, 
                search_request.query, 
                search_request.max_results
            )
            if direct_results:
                logger.info(f"✅ Direct search succeeded with {len(direct_results.results)} results")
                return direct_results

        # IMPROVEMENT 1: Native video search using Gemini's video understanding capabilities
        if (use_native and 
            ENHANCED_FEATURES_AVAILABLE and
            video.video_type == "upload" and
            video.file_path and
            os.path.exists(video.file_path)):
            
            logger.info(f"Attempting native video search for query: '{search_request.query}'")
            try:
                # Initialize native search service
                native_search_service = NativeVideoSearchService()
                
                # Determine search type based on query
                query_lower = search_request.query.lower()
                search_type = "general"
                
                if any(word in query_lower for word in ['how many', 'count', 'number of']):
                    search_type = "counting"
                elif any(word in query_lower for word in ['text', 'sign', 'writing', 'label']):
                    search_type = "text"
                elif any(word in query_lower for word in ['scene', 'background', 'setting', 'location']):
                    search_type = "scene"
                elif any(word in query_lower for word in ['red', 'blue', 'green', 'yellow', 'color']):
                    search_type = "color"
                else:
                    search_type = "object"
                
                # Perform native video search
                clips = await native_search_service.search_visual_content(
                    video_path=video.file_path,
                    query=search_request.query,
                    search_type=search_type
                )
                
                if clips:
                    # Convert clips to SearchResult format
                    results = []
                    all_frames = []
                    
                    for clip in clips[:search_request.max_results]:
                        # Create a SearchResult for the clip's best moment
                        result = SearchResult(
                            timestamp=clip.start_time,
                            confidence=clip.confidence,
                            description=clip.description,
                            frame_path=clip.thumbnail_path if hasattr(clip, 'thumbnail_path') else f"/api/search/{video.id}/frame?timestamp={clip.start_time}",
                            summary=f"Clip {clip.start_time:.1f}s - {clip.end_time:.1f}s: {clip.description[:100]}",
                            detailed_analysis=clip.description
                        )
                        results.append(result)
                        
                        # Also create frame results for the clip
                        clip_duration = clip.end_time - clip.start_time
                        num_frames = min(5, max(2, int(clip_duration / 2)))  # 2-5 frames per clip
                        
                        for i in range(num_frames):
                            frame_time = clip.start_time + (i * clip_duration / num_frames)
                            frame_result = SearchResult(
                                timestamp=frame_time,
                                confidence=clip.confidence * (0.9 - i * 0.1),  # Slightly decrease confidence for later frames
                                description=f"Frame {i+1} from clip: {clip.description[:100]}",
                                frame_path=f"/api/search/{video.id}/frame?timestamp={frame_time}",
                                summary=clip.description[:100]
                            )
                            all_frames.append(frame_result)
                    
                    # Group results into clips
                    clip_results = []
                    for clip in clips[:max(5, search_request.max_results // 2)]:
                        # Get frames for this clip
                        clip_frames = [f for f in all_frames if clip.start_time <= f.timestamp <= clip.end_time]
                        
                        clip_result = ClipResult(
                            start_time=clip.start_time,
                            end_time=clip.end_time,
                            confidence=clip.confidence,
                            description=clip.description,
                            frame_count=len(clip_frames),
                            frames=clip_frames
                        )
                        clip_results.append(clip_result)
                    
                    # Determine query type and generate direct answer
                    direct_answer = None
                    query_type = search_type
                    
                    if search_type == "counting":
                        direct_answer = f"Found {len(clips)} instances of '{search_request.query}' in the video"
                    elif search_type == "object":
                        direct_answer = f"Object '{search_request.query}' detected in {len(clips)} clips"
                    elif search_type == "scene":
                        direct_answer = f"Scene analysis found {len(clips)} relevant moments"
                    elif search_type == "text":
                        direct_answer = f"Text matching '{search_request.query}' found in {len(clips)} locations"
                    
                    return VisualSearchResponse(
                        query=search_request.query,
                        results=results,
                        clips=clip_results,
                        total_results=len(clips),
                        direct_answer=direct_answer,
                        query_type=query_type,
                        processing_method="native_video_search"
                    )
                    
            except Exception as e:
                logger.warning(f"Native video search failed, falling back to other methods: {e}")

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
        enhanced_vector_service = None
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                enhanced_vector_service = EnhancedVectorService()
            except Exception as e:
                logger.warning(f"Failed to initialize EnhancedVectorService: {e}")

        vector_service = VectorService()  # Fallback

        if enhanced_vector_service and enhanced_vector_service.available:
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
        # Only use vector service if it has real embeddings (not hash-based fallback)
        if vector_service.available and getattr(vector_service, 'use_real_embeddings', False):
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

            logger.info(f"🔍 Found {len(frames)} frames in database for video {search_request.video_id}")
            results = await _enhanced_semantic_search(search_request.query, frames)
            logger.info(f"🔍 Semantic search returned {len(results)} results for query '{search_request.query}'")

        # Group results into clips
        clips = group_frames_into_clips(results)
        clips.sort(key=lambda c: c.confidence, reverse=True)

        # Provide clear feedback when no results are found
        direct_answer = None
        if len(results) == 0:
            direct_answer = f"No instances of '{search_request.query}' found in this video. The search analyzed frame descriptions and found no semantic matches."
        else:
            direct_answer = f"Found {len(results)} instances of '{search_request.query}' in the video"

        return VisualSearchResponse(
            query=search_request.query,
            results=results[:search_request.max_results],
            clips=clips[:max(5, search_request.max_results // 2)],
            total_results=len(results),
            direct_answer=direct_answer,
            processing_method="semantic_search"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced visual search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _validate_semantic_match(query: str, description: str, exact_matches: int, semantic_matches: int) -> bool:
    """Validate that a semantic match is genuine and not a false positive"""
    if not description:
        return False

    # Must have at least one exact match OR multiple semantic matches
    if exact_matches > 0:
        return True

    # For semantic-only matches, require multiple matches or high-confidence terms
    if semantic_matches >= 2:
        return True

    # Special case: single semantic match must be very specific
    query_lower = query.lower()
    description_lower = description.lower()

    # High-confidence single semantic matches
    high_confidence_terms = {
        'car': ['vehicle', 'automobile', 'truck', 'van', 'suv'],
        'person': ['human', 'individual', 'man', 'woman'],
        'building': ['structure', 'house', 'office', 'tower'],
        'text': ['sign', 'writing', 'words', 'letters'],
        'screen': ['monitor', 'display', 'television', 'computer']
    }

    for query_word in query_lower.split():
        if query_word in high_confidence_terms:
            for term in high_confidence_terms[query_word]:
                if term in description_lower:
                    return True

    return False

async def _enhanced_semantic_search(query: str, frames: List[VideoFrame]) -> List[SearchResult]:
    """Enhanced semantic visual search using detailed frame descriptions with strict validation"""
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

    logger.info(f"🔍 Enhanced search for '{query}' with expanded terms: {expanded_query_words}")
    logger.info(f"🔍 Processing {len(frames)} frames")

    for frame in frames:
        confidence = 0.0
        description = ""
        
        if frame.description:
            description_lower = frame.description.lower()
            
            # Multi-level matching strategy with WORD BOUNDARIES to prevent partial matches
            import re

            # Use word boundaries to ensure we match complete words only
            exact_matches = 0
            for word in query_words:
                # Create regex pattern with word boundaries
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, description_lower, re.IGNORECASE):
                    exact_matches += 1

            semantic_matches = 0
            for word in expanded_query_words:
                if word not in query_words:  # Don't double-count exact matches
                    pattern = r'\b' + re.escape(word) + r'\b'
                    if re.search(pattern, description_lower, re.IGNORECASE):
                        semantic_matches += 1
            
            # Color + object combination matching (e.g., "red car")
            color_object_bonus = 0.0
            if len(query_words) >= 2:
                colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown']
                objects = ['car', 'person', 'building', 'shirt', 'dress', 'hat', 'bag', 'phone', 'book', 'screen']
                
                query_colors = [w for w in query_words if w in colors]
                query_objects = [w for w in query_words if w in objects]
                
                if query_colors and query_objects:
                    # Use word boundaries for color and object matching too
                    color_in_desc = any(re.search(r'\b' + re.escape(color) + r'\b', description_lower, re.IGNORECASE) for color in query_colors)
                    object_in_desc = any(re.search(r'\b' + re.escape(obj) + r'\b', description_lower, re.IGNORECASE) for obj in query_objects)
                    if color_in_desc and object_in_desc:
                        color_object_bonus = 15.0  # 15% bonus for color+object match
            
            # Calculate confidence score (0-100 range) - STRICT MATCHING WITH VALIDATION
            if exact_matches > 0 or semantic_matches > 0:
                # VALIDATE: Check if this is a genuine semantic match
                is_valid_match = _validate_semantic_match(query, frame.description, exact_matches, semantic_matches)

                if not is_valid_match:
                    logger.debug(f"❌ Rejected invalid semantic match at {frame.timestamp:.1f}s for query '{query}'")
                    continue

                # Require at least one exact match for high confidence
                if exact_matches > 0:
                    exact_score = (exact_matches / len(query_words)) * 70 if query_words else 0
                    semantic_score = (semantic_matches / len(expanded_query_words)) * 15 if expanded_query_words else 0
                    base_bonus = 15.0  # Bonus for having exact matches
                else:
                    # Lower confidence for semantic-only matches
                    exact_score = 0
                    semantic_score = (semantic_matches / len(expanded_query_words)) * 25 if expanded_query_words else 0
                    base_bonus = 0.0

                # Total confidence - more conservative scoring
                confidence = min(95.0, exact_score + semantic_score + color_object_bonus + base_bonus)

                # Extract relevant portion of description using word boundaries
                sentences = frame.description.split('.')
                relevant_sentence = ""
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # Check if any query word appears as a complete word in this sentence
                    if any(re.search(r'\b' + re.escape(word) + r'\b', sentence_lower, re.IGNORECASE) for word in expanded_query_words):
                        relevant_sentence = sentence.strip()
                        break

                if not relevant_sentence and sentences:
                    relevant_sentence = sentences[0].strip()

                description = f"Match for '{query}': {relevant_sentence[:200]}..."
        
        # STRICT MATCHING: Only include frames with actual semantic matches
        # No fallback to pattern-based detection to prevent false positives

        if confidence > 30.0:  # Higher threshold to ensure quality matches
            logger.debug(f"✅ Found match at {frame.timestamp:.1f}s with confidence {confidence:.1f}% for query '{query}'")
            logger.debug(f"   Exact matches: {exact_matches if 'exact_matches' in locals() else 0}, Semantic matches: {semantic_matches if 'semantic_matches' in locals() else 0}")
            logger.debug(f"   Description: {description[:100]}...")

            results.append(SearchResult(
                timestamp=frame.timestamp,
                confidence=round(confidence, 1),
                description=description,
                frame_path=frame.frame_path
            ))
        elif confidence > 0:
            logger.debug(f"❌ Rejected match at {frame.timestamp:.1f}s with low confidence {confidence:.1f}% for query '{query}'")
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x.confidence, reverse=True)

    if len(results) == 0:
        logger.info(f"🔍 No semantic matches found for query '{query}' - this is correct if the content is not in the video")
    else:
        logger.info(f"🔍 Enhanced semantic search returned {len(results)} valid results for '{query}'")

    return results

def _pattern_based_detection(query_lower: str, timestamp: float, original_query: str) -> tuple[float, str]:
    """Fallback pattern-based detection for frames without descriptions - DISABLED to prevent false positives"""
    # IMPORTANT: This function has been disabled to prevent false positives
    # Only return matches if we have actual frame descriptions to analyze
    return 0.0, ""

async def _direct_visual_search(video_path: str, query: str, max_results: int = 10) -> Optional[VisualSearchResponse]:
    """
    Direct, simple visual search that actually works.
    Extracts frames and analyzes them with Gemini Vision.
    """
    try:
        import cv2
        import tempfile
        
        logger.info(f"🎯 Starting direct visual search for '{query}' on {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f} FPS, {total_frames} frames")
        
        # Extract frames every 5 seconds for analysis
        interval_seconds = 5
        frame_interval = int(fps * interval_seconds)
        
        results = []
        clips = []
        
        # Initialize Gemini service
        try:
            from app.services.gemini_service import GeminiService
            gemini_service = GeminiService()
        except Exception as e:
            logger.error(f"Could not initialize Gemini service: {e}")
            return None
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame at intervals
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Save frame to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, frame)
                    
                    # Analyze frame with Gemini Vision
                    try:
                        analysis = await gemini_service.analyze_frame_for_search(
                            tmp_file.name, 
                            query
                        )
                        
                        if analysis.get("match", False):
                            confidence = analysis.get("confidence", 0.5)
                            description = analysis.get("description", f"Match for '{query}' found")
                            
                            # Create search result
                            result = SearchResult(
                                timestamp=timestamp,
                                confidence=confidence * 100,
                                description=description,
                                frame_path=f"/api/search/frame/{timestamp}",
                                summary=f"Found '{query}' at {timestamp:.1f}s"
                            )
                            results.append(result)
                            
                            # Create clip (5-10 seconds around the match)
                            clip_start = max(0, timestamp - 2.5)
                            clip_end = min(duration, timestamp + 7.5)
                            
                            clip = ClipResult(
                                start_time=clip_start,
                                end_time=clip_end,
                                confidence=confidence * 100,
                                description=f"Clip containing '{query}' - {description}",
                                frame_count=1,
                                frames=[result]
                            )
                            clips.append(clip)
                            
                            logger.info(f"✅ Found '{query}' at {timestamp:.1f}s with confidence {confidence:.2f}")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing frame at {timestamp:.1f}s: {e}")
                        continue
                    
                    # Clean up temp file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
            
            frame_count += 1
            
            # Stop if we have enough results
            if len(results) >= max_results:
                break
        
        cap.release()
        
        # Sort results by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        clips.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"🎯 Direct search completed: {len(results)} results for '{query}'")
        
        if results:
            return VisualSearchResponse(
                query=query,
                results=results[:max_results],
                clips=clips[:max_results],
                total_results=len(results),
                direct_answer=f"Found {len(results)} instances of '{query}' in the video",
                query_type="direct_visual_search",
                processing_method="direct_frame_analysis"
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Error in direct visual search: {e}")
        return None

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

        # Try enhanced vector service first (IMPROVEMENT 1: Direct image embeddings)
        enhanced_vector_service = None
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                enhanced_vector_service = EnhancedVectorService()
            except Exception as e:
                logger.warning(f"Failed to initialize EnhancedVectorService: {e}")

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
                    if enhanced_vector_service and enhanced_vector_service.available:
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
        enhanced_stats = enhanced_vector_service.get_stats() if enhanced_vector_service and enhanced_vector_service.available else {}

        return {
            "message": f"Analyzed {analyzed_count} frames with enhanced embeddings",
            "analyzed_count": analyzed_count,
            "total_frames": len(frames),
            "video_total_frames": total_frames,
            "video_analyzed_frames": analyzed_frames,
            "enhanced_embeddings": enhanced_vector_service.available if enhanced_vector_service else False,
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


# Native Video Search Endpoints using Gemini 2.5

class NativeSearchRequest(BaseModel):
    video_id: int
    query: str
    search_type: str = "general"  # general, object, counting, color, text, scene

class NativeSearchResponse(BaseModel):
    query: str
    search_type: str
    clips: List[Dict]
    total_clips: int
    processing_time: float
    
class CountingRequest(BaseModel):
    video_id: int
    element: str
    count_type: str = "unique"  # unique or total

class CountingResponse(BaseModel):
    element: str
    count_type: str
    total_count: int
    instances: List[Dict]
    temporal_pattern: str
    clips: List[Dict]
    confidence: float

class ColorObjectRequest(BaseModel):
    video_id: int
    color: str
    object_type: str

@router.post("/native/search", response_model=NativeSearchResponse)
async def native_video_search(
    search_request: NativeSearchRequest,
    db: Session = Depends(get_db)
):
    """Native video search using Gemini 2.5's video understanding capabilities"""
    try:
        import time
        start_time = time.time()
        
        # Get video
        video = db.query(Video).filter(Video.id == search_request.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Initialize native search service
        search_service = NativeVideoSearchService()
        
        # Perform search based on type
        clips = await search_service.search_visual_content(
            video_path=video.file_path,
            query=search_request.query,
            search_type=search_request.search_type
        )
        
        # Convert clips to dict format
        clips_data = [clip.to_dict() for clip in clips]
        
        processing_time = time.time() - start_time
        
        return NativeSearchResponse(
            query=search_request.query,
            search_type=search_request.search_type,
            clips=clips_data,
            total_clips=len(clips_data),
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in native video search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/native/count", response_model=CountingResponse)
async def native_count_elements(
    counting_request: CountingRequest,
    db: Session = Depends(get_db)
):
    """Count visual elements in video using native video understanding"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == counting_request.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Perform counting
        result = await count_occurrences(
            video.file_path,
            counting_request.element,
            unique_only=(counting_request.count_type == "unique")
        )
        
        return CountingResponse(
            element=counting_request.element,
            count_type=counting_request.count_type,
            total_count=result.get("total_count", 0),
            instances=result.get("instances", []),
            temporal_pattern=result.get("temporal_pattern", ""),
            clips=result.get("clips", []),
            confidence=result.get("confidence", 0.8)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in native counting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/native/color-object")
async def native_color_object_search(
    color_request: ColorObjectRequest,
    db: Session = Depends(get_db)
):
    """Search for specific color + object combinations"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == color_request.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Search for color + object
        clips = await find_color_objects(
            video.file_path,
            color_request.color,
            color_request.object_type
        )
        
        return {
            "query": f"{color_request.color} {color_request.object_type}",
            "clips": clips,
            "total_clips": len(clips),
            "search_type": "color_object_combo"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in color-object search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/native/text-search")
async def native_text_search(
    video_id: int,
    text: str,
    db: Session = Depends(get_db)
):
    """Search for text appearing in video"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Search for text
        clips = await find_text(video.file_path, text)
        
        return {
            "query": text,
            "clips": clips,
            "total_clips": len(clips),
            "search_type": "text_detection"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/native/scene-search")
async def native_scene_search(
    video_id: int,
    scene_type: str,
    db: Session = Depends(get_db)
):
    """Search for specific types of scenes"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Search for scenes
        clips = await find_scenes(video.file_path, scene_type)
        
        return {
            "query": scene_type,
            "clips": clips,
            "total_clips": len(clips),
            "search_type": "scene_analysis"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scene search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/native/find-all")
async def native_find_all_occurrences(
    video_id: int,
    element: str,
    db: Session = Depends(get_db)
):
    """Find all occurrences of a visual element throughout the video"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Initialize service and find all occurrences
        search_service = NativeVideoSearchService()
        clips = await search_service.find_all_occurrences(video.file_path, element)
        
        # Convert to dict format
        clips_data = [clip.to_dict() for clip in clips]
        
        return {
            "element": element,
            "clips": clips_data,
            "total_occurrences": len(clips_data),
            "search_type": "exhaustive_search"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding all occurrences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/native/cleanup/{video_id}")
async def cleanup_native_upload(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Clean up uploaded video from Gemini to free resources"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path:
            return {"message": "No video file to clean up"}
        
        # Initialize service and cleanup
        search_service = NativeVideoSearchService()
        success = await search_service.cleanup_upload(video.file_path)
        
        return {
            "video_id": video_id,
            "cleanup_success": success,
            "message": "Video upload cleaned up from Gemini" if success else "No upload found to clean"
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up video upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}/frame")
async def get_video_frame_at_timestamp(
    video_id: int,
    timestamp: float,
    size: str = "medium",
    db: Session = Depends(get_db)
):
    """Get a frame from the video at a specific timestamp"""
    try:
        from fastapi.responses import FileResponse
        import cv2
        import tempfile
        
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video.file_path or not os.path.exists(video.file_path):
            # Return a placeholder image if video file not found
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Initialize thumbnail service if available
        thumbnail_service = None
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                thumbnail_service = ThumbnailService()
            except Exception as e:
                logger.warning(f"Could not initialize thumbnail service: {e}")
        
        # Try to use thumbnail service first
        if thumbnail_service:
            try:
                thumbnail_path = thumbnail_service.get_frame_at_timestamp(
                    video.file_path,
                    timestamp,
                    size=size
                )
                if thumbnail_path and os.path.exists(thumbnail_path):
                    return FileResponse(
                        thumbnail_path,
                        media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"}
                    )
            except Exception as e:
                logger.warning(f"Thumbnail service failed: {e}")
        
        # Fallback to direct frame extraction
        cap = cv2.VideoCapture(video.file_path)
        try:
            # Set position to timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                raise HTTPException(status_code=404, detail="Could not extract frame")
            
            # Resize based on size parameter
            sizes = {
                "small": (120, 68),
                "medium": (240, 135),
                "large": (480, 270),
                "timeline": (160, 90)
            }
            
            target_size = sizes.get(size, sizes["medium"])
            frame = cv2.resize(frame, target_size)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return FileResponse(
                    tmp.name,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"}
                )
                
        finally:
            cap.release()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))
