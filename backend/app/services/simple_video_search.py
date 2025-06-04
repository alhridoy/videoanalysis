
import logging
import os
import time
from typing import List, Dict, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    timestamp: float
    confidence: float
    description: str
    frame_path: str
    clip_start: float
    clip_end: float

class ClipResult(BaseModel):
    start_time: float
    end_time: float
    confidence: float
    description: str
    thumbnail_url: Optional[str] = None

class VideoSearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    clips: List[ClipResult] = []
    total_results: int
    processing_method: str
    processing_time: float
    direct_answer: Optional[str] = None
    query_type: Optional[str] = None

class SimpleVideoSearch:
    """
    Simple, effective video search using the best available method.
    
    Priority:
    1. Native Gemini 2.5 video analysis (best accuracy, no preprocessing)
    2. Fallback to basic frame analysis if needed
    """
    
    def __init__(self):
        self.gemini_service = None
        self.native_service = None
        self._native_available = False

        # Native video search will be initialized when available
        logger.info("🔍 Simple video search initialized - native search will be enabled when available")
    
    def initialize_gemini(self, gemini_service):
        """Initialize with Gemini service from app state"""
        self.gemini_service = gemini_service

    def initialize_native_service(self, native_service):
        """Initialize the native video search service for fast searches"""
        self.native_service = native_service
        self._native_available = True
        logger.info("✅ Native video search service initialized - fast searches enabled!")
    
    async def search(self, video_path: str, query: str, video_id: int) -> VideoSearchResponse:
        """
        Simple search: try the best method first, fallback if needed
        """
        start_time = time.time()
        
        logger.info(f"🔍 Simple search for '{query}' in video {video_id}")
        
        # METHOD 1: Native Gemini 2.5 Video Analysis (DISABLED - NOT WORKING)
        if False and self.native_service and video_path and os.path.exists(video_path):
            try:
                results = await self._native_search(video_path, query)
                if results:
                    processing_time = time.time() - start_time
                    logger.info(f"✅ Native search found {len(results)} results in {processing_time:.2f}s")

                    # Convert results to clips for timeline/clips view
                    clips = self._results_to_clips(results)

                    return VideoSearchResponse(
                        query=query,
                        results=results,
                        clips=clips,
                        total_results=len(results),
                        processing_method="native_gemini_video",
                        processing_time=processing_time,
                        direct_answer=f"Found {len(results)} instances of '{query}' in the video",
                        query_type="visual_search"
                    )
            except Exception as e:
                logger.warning(f"Native search failed: {e}")
        
        # METHOD 2: Direct Frame Analysis (FALLBACK)
        if self.gemini_service and video_path and os.path.exists(video_path):
            try:
                results = await self._direct_frame_search(video_path, query, video_id)
                processing_time = time.time() - start_time
                logger.info(f"✅ Frame search found {len(results)} results in {processing_time:.2f}s")

                # Convert results to clips for timeline/clips view
                clips = self._results_to_clips(results)

                return VideoSearchResponse(
                    query=query,
                    results=results,
                    clips=clips,
                    total_results=len(results),
                    processing_method="direct_frame_analysis",
                    processing_time=processing_time,
                    direct_answer=f"Found {len(results)} instances of '{query}' in the video",
                    query_type="visual_search"
                )
            except Exception as e:
                logger.error(f"Frame search failed: {e}")
        
        # METHOD 3: No results (but don't fail)
        processing_time = time.time() - start_time
        return VideoSearchResponse(
            query=query,
            results=[],
            total_results=0,
            processing_method="no_results",
            processing_time=processing_time
        )
    
    async def _native_search(self, video_path: str, query: str) -> List[SearchResult]:
        """Use native Gemini 2.5 video understanding - the best method"""
        if not self.native_service:
            return []
        
        # Determine search type for optimal results
        query_lower = query.lower()
        if any(word in query_lower for word in ['how many', 'count', 'number of']):
            search_type = "counting"
        elif any(word in query_lower for word in ['red', 'blue', 'green', 'yellow', 'color']):
            search_type = "color"
        elif any(word in query_lower for word in ['text', 'sign', 'writing']):
            search_type = "text"
        else:
            search_type = "object"
        
        # Get clips from native search
        clips = await self.native_service.search_visual_content(
            video_path=video_path,
            query=query,
            search_type=search_type
        )
        
        # Convert to SearchResult format
        results = []
        for clip in clips:
            results.append(SearchResult(
                timestamp=clip.start_time,
                confidence=clip.confidence,
                description=clip.description,
                frame_path=f"/api/search/frame?video_path={video_path}&timestamp={clip.start_time}",
                clip_start=clip.start_time,
                clip_end=clip.end_time
            ))
        
        return results
    
    async def _direct_frame_search(self, video_path: str, query: str, video_id: int = None) -> List[SearchResult]:
        """
        Direct frame analysis - extract frames at reasonable density and analyze
        Much simpler than the overengineered version
        """
        import cv2
        import tempfile
        
        results = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample sparsely to prevent blocking - every 30 seconds for fast response
        sample_interval = 30.0  # seconds (much less dense but faster)
        frame_interval = int(fps * sample_interval)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame at intervals
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    
                    # Save frame temporarily
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        cv2.imwrite(tmp.name, frame)
                        
                        # Analyze with Gemini using correct method
                        try:
                            # Use the existing analyze_frame_for_search method
                            analysis = await self.gemini_service.analyze_frame_for_search(tmp.name, query)

                            # Check if match found
                            if analysis.get("match", False):
                                confidence = analysis.get("confidence", 0.5) * 100
                                description = analysis.get("description", f"Found '{query}' at {timestamp:.1f}s")

                                # Create clip around the match (±5 seconds)
                                clip_start = max(0, timestamp - 5)
                                clip_end = min(duration, timestamp + 5)

                                results.append(SearchResult(
                                    timestamp=timestamp,
                                    confidence=confidence,
                                    description=description,
                                    frame_path=f"/api/v1/search/{video_id}/frame?timestamp={timestamp}",
                                    clip_start=clip_start,
                                    clip_end=clip_end
                                ))
                        except Exception as e:
                            logger.warning(f"Frame analysis failed at {timestamp:.1f}s: {e}")
                        
                        # Clean up
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass
                
                frame_count += 1

                # Hard limits to prevent blocking
                if len(results) >= 5:  # Max 5 results
                    break
                if frame_count > total_frames // 10:  # Max 10% of frames
                    break
                    
        finally:
            cap.release()
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def _results_to_clips(self, results: List[SearchResult]) -> List[ClipResult]:
        """Convert search results to clips for timeline/clips view"""
        clips = []
        for result in results:
            clips.append(ClipResult(
                start_time=result.clip_start,
                end_time=result.clip_end,
                confidence=result.confidence,
                description=result.description,
                thumbnail_url=result.frame_path
            ))
        return clips

# Global instance
simple_search = SimpleVideoSearch()