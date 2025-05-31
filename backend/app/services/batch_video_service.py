"""
Batch Video Processing Service using Gemini's Video API
Implements efficient batch processing for entire videos instead of frame-by-frame analysis
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import google.generativeai as genai
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

@dataclass
class VideoAnalysisResult:
    """Result from batch video analysis"""
    video_id: int
    status: str
    summary: str
    key_moments: List[Dict[str, Any]]
    objects_timeline: Dict[str, List[float]]  # object -> list of timestamps
    people_timeline: Dict[str, List[float]]   # person -> list of timestamps
    scene_changes: List[Dict[str, Any]]
    searchable_content: List[Dict[str, Any]]
    processing_time: float
    error: Optional[str] = None

@dataclass
class BatchSearchResult:
    """Result from batch video search"""
    query: str
    direct_answer: str
    query_type: str
    confidence: float
    timestamps: List[float]
    descriptions: List[str]
    context: Dict[str, Any]

class BatchVideoService:
    """Service for batch video processing using Gemini's video capabilities"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.available = False
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Gemini service for video processing"""
        try:
            genai.configure(api_key=self.api_key)
            
            # Use Gemini 2.5 Flash for video processing
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            self.available = True
            logger.info("Batch Video Service initialized with Gemini 2.5 Flash")
            
        except Exception as e:
            logger.error(f"Failed to initialize Batch Video Service: {e}")
            self.available = False
    
    async def upload_video_to_gemini(self, video_path: str) -> Optional[str]:
        """Upload video to Gemini and get file URI"""
        try:
            # Upload video file to Gemini
            video_file = genai.upload_file(
                path=video_path,
                display_name=f"video_analysis_{Path(video_path).stem}"
            )
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                logger.error(f"Video upload failed: {video_file.state}")
                return None
            
            logger.info(f"Video uploaded successfully: {video_file.uri}")
            return video_file.uri
            
        except Exception as e:
            logger.error(f"Error uploading video to Gemini: {e}")
            return None
    
    async def analyze_entire_video(self, video_path: str, video_id: int, video_title: str = "") -> VideoAnalysisResult:
        """Analyze entire video in one batch request"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Upload video to Gemini
            video_uri = await self.upload_video_to_gemini(video_path)
            if not video_uri:
                return VideoAnalysisResult(
                    video_id=video_id,
                    status="error",
                    summary="",
                    key_moments=[],
                    objects_timeline={},
                    people_timeline={},
                    scene_changes=[],
                    searchable_content=[],
                    processing_time=0,
                    error="Failed to upload video"
                )
            
            # Create comprehensive analysis prompt
            prompt = f"""
            Analyze this entire video comprehensively for multimodal search capabilities.
            Video Title: {video_title}
            
            Please provide a detailed analysis in the following JSON format:
            
            {{
                "summary": "Brief 2-3 sentence summary of the video content",
                "key_moments": [
                    {{
                        "timestamp": 15.5,
                        "title": "Introduction",
                        "description": "Speaker introduces himself and the topic",
                        "importance": "high",
                        "objects": ["microphone", "person", "chair"],
                        "people_count": 1,
                        "scene_type": "presentation"
                    }}
                ],
                "objects_timeline": {{
                    "microphone": [0, 15.5, 45.2, 120.8, 180.3, 240.7],
                    "person": [0, 15.5, 45.2, 120.8, 180.3, 240.7],
                    "chair": [0, 15.5, 45.2, 120.8]
                }},
                "people_timeline": {{
                    "adult_male_speaker": [0, 15.5, 45.2, 120.8, 180.3, 240.7]
                }},
                "scene_changes": [
                    {{
                        "timestamp": 0,
                        "scene_type": "studio_setup",
                        "description": "Professional recording studio with microphone",
                        "lighting": "professional",
                        "background": "solid_color"
                    }}
                ],
                "searchable_content": [
                    {{
                        "timestamp": 15.5,
                        "content_type": "object",
                        "primary_object": "microphone",
                        "description": "Professional Shure SM7B microphone visible",
                        "searchable_terms": ["microphone", "mic", "shure", "sm7b", "audio", "recording"],
                        "confidence": 0.95
                    }},
                    {{
                        "timestamp": 15.5,
                        "content_type": "person",
                        "description": "Adult male speaker in black shirt",
                        "searchable_terms": ["person", "speaker", "man", "presenter", "black shirt"],
                        "confidence": 0.92
                    }}
                ]
            }}
            
            Focus on:
            1. Temporal object detection and tracking
            2. People counting and identification throughout video
            3. Scene analysis and transitions
            4. Searchable moments with high-confidence object/person detection
            5. Detailed timestamps for all significant visual elements
            
            Provide precise timestamps and be extremely detailed for search optimization.
            """
            
            # Get video file object
            video_file = genai.get_file(video_uri.split('/')[-1])
            
            # Generate analysis
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, video_file]
            )
            
            # Parse response
            try:
                analysis_data = json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback: extract JSON from response text
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON response")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return VideoAnalysisResult(
                video_id=video_id,
                status="success",
                summary=analysis_data.get("summary", ""),
                key_moments=analysis_data.get("key_moments", []),
                objects_timeline=analysis_data.get("objects_timeline", {}),
                people_timeline=analysis_data.get("people_timeline", {}),
                scene_changes=analysis_data.get("scene_changes", []),
                searchable_content=analysis_data.get("searchable_content", []),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Error in batch video analysis: {e}")
            
            return VideoAnalysisResult(
                video_id=video_id,
                status="error",
                summary="",
                key_moments=[],
                objects_timeline={},
                people_timeline={},
                scene_changes=[],
                searchable_content=[],
                processing_time=processing_time,
                error=str(e)
            )
    
    async def batch_search_video(self, video_uri: str, query: str, video_context: str = "") -> BatchSearchResult:
        """Perform search query on entire video using batch processing"""
        try:
            prompt = f"""
            Search this video for: "{query}"
            Video Context: {video_context}
            
            Analyze the entire video and provide search results in this JSON format:
            
            {{
                "direct_answer": "Concise answer to the query",
                "query_type": "counting|object_detection|scene_analysis|general",
                "confidence": 0.95,
                "timestamps": [15.5, 45.2, 120.8],
                "descriptions": [
                    "Description of what's found at timestamp 15.5",
                    "Description of what's found at timestamp 45.2",
                    "Description of what's found at timestamp 120.8"
                ],
                "context": {{
                    "total_occurrences": 3,
                    "pattern": "Consistent throughout video",
                    "objects_detected": ["microphone", "boom arm"],
                    "people_count": 1,
                    "scene_analysis": "Professional studio setup"
                }}
            }}
            
            For counting queries: Provide exact counts and all timestamps
            For object detection: List all occurrences with precise timestamps
            For scene analysis: Describe visual elements and context
            For general queries: Provide relevant moments and descriptions
            
            Be precise with timestamps and provide detailed context.
            """
            
            # Get video file object
            video_file = genai.get_file(video_uri.split('/')[-1])
            
            # Generate search results
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, video_file]
            )
            
            # Parse response
            try:
                search_data = json.loads(response.text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    search_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON response")
            
            return BatchSearchResult(
                query=query,
                direct_answer=search_data.get("direct_answer", ""),
                query_type=search_data.get("query_type", "general"),
                confidence=search_data.get("confidence", 0.0),
                timestamps=search_data.get("timestamps", []),
                descriptions=search_data.get("descriptions", []),
                context=search_data.get("context", {})
            )
            
        except Exception as e:
            logger.error(f"Error in batch video search: {e}")
            return BatchSearchResult(
                query=query,
                direct_answer=f"Error processing query: {str(e)}",
                query_type="error",
                confidence=0.0,
                timestamps=[],
                descriptions=[],
                context={"error": str(e)}
            )
    
    def cleanup_uploaded_video(self, video_uri: str):
        """Clean up uploaded video from Gemini"""
        try:
            file_name = video_uri.split('/')[-1]
            genai.delete_file(file_name)
            logger.info(f"Cleaned up uploaded video: {file_name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup video {video_uri}: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and capabilities"""
        return {
            "available": self.available,
            "model": "gemini-2.5-flash",
            "capabilities": [
                "batch_video_analysis",
                "temporal_object_tracking",
                "scene_change_detection",
                "multimodal_search",
                "people_counting",
                "6_hour_video_support"
            ],
            "max_video_duration": "6 hours",
            "supported_formats": ["mp4", "mov", "avi", "webm"]
        }
