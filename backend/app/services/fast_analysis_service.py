"""
Fast Analysis Service
Optimized AI analysis with batch processing and smart caching
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import hashlib
from pathlib import Path
import google.generativeai as genai

logger = logging.getLogger(__name__)

@dataclass
class FastAnalysisResult:
    """Result from fast video analysis"""
    video_id: int
    status: str
    processing_time: float
    analysis_method: str
    summary: str
    key_moments: List[Dict]
    searchable_content: List[Dict]
    frame_descriptions: List[Dict]
    confidence_score: float
    cached: bool = False

class FastAnalysisService:
    """High-speed video analysis with intelligent optimizations"""
    
    def __init__(self, api_key: str, cache_dir: str = "./analysis_cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize Gemini with optimized settings
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # Fastest model
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,  # Reduced for speed
            }
        )
        
        # Performance settings
        self.max_concurrent_requests = 3
        self.batch_size = 5  # Analyze 5 frames at once
        self.use_cache = True
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    
    def _get_cache_key(self, video_path: str, frame_paths: List[str]) -> str:
        """Generate cache key for analysis results"""
        content = f"{video_path}:{':'.join(sorted(frame_paths))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load analysis from cache"""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save analysis to cache"""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def fast_analyze_video(self, video_path: str, video_id: int, 
                               frame_paths: List[str], video_title: str = "") -> FastAnalysisResult:
        """Fast video analysis using optimized batch processing"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(video_path, frame_paths)
            cached_result = self._load_from_cache(cache_key)
            
            if cached_result:
                logger.info(f"Using cached analysis for video {video_id}")
                return FastAnalysisResult(
                    video_id=video_id,
                    status="success",
                    processing_time=time.time() - start_time,
                    analysis_method="cached",
                    summary=cached_result.get("summary", ""),
                    key_moments=cached_result.get("key_moments", []),
                    searchable_content=cached_result.get("searchable_content", []),
                    frame_descriptions=cached_result.get("frame_descriptions", []),
                    confidence_score=cached_result.get("confidence_score", 0.8),
                    cached=True
                )
            
            # Determine analysis strategy based on video length
            if len(frame_paths) <= 10:
                # Small video: analyze all frames individually for high quality
                result = await self._analyze_individual_frames(video_id, frame_paths, video_title)
                analysis_method = "individual_frames"
            elif len(frame_paths) <= 30:
                # Medium video: batch analysis
                result = await self._analyze_frame_batches(video_id, frame_paths, video_title)
                analysis_method = "batch_analysis"
            else:
                # Large video: use video-level analysis if possible
                if video_path and Path(video_path).exists():
                    result = await self._analyze_whole_video(video_path, video_id, video_title)
                    analysis_method = "whole_video"
                else:
                    # Fallback to sampling key frames
                    sampled_frames = self._sample_key_frames(frame_paths, 20)
                    result = await self._analyze_frame_batches(video_id, sampled_frames, video_title)
                    analysis_method = "sampled_frames"
            
            processing_time = time.time() - start_time
            
            # Save to cache
            cache_data = {
                "summary": result.get("summary", ""),
                "key_moments": result.get("key_moments", []),
                "searchable_content": result.get("searchable_content", []),
                "frame_descriptions": result.get("frame_descriptions", []),
                "confidence_score": result.get("confidence_score", 0.8),
                "analysis_method": analysis_method,
                "processing_time": processing_time
            }
            self._save_to_cache(cache_key, cache_data)
            
            return FastAnalysisResult(
                video_id=video_id,
                status="success",
                processing_time=processing_time,
                analysis_method=analysis_method,
                summary=result.get("summary", ""),
                key_moments=result.get("key_moments", []),
                searchable_content=result.get("searchable_content", []),
                frame_descriptions=result.get("frame_descriptions", []),
                confidence_score=result.get("confidence_score", 0.8),
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Error in fast video analysis: {e}")
            return FastAnalysisResult(
                video_id=video_id,
                status="error",
                processing_time=time.time() - start_time,
                analysis_method="failed",
                summary=f"Analysis failed: {str(e)}",
                key_moments=[],
                searchable_content=[],
                frame_descriptions=[],
                confidence_score=0.0
            )
    
    async def _analyze_individual_frames(self, video_id: int, frame_paths: List[str], 
                                       video_title: str) -> Dict:
        """Analyze frames individually for high quality"""
        frame_descriptions = []
        
        # Process frames in parallel with semaphore
        async def analyze_single_frame(frame_path: str, index: int):
            async with self.semaphore:
                try:
                    # Upload frame to Gemini
                    frame_file = genai.upload_file(frame_path)
                    
                    # Quick analysis prompt
                    prompt = f"""
                    Analyze this video frame briefly and concisely.
                    Video: {video_title}
                    
                    Provide a JSON response:
                    {{
                        "description": "Brief description of what's visible",
                        "objects": ["list", "of", "objects"],
                        "people_count": 0,
                        "scene_type": "indoor/outdoor/studio/etc",
                        "confidence": 0.9
                    }}
                    
                    Be concise but accurate.
                    """
                    
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        [prompt, frame_file]
                    )
                    
                    # Parse response
                    try:
                        analysis = json.loads(response.text)
                    except:
                        # Fallback parsing
                        analysis = {
                            "description": response.text[:200],
                            "objects": [],
                            "people_count": 0,
                            "scene_type": "unknown",
                            "confidence": 0.5
                        }
                    
                    # Clean up uploaded file
                    genai.delete_file(frame_file.name)
                    
                    return {
                        "frame_index": index,
                        "frame_path": frame_path,
                        "timestamp": index * 3.0,  # Estimate timestamp
                        **analysis
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing frame {frame_path}: {e}")
                    return {
                        "frame_index": index,
                        "frame_path": frame_path,
                        "timestamp": index * 3.0,
                        "description": "Analysis failed",
                        "objects": [],
                        "people_count": 0,
                        "scene_type": "unknown",
                        "confidence": 0.0
                    }
        
        # Analyze all frames in parallel
        tasks = [analyze_single_frame(frame_path, i) for i, frame_path in enumerate(frame_paths)]
        frame_descriptions = await asyncio.gather(*tasks)
        
        # Generate summary and key moments
        summary = self._generate_summary_from_frames(frame_descriptions)
        key_moments = self._extract_key_moments(frame_descriptions)
        searchable_content = self._create_searchable_content(frame_descriptions)
        
        return {
            "summary": summary,
            "key_moments": key_moments,
            "searchable_content": searchable_content,
            "frame_descriptions": frame_descriptions,
            "confidence_score": sum(f.get("confidence", 0) for f in frame_descriptions) / len(frame_descriptions)
        }
    
    async def _analyze_frame_batches(self, video_id: int, frame_paths: List[str], 
                                   video_title: str) -> Dict:
        """Analyze frames in batches for efficiency"""
        all_descriptions = []
        
        # Process in batches
        for i in range(0, len(frame_paths), self.batch_size):
            batch_paths = frame_paths[i:i + self.batch_size]
            
            async with self.semaphore:
                try:
                    # Upload batch of frames
                    frame_files = [genai.upload_file(path) for path in batch_paths]
                    
                    # Batch analysis prompt
                    prompt = f"""
                    Analyze these {len(batch_paths)} video frames from: {video_title}
                    
                    For each frame, provide analysis in this JSON format:
                    {{
                        "frames": [
                            {{
                                "frame_number": 0,
                                "description": "Brief description",
                                "objects": ["object1", "object2"],
                                "people_count": 0,
                                "scene_type": "indoor/outdoor/studio",
                                "confidence": 0.9
                            }}
                        ]
                    }}
                    
                    Be efficient and accurate.
                    """
                    
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        [prompt] + frame_files
                    )
                    
                    # Parse batch response
                    try:
                        batch_analysis = json.loads(response.text)
                        frames_data = batch_analysis.get("frames", [])
                    except:
                        # Fallback
                        frames_data = [{"frame_number": j, "description": "Batch analysis", 
                                      "objects": [], "people_count": 0, "scene_type": "unknown", 
                                      "confidence": 0.5} for j in range(len(batch_paths))]
                    
                    # Add frame paths and timestamps
                    for j, frame_data in enumerate(frames_data):
                        if j < len(batch_paths):
                            frame_data.update({
                                "frame_path": batch_paths[j],
                                "timestamp": (i + j) * 3.0,
                                "frame_index": i + j
                            })
                            all_descriptions.append(frame_data)
                    
                    # Clean up uploaded files
                    for frame_file in frame_files:
                        genai.delete_file(frame_file.name)
                        
                except Exception as e:
                    logger.error(f"Error in batch analysis: {e}")
                    # Add fallback descriptions
                    for j, path in enumerate(batch_paths):
                        all_descriptions.append({
                            "frame_index": i + j,
                            "frame_path": path,
                            "timestamp": (i + j) * 3.0,
                            "description": "Batch analysis failed",
                            "objects": [],
                            "people_count": 0,
                            "scene_type": "unknown",
                            "confidence": 0.0
                        })
        
        # Generate summary and key moments
        summary = self._generate_summary_from_frames(all_descriptions)
        key_moments = self._extract_key_moments(all_descriptions)
        searchable_content = self._create_searchable_content(all_descriptions)
        
        return {
            "summary": summary,
            "key_moments": key_moments,
            "searchable_content": searchable_content,
            "frame_descriptions": all_descriptions,
            "confidence_score": sum(f.get("confidence", 0) for f in all_descriptions) / len(all_descriptions) if all_descriptions else 0
        }
    
    def _sample_key_frames(self, frame_paths: List[str], target_count: int) -> List[str]:
        """Sample key frames from a large set"""
        if len(frame_paths) <= target_count:
            return frame_paths
        
        # Sample evenly distributed frames
        indices = [int(i * len(frame_paths) / target_count) for i in range(target_count)]
        return [frame_paths[i] for i in indices]
    
    def _generate_summary_from_frames(self, frame_descriptions: List[Dict]) -> str:
        """Generate video summary from frame descriptions"""
        if not frame_descriptions:
            return "No analysis available"
        
        # Extract common themes
        all_objects = []
        scene_types = []
        people_counts = []
        
        for frame in frame_descriptions:
            all_objects.extend(frame.get("objects", []))
            scene_types.append(frame.get("scene_type", "unknown"))
            people_counts.append(frame.get("people_count", 0))
        
        # Find most common elements
        from collections import Counter
        common_objects = [obj for obj, count in Counter(all_objects).most_common(3)]
        common_scene = Counter(scene_types).most_common(1)[0][0] if scene_types else "unknown"
        avg_people = sum(people_counts) / len(people_counts) if people_counts else 0
        
        summary = f"Video shows {common_scene} setting"
        if common_objects:
            summary += f" featuring {', '.join(common_objects)}"
        if avg_people > 0:
            summary += f" with approximately {avg_people:.0f} people visible"
        
        return summary
    
    def _extract_key_moments(self, frame_descriptions: List[Dict]) -> List[Dict]:
        """Extract key moments from frame analysis"""
        key_moments = []
        
        for i, frame in enumerate(frame_descriptions):
            # Consider frames with high confidence or interesting content as key moments
            if (frame.get("confidence", 0) > 0.8 or 
                len(frame.get("objects", [])) > 3 or
                frame.get("people_count", 0) > 0):
                
                key_moments.append({
                    "timestamp": frame.get("timestamp", i * 3.0),
                    "title": f"Key moment at {frame.get('timestamp', i * 3.0):.1f}s",
                    "description": frame.get("description", ""),
                    "importance": "high" if frame.get("confidence", 0) > 0.9 else "medium",
                    "objects": frame.get("objects", []),
                    "people_count": frame.get("people_count", 0)
                })
        
        return key_moments[:10]  # Limit to top 10 moments
    
    def _create_searchable_content(self, frame_descriptions: List[Dict]) -> List[Dict]:
        """Create searchable content from frame analysis"""
        searchable_content = []
        
        for frame in frame_descriptions:
            if frame.get("confidence", 0) > 0.5:  # Only include confident results
                searchable_content.append({
                    "timestamp": frame.get("timestamp", 0),
                    "content_type": "frame_analysis",
                    "description": frame.get("description", ""),
                    "objects": frame.get("objects", []),
                    "people_count": frame.get("people_count", 0),
                    "scene_type": frame.get("scene_type", "unknown"),
                    "confidence": frame.get("confidence", 0)
                })
        
        return searchable_content
