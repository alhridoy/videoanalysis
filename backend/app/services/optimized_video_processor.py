"""
Optimized Video Processing Service
Addresses performance bottlenecks for faster video analysis
"""

import logging
import asyncio
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import time

try:
    from app.core.config import settings
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ProcessingProgress:
    """Track processing progress for real-time updates"""
    stage: str
    progress: float
    message: str
    estimated_time_remaining: Optional[float] = None

class OptimizedVideoProcessor:
    """High-performance video processor with parallel processing and smart optimizations"""
    
    def __init__(self, upload_dir: str = "./uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.upload_dir / "frames").mkdir(exist_ok=True)
        (self.upload_dir / "thumbnails").mkdir(exist_ok=True)
        
        # Performance settings
        self.max_workers = min(4, os.cpu_count() or 1)  # Limit CPU usage
        self.batch_size = 10  # Process frames in batches

        # Use config values if available, otherwise use defaults
        if CONFIG_AVAILABLE:
            self.max_frames_per_video = settings.MAX_FRAMES_PER_VIDEO
            self.max_duration = settings.MAX_VIDEO_DURATION_SECONDS
        else:
            self.max_frames_per_video = 300  # Increased limit for longer videos
            self.max_duration = 3600  # 1 hour default

        self.target_fps = 1  # Extract 1 frame per second max
        
        # Progress tracking
        self.progress_callbacks = {}
    
    def register_progress_callback(self, video_id: int, callback):
        """Register callback for progress updates"""
        self.progress_callbacks[video_id] = callback
    
    def _update_progress(self, video_id: int, stage: str, progress: float, message: str, eta: Optional[float] = None):
        """Update processing progress"""
        if video_id in self.progress_callbacks:
            progress_info = ProcessingProgress(stage, progress, message, eta)
            try:
                self.progress_callbacks[video_id](progress_info)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def quick_process_video(self, file_path: str, video_id: int,
                                 max_duration: Optional[float] = None) -> Dict:
        """Optimized video processing with smart frame selection"""
        start_time = time.time()
        
        try:
            self._update_progress(video_id, "initializing", 0.0, "Starting video analysis...")
            
            # Get video info quickly
            video_info = await self._get_video_info_fast(file_path)
            duration = video_info.get("duration", 0)

            # Use provided max_duration or instance default
            effective_max_duration = max_duration if max_duration is not None else self.max_duration

            # Limit processing for very long videos
            if duration > effective_max_duration:
                logger.warning(f"Video duration {duration}s exceeds limit {effective_max_duration}s, truncating")
                duration = effective_max_duration
                video_info["duration"] = duration
            
            self._update_progress(video_id, "analyzing", 10.0, f"Video duration: {duration:.1f}s")
            
            # Smart frame extraction
            frames = await self._extract_frames_optimized(file_path, video_id, duration)
            
            self._update_progress(video_id, "completing", 90.0, f"Extracted {len(frames)} frames")
            
            processing_time = time.time() - start_time
            
            self._update_progress(video_id, "completed", 100.0, 
                                f"Processing completed in {processing_time:.1f}s")
            
            return {
                "status": "success",
                "video_info": video_info,
                "frames": frames,
                "frame_count": len(frames),
                "processing_time": processing_time,
                "optimization_applied": True
            }
            
        except Exception as e:
            logger.error(f"Error in optimized video processing: {e}")
            self._update_progress(video_id, "error", 0.0, f"Processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "frames": [],
                "frame_count": 0
            }
    
    async def _get_video_info_fast(self, file_path: str) -> Dict:
        """Fast video info extraction"""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {file_path}")
            
            # Get basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "file_size": os.path.getsize(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {"duration": 0, "fps": 30, "frame_count": 0, "width": 640, "height": 360}
    
    async def _extract_frames_optimized(self, file_path: str, video_id: int, duration: float) -> List[Dict]:
        """Optimized frame extraction with smart sampling"""
        try:
            # Calculate optimal frame extraction strategy for longer videos
            # For videos longer than 10 minutes, extract fewer frames per minute
            if duration > 600:  # 10 minutes
                target_frame_count = min(self.max_frames_per_video, max(30, int(duration / 10)))  # 1 frame per 10 seconds
            else:
                target_frame_count = min(self.max_frames_per_video, max(10, int(duration / 3)))  # 1 frame per 3 seconds
            
            # Use scene detection for smart sampling
            scene_timestamps = await self._detect_scene_changes_fast(file_path, duration, target_frame_count)
            
            self._update_progress(video_id, "extracting", 30.0, 
                                f"Extracting {len(scene_timestamps)} key frames...")
            
            # Extract frames in parallel
            frames = await self._extract_frames_parallel(file_path, video_id, scene_timestamps)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error in optimized frame extraction: {e}")
            return []
    
    async def _detect_scene_changes_fast(self, file_path: str, duration: float, 
                                       target_count: int) -> List[float]:
        """Fast scene change detection using sampling"""
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames for scene detection (every 30 frames for speed)
            sample_interval = max(30, total_frames // 100)  # Sample ~100 frames max
            scene_changes = [0.0]  # Always include first frame
            
            prev_hist = None
            frame_count = 0
            
            while frame_count < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Calculate histogram for scene change detection
                if frame is not None:
                    # Resize for speed
                    small_frame = cv2.resize(frame, (160, 90))
                    hist = cv2.calcHist([small_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    
                    if prev_hist is not None:
                        # Calculate histogram correlation
                        correlation = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                        
                        # If correlation is low, it's a scene change
                        if correlation < 0.7:  # Threshold for scene change
                            timestamp = frame_count / fps
                            if timestamp not in scene_changes and timestamp <= duration:
                                scene_changes.append(timestamp)
                    
                    prev_hist = hist
                
                frame_count += sample_interval
            
            cap.release()
            
            # Ensure we don't exceed target count
            if len(scene_changes) > target_count:
                # Keep evenly distributed frames
                indices = np.linspace(0, len(scene_changes) - 1, target_count, dtype=int)
                scene_changes = [scene_changes[i] for i in indices]
            elif len(scene_changes) < target_count:
                # Add evenly spaced frames to reach target
                additional_needed = target_count - len(scene_changes)
                for i in range(additional_needed):
                    timestamp = (i + 1) * duration / (additional_needed + 1)
                    if timestamp not in scene_changes:
                        scene_changes.append(timestamp)
            
            return sorted(scene_changes)
            
        except Exception as e:
            logger.error(f"Error in scene detection: {e}")
            # Fallback to evenly spaced frames
            return [i * duration / target_count for i in range(target_count)]
    
    async def _extract_frames_parallel(self, file_path: str, video_id: int, 
                                     timestamps: List[float]) -> List[Dict]:
        """Extract frames in parallel for speed"""
        try:
            frames = []
            
            # Process frames in batches to avoid memory issues
            for i in range(0, len(timestamps), self.batch_size):
                batch_timestamps = timestamps[i:i + self.batch_size]
                
                # Update progress
                progress = 30 + (i / len(timestamps)) * 50  # 30-80% range
                self._update_progress(video_id, "extracting", progress, 
                                    f"Processing batch {i//self.batch_size + 1}/{(len(timestamps) + self.batch_size - 1)//self.batch_size}")
                
                # Extract batch in parallel
                batch_frames = await self._extract_frame_batch(file_path, video_id, batch_timestamps, i)
                frames.extend(batch_frames)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error in parallel frame extraction: {e}")
            return []
    
    async def _extract_frame_batch(self, file_path: str, video_id: int, 
                                 timestamps: List[float], batch_index: int) -> List[Dict]:
        """Extract a batch of frames"""
        def extract_single_frame(args):
            file_path, timestamp, frame_index = args
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    return None
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_number = int(timestamp * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    return None
                
                # Optimize frame size for storage and processing
                height, width = frame.shape[:2]
                if width > 640:  # Resize large frames
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Save frame
                frame_filename = f"video_{video_id}_frame_{batch_index}_{frame_index}_{int(timestamp)}.jpg"
                frame_path = self.upload_dir / "frames" / frame_filename
                
                # Save with optimization
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                return {
                    "timestamp": timestamp,
                    "frame_path": str(frame_path),
                    "frame_number": frame_index
                }
                
            except Exception as e:
                logger.error(f"Error extracting frame at {timestamp}: {e}")
                return None
        
        # Prepare arguments for parallel processing
        args_list = [(file_path, timestamp, i) for i, timestamp in enumerate(timestamps)]
        
        # Use thread pool for I/O bound operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = await loop.run_in_executor(None, lambda: [extract_single_frame(args) for args in args_list])
        
        # Filter out failed extractions
        return [result for result in results if result is not None]
    
    async def get_processing_status(self, video_id: int) -> Dict:
        """Get current processing status"""
        # This would typically check a database or cache
        # For now, return a simple status
        return {
            "video_id": video_id,
            "status": "processing",
            "stage": "extracting",
            "progress": 50.0,
            "message": "Extracting key frames..."
        }
    
    def cleanup_progress_callback(self, video_id: int):
        """Clean up progress callback after processing"""
        if video_id in self.progress_callbacks:
            del self.progress_callbacks[video_id]
