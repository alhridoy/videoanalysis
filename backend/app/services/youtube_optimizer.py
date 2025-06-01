import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import subprocess
import json
import tempfile

from app.core.config import settings
from app.services.gemini_service import GeminiService
from app.services.youtube_service import YouTubeService

logger = logging.getLogger(__name__)


class YouTubeOptimizer:
    """
    Optimized YouTube video processor leveraging Gemini 2.5's native video understanding.
    Minimizes downloads and maximizes parallel processing.
    """
    
    def __init__(self):
        """Initialize the YouTube optimizer with required services"""
        self.gemini_service = GeminiService()
        self.youtube_service = YouTubeService()
        
        # Configure Gemini for direct video analysis
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for YouTube optimization")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Progress tracking
        self.progress_callbacks = {}
        
    async def analyze_youtube_video_direct(
        self, 
        youtube_url: str, 
        analysis_options: Dict = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Analyze YouTube video directly using Gemini without downloading the full video.
        
        Args:
            youtube_url: YouTube video URL
            analysis_options: Dict with options like 'include_transcript', 'extract_moments', etc.
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with analysis results including transcript, moments, visual descriptions
        """
        try:
            # Extract video ID
            video_id = self.youtube_service.extract_video_id(youtube_url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            if progress_callback:
                self.progress_callbacks[video_id] = progress_callback
                await self._update_progress(video_id, 10, "Extracting video information...")
            
            # Parallel task 1: Get video info and transcript
            video_info_task = asyncio.create_task(self._get_video_info_enhanced(video_id))
            transcript_task = asyncio.create_task(self._get_transcript_with_timing(video_id))
            
            # Wait for basic info
            video_info, transcript_data = await asyncio.gather(
                video_info_task, 
                transcript_task
            )
            
            await self._update_progress(video_id, 30, "Analyzing video content with Gemini...")
            
            # Analyze video directly with Gemini using YouTube URL
            analysis_tasks = []
            
            # Task 1: Direct video analysis (if Gemini 2.5 supports YouTube URLs)
            if self._check_gemini_youtube_support():
                analysis_tasks.append(
                    asyncio.create_task(self._analyze_youtube_with_gemini_direct(youtube_url, video_info, transcript_data))
                )
            
            # Task 2: Transcript-based analysis (always available)
            analysis_tasks.append(
                asyncio.create_task(self._analyze_transcript_content(transcript_data, video_info))
            )
            
            # Task 3: Generate video sections
            if analysis_options and analysis_options.get('extract_sections', True):
                analysis_tasks.append(
                    asyncio.create_task(self._generate_smart_sections(transcript_data, video_info))
                )
            
            await self._update_progress(video_id, 50, "Processing analysis results...")
            
            # Gather all analysis results
            analysis_results = await asyncio.gather(*analysis_tasks)
            
            # Combine results
            combined_analysis = {
                "video_info": video_info,
                "transcript": transcript_data.get('text', ''),
                "transcript_data": transcript_data.get('transcript', []),
                "analysis": {},
                "sections": [],
                "visual_moments": [],
                "status": "success"
            }
            
            # Merge analysis results
            for result in analysis_results:
                if isinstance(result, dict):
                    if 'sections' in result:
                        combined_analysis['sections'] = result['sections']
                    elif 'visual_analysis' in result:
                        combined_analysis['analysis']['visual'] = result['visual_analysis']
                        combined_analysis['visual_moments'] = result.get('moments', [])
                    elif 'content_analysis' in result:
                        combined_analysis['analysis']['content'] = result['content_analysis']
            
            await self._update_progress(video_id, 80, "Finalizing analysis...")
            
            # Extract key frames only if needed for visual search
            if analysis_options and analysis_options.get('enable_visual_search', False):
                await self._update_progress(video_id, 85, "Preparing visual search capabilities...")
                key_frames = await self._extract_key_frames_smart(video_id, combined_analysis)
                combined_analysis['key_frames'] = key_frames
            
            await self._update_progress(video_id, 100, "Analysis complete!")
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error in YouTube video analysis: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # Clean up progress callback
            if video_id in self.progress_callbacks:
                del self.progress_callbacks[video_id]
    
    async def process_visual_search_query(
        self,
        video_id: str,
        search_query: str,
        cached_analysis: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process a visual search query by extracting only necessary frames.
        
        Args:
            video_id: YouTube video ID
            search_query: Visual search query
            cached_analysis: Optional cached analysis to avoid re-processing
            
        Returns:
            List of matching moments with timestamps and descriptions
        """
        try:
            await self._update_progress(video_id, 10, "Processing visual search query...")
            
            # If we have cached visual analysis, search within it first
            if cached_analysis and 'visual_moments' in cached_analysis:
                matches = await self._search_cached_visual_data(
                    cached_analysis['visual_moments'], 
                    search_query
                )
                if matches:
                    return matches
            
            await self._update_progress(video_id, 30, "Identifying relevant video segments...")
            
            # Download only specific segments based on query analysis
            relevant_segments = await self._identify_relevant_segments(video_id, search_query, cached_analysis)
            
            if not relevant_segments:
                # Fallback: Extract frames at regular intervals
                relevant_segments = self._generate_default_segments(
                    cached_analysis.get('video_info', {}).get('duration', 300)
                )
            
            await self._update_progress(video_id, 50, "Extracting and analyzing frames...")
            
            # Process segments in parallel
            segment_tasks = []
            for i, segment in enumerate(relevant_segments[:10]):  # Limit to 10 segments
                task = asyncio.create_task(
                    self._process_video_segment(video_id, segment, search_query, i)
                )
                segment_tasks.append(task)
            
            # Gather results
            segment_results = await asyncio.gather(*segment_tasks)
            
            await self._update_progress(video_id, 80, "Ranking search results...")
            
            # Combine and rank results
            all_matches = []
            for results in segment_results:
                if results:
                    all_matches.extend(results)
            
            # Sort by relevance score
            all_matches.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            await self._update_progress(video_id, 100, "Visual search complete!")
            
            return all_matches[:20]  # Return top 20 matches
            
        except Exception as e:
            logger.error(f"Error in visual search: {e}")
            return []
    
    async def _analyze_youtube_with_gemini_direct(
        self,
        youtube_url: str,
        video_info: Dict,
        transcript_data: Dict
    ) -> Dict:
        """Analyze YouTube video directly with Gemini (when supported)"""
        try:
            # For now, download a low-quality version for analysis
            # In future, this would use direct YouTube URL analysis
            temp_video_path = await self._download_low_quality_video(
                video_info['id'], 
                max_duration=300  # Limit to 5 minutes for quick analysis
            )
            
            if not temp_video_path:
                return {}
            
            try:
                # Use Gemini native video analysis
                result = await self.gemini_service.analyze_video_native(
                    temp_video_path,
                    query="Provide comprehensive visual analysis including objects, scenes, people, actions, and key moments"
                )
                
                # Parse the analysis to extract moments
                moments = self._extract_moments_from_analysis(result.get('analysis', ''))
                
                return {
                    "visual_analysis": result.get('analysis', ''),
                    "moments": moments,
                    "method": "direct_video_analysis"
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                    
        except Exception as e:
            logger.error(f"Error in direct Gemini analysis: {e}")
            return {}
    
    async def _extract_key_frames_smart(
        self,
        video_id: str,
        analysis_data: Dict
    ) -> List[Dict]:
        """Extract only key frames based on the analysis"""
        try:
            key_moments = []
            
            # Extract key timestamps from sections
            if 'sections' in analysis_data:
                for section in analysis_data['sections']:
                    # Add start of each section
                    start_time = self._parse_timestamp(section.get('start_time', '0:00'))
                    key_moments.append({
                        'timestamp': start_time,
                        'reason': f"Start of section: {section.get('title', 'Unknown')}"
                    })
            
            # Extract from visual moments if available
            if 'visual_moments' in analysis_data:
                for moment in analysis_data['visual_moments'][:10]:  # Limit to 10
                    key_moments.append({
                        'timestamp': moment.get('timestamp', 0),
                        'reason': moment.get('description', 'Visual moment')
                    })
            
            # Remove duplicates and sort
            seen_timestamps = set()
            unique_moments = []
            for moment in sorted(key_moments, key=lambda x: x['timestamp']):
                timestamp = int(moment['timestamp'])  # Round to nearest second
                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    unique_moments.append(moment)
            
            # Download frames for these moments only
            frames = await self._download_specific_frames(video_id, unique_moments[:20])  # Max 20 frames
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting key frames: {e}")
            return []
    
    async def _download_specific_frames(
        self,
        video_id: str,
        moments: List[Dict]
    ) -> List[Dict]:
        """Download specific frames at given timestamps"""
        frames = []
        
        try:
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp()
            
            # Download frames in parallel
            frame_tasks = []
            for i, moment in enumerate(moments):
                task = asyncio.create_task(
                    self._download_frame_at_timestamp(
                        video_id, 
                        moment['timestamp'], 
                        i, 
                        temp_dir
                    )
                )
                frame_tasks.append(task)
            
            # Gather results
            frame_results = await asyncio.gather(*frame_tasks)
            
            # Process results
            for i, (frame_path, moment) in enumerate(zip(frame_results, moments)):
                if frame_path and os.path.exists(frame_path):
                    # Analyze frame with Gemini
                    analysis = await self.gemini_service.analyze_frame(
                        frame_path,
                        context=moment.get('reason', ''),
                        timestamp=moment['timestamp']
                    )
                    
                    frames.append({
                        'timestamp': moment['timestamp'],
                        'frame_path': frame_path,
                        'frame_number': i,
                        'description': analysis.get('description', ''),
                        'reason': moment.get('reason', '')
                    })
            
            return frames
            
        except Exception as e:
            logger.error(f"Error downloading specific frames: {e}")
            return []
    
    async def _download_frame_at_timestamp(
        self,
        video_id: str,
        timestamp: float,
        frame_index: int,
        output_dir: str
    ) -> Optional[str]:
        """Download a single frame at specific timestamp using yt-dlp"""
        try:
            output_path = os.path.join(output_dir, f"frame_{video_id}_{frame_index}_{int(timestamp)}.jpg")
            
            # Use yt-dlp to download frame
            cmd = [
                'yt-dlp',
                '--skip-download',
                '--write-thumbnail',
                '--convert-thumbnails', 'jpg',
                '-o', output_path.replace('.jpg', ''),
                f'--force-keyframes-at-cuts',
                f'--download-sections', f'*{int(timestamp)}-{int(timestamp)+1}',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            # Try alternative method using ffmpeg if available
            alt_cmd = [
                'ffmpeg',
                '-ss', str(timestamp),
                '-i', f'https://www.youtube.com/watch?v={video_id}',
                '-frames:v', '1',
                '-q:v', '2',
                output_path
            ]
            
            try:
                # First try yt-dlp method
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(cmd, capture_output=True, text=True)
                )
                
                if os.path.exists(output_path):
                    return output_path
                    
            except:
                pass
            
            # If yt-dlp fails, we'll need to implement frame extraction differently
            logger.warning(f"Could not extract frame at {timestamp}s for video {video_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading frame: {e}")
            return None
    
    async def _get_video_info_enhanced(self, video_id: str) -> Dict:
        """Get enhanced video information"""
        try:
            info = await self.youtube_service.get_video_info(video_id)
            
            # Add additional metadata if needed
            info['video_id'] = video_id
            info['url'] = f"https://www.youtube.com/watch?v={video_id}"
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {
                "id": video_id,
                "title": f"YouTube Video {video_id}",
                "duration": 0
            }
    
    async def _get_transcript_with_timing(self, video_id: str) -> Dict:
        """Get transcript with accurate timing information"""
        try:
            transcript_data = self.youtube_service.get_transcript(video_id)
            return transcript_data
        except Exception as e:
            logger.error(f"Error getting transcript: {e}")
            return {
                "text": "",
                "transcript": [],
                "language": "unknown"
            }
    
    async def _analyze_transcript_content(
        self,
        transcript_data: Dict,
        video_info: Dict
    ) -> Dict:
        """Analyze transcript content for insights"""
        try:
            result = await self.gemini_service.analyze_video_content(
                transcript_data.get('text', ''),
                video_info
            )
            
            return {
                "content_analysis": result.get('analysis', ''),
                "method": "transcript_analysis"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {e}")
            return {}
    
    async def _generate_smart_sections(
        self,
        transcript_data: Dict,
        video_info: Dict
    ) -> Dict:
        """Generate smart video sections using AI"""
        try:
            sections = await self.gemini_service.generate_video_sections(
                transcript_data.get('text', ''),
                video_info
            )
            
            return {
                "sections": sections
            }
            
        except Exception as e:
            logger.error(f"Error generating sections: {e}")
            return {"sections": []}
    
    async def _identify_relevant_segments(
        self,
        video_id: str,
        search_query: str,
        cached_analysis: Optional[Dict]
    ) -> List[Dict]:
        """Identify video segments relevant to the search query"""
        segments = []
        
        try:
            # Use transcript data to identify relevant timestamps
            if cached_analysis and 'transcript_data' in cached_analysis:
                transcript_entries = cached_analysis['transcript_data']
                
                # Simple keyword matching - in production, use semantic search
                query_words = search_query.lower().split()
                
                for entry in transcript_entries:
                    text = entry.get('text', '').lower()
                    if any(word in text for word in query_words):
                        segments.append({
                            'start': entry.get('start', 0),
                            'end': entry.get('start', 0) + entry.get('duration', 5),
                            'text': entry.get('text', ''),
                            'relevance': sum(1 for word in query_words if word in text)
                        })
                
                # Sort by relevance
                segments.sort(key=lambda x: x['relevance'], reverse=True)
                
                # Merge nearby segments
                merged_segments = []
                for segment in segments[:20]:  # Top 20 segments
                    if not merged_segments:
                        merged_segments.append(segment)
                    else:
                        last_segment = merged_segments[-1]
                        if segment['start'] - last_segment['end'] < 10:  # Within 10 seconds
                            last_segment['end'] = segment['end']
                            last_segment['text'] += ' ' + segment['text']
                        else:
                            merged_segments.append(segment)
                
                return merged_segments[:10]  # Return top 10 merged segments
                
        except Exception as e:
            logger.error(f"Error identifying segments: {e}")
        
        return segments
    
    async def _process_video_segment(
        self,
        video_id: str,
        segment: Dict,
        search_query: str,
        segment_index: int
    ) -> List[Dict]:
        """Process a video segment for visual search"""
        matches = []
        
        try:
            # Extract 2-3 frames from the segment
            segment_duration = segment['end'] - segment['start']
            timestamps = [
                segment['start'],
                segment['start'] + segment_duration / 2,
                segment['end'] - 1
            ]
            
            for timestamp in timestamps:
                frame_path = await self._download_frame_at_timestamp(
                    video_id,
                    timestamp,
                    f"{segment_index}_{int(timestamp)}",
                    tempfile.gettempdir()
                )
                
                if frame_path and os.path.exists(frame_path):
                    # Analyze frame for search query
                    analysis = await self.gemini_service.analyze_frame(
                        frame_path,
                        context=f"Search for: {search_query}",
                        timestamp=timestamp
                    )
                    
                    # Calculate relevance score
                    description = analysis.get('description', '').lower()
                    query_words = search_query.lower().split()
                    relevance_score = sum(1 for word in query_words if word in description) / len(query_words)
                    
                    if relevance_score > 0.3:  # Threshold for relevance
                        matches.append({
                            'timestamp': timestamp,
                            'end_timestamp': timestamp + 5,
                            'description': analysis.get('description', ''),
                            'relevance_score': relevance_score,
                            'frame_path': frame_path,
                            'segment_text': segment.get('text', '')
                        })
                    
                    # Clean up frame
                    os.remove(frame_path)
            
        except Exception as e:
            logger.error(f"Error processing segment: {e}")
        
        return matches
    
    async def _search_cached_visual_data(
        self,
        visual_moments: List[Dict],
        search_query: str
    ) -> List[Dict]:
        """Search within cached visual analysis data"""
        matches = []
        query_lower = search_query.lower()
        query_words = query_lower.split()
        
        for moment in visual_moments:
            description = moment.get('description', '').lower()
            
            # Simple keyword matching
            match_count = sum(1 for word in query_words if word in description)
            if match_count > 0:
                relevance_score = match_count / len(query_words)
                matches.append({
                    'timestamp': moment.get('timestamp', 0),
                    'end_timestamp': moment.get('timestamp', 0) + 5,
                    'description': moment.get('description', ''),
                    'relevance_score': relevance_score,
                    'cached': True
                })
        
        # Sort by relevance
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        return matches[:10]
    
    async def _download_low_quality_video(
        self,
        video_id: str,
        max_duration: int = 300
    ) -> Optional[str]:
        """Download low quality video for quick analysis"""
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Download lowest quality video
            cmd = [
                'yt-dlp',
                '-f', 'worst[ext=mp4]/worst',
                '-o', temp_path,
                '--max-filesize', '50M',  # Max 50MB
                '--no-playlist',
                '--quiet',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            if max_duration:
                cmd.extend(['--match-filter', f'duration<{max_duration}'])
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True)
            )
            
            if result.returncode == 0 and os.path.exists(temp_path):
                return temp_path
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return None
                
        except Exception as e:
            logger.error(f"Error downloading low quality video: {e}")
            return None
    
    def _check_gemini_youtube_support(self) -> bool:
        """Check if Gemini supports direct YouTube URL analysis"""
        # For now, return False as we need to download
        # In future with Gemini 2.5, this might return True
        return False
    
    def _extract_moments_from_analysis(self, analysis_text: str) -> List[Dict]:
        """Extract moment information from Gemini analysis text"""
        moments = []
        
        try:
            # Parse analysis for timestamp mentions
            import re
            
            # Look for patterns like [00:30] or at 0:30 or @30s
            timestamp_patterns = [
                r'\[(\d{1,2}:\d{2})\]',
                r'at (\d{1,2}:\d{2})',
                r'@(\d+)s',
                r'(\d{1,2}:\d{2})'
            ]
            
            lines = analysis_text.split('\n')
            for line in lines:
                for pattern in timestamp_patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        timestamp = self._parse_timestamp(match)
                        if timestamp is not None:
                            moments.append({
                                'timestamp': timestamp,
                                'description': line.strip()
                            })
                            break
            
        except Exception as e:
            logger.error(f"Error extracting moments: {e}")
        
        return moments
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string to seconds"""
        try:
            if isinstance(timestamp_str, (int, float)):
                return float(timestamp_str)
            
            if 's' in str(timestamp_str):
                return float(timestamp_str.replace('s', ''))
            
            parts = str(timestamp_str).split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
        except:
            pass
        
        return 0.0
    
    def _generate_default_segments(self, duration: float) -> List[Dict]:
        """Generate default segments for video"""
        segments = []
        segment_duration = 30  # 30 second segments
        
        current_time = 0
        while current_time < duration:
            segments.append({
                'start': current_time,
                'end': min(current_time + segment_duration, duration),
                'text': f"Segment at {current_time}s",
                'relevance': 1
            })
            current_time += segment_duration
        
        return segments
    
    async def _update_progress(
        self,
        video_id: str,
        percentage: int,
        message: str
    ):
        """Update progress for a video processing task"""
        if video_id in self.progress_callbacks:
            callback = self.progress_callbacks[video_id]
            try:
                await callback(percentage, message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")