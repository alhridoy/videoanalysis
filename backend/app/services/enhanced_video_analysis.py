"""
Enhanced Video Analysis Service
Implements Google Gemini 2.5 native video understanding capabilities
"""

import logging
import os
import asyncio
from typing import List, Dict, Optional
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

class EnhancedVideoAnalysis:
    """Enhanced video analysis using Gemini 2.5 native video capabilities"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.5 Pro for best video understanding
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro-preview-0506",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
    
    async def analyze_video_moments_native(self, video_path: str, context: str = "") -> List[Dict]:
        """
        Native video moment retrieval using Gemini 2.5 capabilities
        Mimics Google's demo: identifying 16+ distinct segments with audio-visual cues
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Upload video to Gemini
            logger.info(f"Uploading video for analysis: {video_path}")
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise Exception("Video processing failed")
            
            prompt = f"""
            Analyze this video using advanced moment retrieval capabilities similar to Google's Gemini 2.5 demo.
            
            TASK: Identify distinct moments/segments in this video using both audio and visual cues.
            
            Context: {context}
            
            For each moment, analyze:
            1. **Visual Changes**: Scene transitions, new objects, people, settings
            2. **Audio Cues**: Topic changes, speaker changes, music changes
            3. **Content Shifts**: New subjects, demonstrations, conclusions
            4. **Temporal Boundaries**: Clear start/end points for each segment
            
            Provide 8-20 moments covering the entire video timeline with:
            - timestamp: precise time in seconds
            - title: descriptive title (max 50 chars)
            - description: what happens in this moment (100-200 chars)
            - category: topic_change|demonstration|conclusion|interaction|transition
            - confidence: 0.0-1.0 confidence score
            - visual_cues: what you see changing
            - audio_cues: what you hear changing (if any)
            - importance: high|medium|low
            
            Return as JSON array. Be precise with timestamps and comprehensive in coverage.
            """
            
            response = self.model.generate_content([video_file, prompt])
            
            # Parse response
            moments = self._parse_json_response(response.text)
            
            # Clean up
            genai.delete_file(video_file.name)
            
            logger.info(f"Extracted {len(moments)} moments from video")
            return moments
            
        except Exception as e:
            logger.error(f"Error in native video moment analysis: {e}")
            return []
    
    async def temporal_counting_native(self, video_path: str, query: str, context: str = "") -> Dict:
        """
        Native temporal counting using Gemini 2.5 capabilities
        Mimics Google's demo: counting 17 phone usage instances with precision
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Upload video to Gemini
            logger.info(f"Uploading video for temporal counting: {video_path}")
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise Exception("Video processing failed")
            
            prompt = f"""
            Perform precise temporal counting analysis on this video similar to Google's Gemini 2.5 demo.
            
            COUNTING QUERY: "{query}"
            Context: {context}
            
            TASK: Count every occurrence of the specified item/action throughout the entire video.
            
            Instructions:
            1. Watch the entire video carefully
            2. Identify EVERY instance where "{query}" occurs
            3. Note the precise timestamp for each occurrence
            4. Provide context for each instance
            5. Analyze temporal patterns and clustering
            
            For each occurrence, provide:
            - timestamp: exact time in seconds
            - description: what specifically happens
            - confidence: 0.0-1.0 confidence this matches the query
            - context: surrounding activity/scene
            - duration: how long the occurrence lasts (if applicable)
            
            Also analyze:
            - Total count
            - Temporal patterns (clustering, frequency changes)
            - Peak activity periods
            - Notable observations
            
            Return as JSON with:
            {{
                "total_count": number,
                "occurrences": [list of occurrence objects],
                "patterns": "description of temporal patterns",
                "peak_periods": [list of high-activity time ranges],
                "confidence_score": overall_confidence,
                "notes": "additional observations"
            }}
            """
            
            response = self.model.generate_content([video_file, prompt])
            
            # Parse response
            result = self._parse_json_response(response.text)
            
            # Clean up
            genai.delete_file(video_file.name)
            
            logger.info(f"Temporal counting completed: {result.get('total_count', 0)} occurrences found")
            return result
            
        except Exception as e:
            logger.error(f"Error in native temporal counting: {e}")
            return {
                "total_count": 0,
                "occurrences": [],
                "patterns": "Analysis failed",
                "peak_periods": [],
                "confidence_score": 0.0,
                "notes": str(e)
            }
    
    async def video_to_interactive_app(self, video_path: str, app_type: str = "learning") -> Dict:
        """
        Generate interactive application from video (like Google's demo)
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Upload video
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            prompt = f"""
            Analyze this video and create a detailed specification for an interactive {app_type} application.
            
            Based on the video content, design an application that:
            1. Reinforces key concepts from the video
            2. Provides interactive elements for engagement
            3. Includes visual components that relate to video content
            4. Offers educational value and user interaction
            
            Provide:
            - App concept and purpose
            - Key features and interactions
            - Visual design suggestions
            - Technical implementation approach
            - Code structure recommendations
            
            Return as JSON with detailed specifications.
            """
            
            response = self.model.generate_content([video_file, prompt])
            result = self._parse_json_response(response.text)
            
            # Clean up
            genai.delete_file(video_file.name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in video-to-app generation: {e}")
            return {"error": str(e)}
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON response with multiple fallback strategies"""
        import json
        
        try:
            # Strategy 1: Direct JSON parsing
            return json.loads(response_text.strip())
        except:
            pass
        
        try:
            # Strategy 2: Extract from markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    return json.loads(response_text[start:end].strip())
        except:
            pass
        
        try:
            # Strategy 3: Find JSON object/array
            start = max(response_text.find('{'), response_text.find('['))
            if start != -1:
                if response_text[start] == '{':
                    end = response_text.rfind('}') + 1
                else:
                    end = response_text.rfind(']') + 1
                
                if end > start:
                    return json.loads(response_text[start:end])
        except:
            pass
        
        # Fallback
        return {"error": "Could not parse response", "raw_response": response_text}
