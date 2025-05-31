"""
Scene Change Detection Service
Intelligently selects ~10% of frames as keyframes based on visual scene changes
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SceneChangeResult:
    """Result from scene change detection"""
    frame_id: str
    timestamp: float
    frame_path: str
    scene_change_score: float
    is_keyframe: bool
    scene_id: int
    transition_type: str  # 'cut', 'fade', 'dissolve', 'static'

class SceneDetector:
    """Advanced scene change detection for keyframe selection"""
    
    def __init__(self, keyframe_percentage: float = 0.1):
        self.keyframe_percentage = keyframe_percentage
        self.scene_threshold = 0.3
        self.fade_threshold = 0.15
        self.static_threshold = 0.05
        
        # Scene detection parameters
        self.histogram_bins = 64
        self.edge_threshold = 100
        self.motion_threshold = 0.2
        
    def _calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between two frames"""
        try:
            # Convert to HSV for better color representation
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            hist1_h = cv2.calcHist([hsv1], [0], None, [self.histogram_bins], [0, 180])
            hist1_s = cv2.calcHist([hsv1], [1], None, [self.histogram_bins], [0, 256])
            hist1_v = cv2.calcHist([hsv1], [2], None, [self.histogram_bins], [0, 256])
            
            hist2_h = cv2.calcHist([hsv2], [0], None, [self.histogram_bins], [0, 180])
            hist2_s = cv2.calcHist([hsv2], [1], None, [self.histogram_bins], [0, 256])
            hist2_v = cv2.calcHist([hsv2], [2], None, [self.histogram_bins], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1_h, hist1_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist1_s, hist1_s, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist1_v, hist1_v, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2_h, hist2_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2_s, hist2_s, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2_v, hist2_v, 0, 1, cv2.NORM_MINMAX)
            
            # Calculate correlation for each channel
            corr_h = cv2.compareHist(hist1_h, hist2_h, cv2.HISTCMP_CORREL)
            corr_s = cv2.compareHist(hist1_s, hist2_s, cv2.HISTCMP_CORREL)
            corr_v = cv2.compareHist(hist1_v, hist2_v, cv2.HISTCMP_CORREL)
            
            # Weighted average (hue is most important for scene changes)
            correlation = 0.5 * corr_h + 0.3 * corr_s + 0.2 * corr_v
            
            # Convert to difference score
            return 1.0 - max(0, correlation)
            
        except Exception as e:
            logger.error(f"Error calculating histogram difference: {e}")
            return 0.5
    
    def _calculate_edge_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate edge-based difference between frames"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
            gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
            
            # Detect edges using Canny
            edges1 = cv2.Canny(gray1, self.edge_threshold, self.edge_threshold * 2)
            edges2 = cv2.Canny(gray2, self.edge_threshold, self.edge_threshold * 2)
            
            # Calculate difference in edge maps
            edge_diff = cv2.absdiff(edges1, edges2)
            
            # Calculate percentage of different pixels
            total_pixels = edge_diff.shape[0] * edge_diff.shape[1]
            different_pixels = np.count_nonzero(edge_diff)
            
            return different_pixels / total_pixels
            
        except Exception as e:
            logger.error(f"Error calculating edge difference: {e}")
            return 0.5
    
    def _calculate_motion_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate motion-based difference using optical flow"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, 
                corners=cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10),
                nextPts=None
            )[0]
            
            if flow is not None and len(flow) > 0:
                # Calculate average motion magnitude
                motion_magnitude = np.mean(np.linalg.norm(flow, axis=1))
                return min(1.0, motion_magnitude / 50.0)  # Normalize to 0-1
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating motion difference: {e}")
            return 0.0
    
    def _detect_transition_type(self, hist_diff: float, edge_diff: float, motion_diff: float) -> str:
        """Detect the type of scene transition"""
        if hist_diff > self.scene_threshold and edge_diff > self.scene_threshold:
            return 'cut'  # Hard cut
        elif hist_diff > self.fade_threshold and motion_diff < self.motion_threshold:
            return 'fade'  # Fade transition
        elif hist_diff > self.fade_threshold and edge_diff < self.scene_threshold:
            return 'dissolve'  # Dissolve transition
        else:
            return 'static'  # No significant change
    
    def detect_scene_changes(self, frame_paths: List[str], timestamps: List[float]) -> List[SceneChangeResult]:
        """Detect scene changes in a sequence of frames"""
        if len(frame_paths) != len(timestamps):
            raise ValueError("Frame paths and timestamps must have the same length")
        
        results = []
        scene_id = 0
        
        for i, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps)):
            frame_id = f"frame_{i}"
            
            if i == 0:
                # First frame is always a scene change
                results.append(SceneChangeResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_path=frame_path,
                    scene_change_score=1.0,
                    is_keyframe=True,
                    scene_id=scene_id,
                    transition_type='cut'
                ))
                continue
            
            try:
                # Load current and previous frames
                current_frame = cv2.imread(frame_path)
                previous_frame = cv2.imread(frame_paths[i-1])
                
                if current_frame is None or previous_frame is None:
                    logger.warning(f"Could not load frame: {frame_path}")
                    results.append(SceneChangeResult(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        frame_path=frame_path,
                        scene_change_score=0.5,
                        is_keyframe=False,
                        scene_id=scene_id,
                        transition_type='static'
                    ))
                    continue
                
                # Resize frames for faster processing
                height, width = 224, 224
                current_frame = cv2.resize(current_frame, (width, height))
                previous_frame = cv2.resize(previous_frame, (width, height))
                
                # Calculate different types of differences
                hist_diff = self._calculate_histogram_difference(previous_frame, current_frame)
                edge_diff = self._calculate_edge_difference(previous_frame, current_frame)
                motion_diff = self._calculate_motion_difference(previous_frame, current_frame)
                
                # Weighted combination of differences
                scene_change_score = (0.5 * hist_diff + 0.3 * edge_diff + 0.2 * motion_diff)
                
                # Detect transition type
                transition_type = self._detect_transition_type(hist_diff, edge_diff, motion_diff)
                
                # Determine if this is a new scene
                is_scene_change = scene_change_score > self.scene_threshold
                if is_scene_change:
                    scene_id += 1
                
                results.append(SceneChangeResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_path=frame_path,
                    scene_change_score=scene_change_score,
                    is_keyframe=False,  # Will be determined later
                    scene_id=scene_id,
                    transition_type=transition_type
                ))
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {e}")
                results.append(SceneChangeResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    frame_path=frame_path,
                    scene_change_score=0.5,
                    is_keyframe=False,
                    scene_id=scene_id,
                    transition_type='static'
                ))
        
        return results
    
    def select_keyframes(self, scene_results: List[SceneChangeResult]) -> List[SceneChangeResult]:
        """Select ~10% of frames as keyframes based on scene change scores"""
        if not scene_results:
            return []
        
        # Calculate target number of keyframes
        target_keyframes = max(1, int(len(scene_results) * self.keyframe_percentage))
        
        # Sort by scene change score (descending)
        sorted_results = sorted(scene_results, key=lambda x: x.scene_change_score, reverse=True)
        
        # Select top keyframes, but ensure we have at least one per scene
        keyframes = []
        selected_scenes = set()
        
        # First, select the highest scoring frame from each scene
        for result in sorted_results:
            if result.scene_id not in selected_scenes:
                result.is_keyframe = True
                keyframes.append(result)
                selected_scenes.add(result.scene_id)
                
                if len(keyframes) >= target_keyframes:
                    break
        
        # If we still need more keyframes, select additional high-scoring frames
        if len(keyframes) < target_keyframes:
            for result in sorted_results:
                if not result.is_keyframe and len(keyframes) < target_keyframes:
                    result.is_keyframe = True
                    keyframes.append(result)
        
        # Update the original results
        keyframe_ids = {kf.frame_id for kf in keyframes}
        for result in scene_results:
            result.is_keyframe = result.frame_id in keyframe_ids
        
        logger.info(f"Selected {len(keyframes)} keyframes from {len(scene_results)} total frames "
                   f"({len(selected_scenes)} scenes detected)")
        
        return keyframes
    
    def get_scene_statistics(self, scene_results: List[SceneChangeResult]) -> Dict[str, any]:
        """Get statistics about scene detection results"""
        if not scene_results:
            return {}
        
        # Count scenes and transitions
        scene_count = len(set(r.scene_id for r in scene_results))
        transition_counts = {}
        for result in scene_results:
            transition_counts[result.transition_type] = transition_counts.get(result.transition_type, 0) + 1
        
        # Calculate average scene change scores
        scores = [r.scene_change_score for r in scene_results]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        # Count keyframes
        keyframe_count = sum(1 for r in scene_results if r.is_keyframe)
        keyframe_percentage = keyframe_count / len(scene_results) * 100
        
        return {
            "total_frames": len(scene_results),
            "scene_count": scene_count,
            "keyframe_count": keyframe_count,
            "keyframe_percentage": round(keyframe_percentage, 1),
            "avg_scene_change_score": round(avg_score, 3),
            "max_scene_change_score": round(max_score, 3),
            "min_scene_change_score": round(min_score, 3),
            "transition_types": transition_counts,
            "avg_frames_per_scene": round(len(scene_results) / scene_count, 1) if scene_count > 0 else 0
        }
