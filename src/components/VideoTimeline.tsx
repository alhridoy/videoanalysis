import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize, SkipBack, SkipForward, Layers } from 'lucide-react';

interface ClipMarker {
  id: string;
  startTime: number;
  endTime: number;
  title: string;
  color: string;
  confidence: number;
  type: 'search' | 'scene' | 'moment' | 'bookmark';
}

interface ThumbnailPreview {
  timestamp: number;
  url: string;
}

interface VideoTimelineProps {
  videoUrl: string;
  duration: number;
  currentTime: number;
  isPlaying: boolean;
  volume: number;
  clipMarkers: ClipMarker[];
  thumbnails: ThumbnailPreview[];
  onTimeChange: (time: number) => void;
  onPlayPause: () => void;
  onVolumeChange: (volume: number) => void;
  onSeek: (time: number) => void;
  onMarkerClick?: (marker: ClipMarker) => void;
}

export const VideoTimeline: React.FC<VideoTimelineProps> = ({
  videoUrl,
  duration,
  currentTime,
  isPlaying,
  volume,
  clipMarkers,
  thumbnails,
  onTimeChange,
  onPlayPause,
  onVolumeChange,
  onSeek,
  onMarkerClick
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [showThumbnail, setShowThumbnail] = useState(false);
  const [thumbnailPosition, setThumbnailPosition] = useState({ x: 0, time: 0 });
  const [hoveredMarker, setHoveredMarker] = useState<ClipMarker | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);

  // Format time for display
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle timeline click/drag
  const handleTimelineInteraction = (e: React.MouseEvent) => {
    if (!timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const newTime = percentage * duration;

    onSeek(newTime);
  };

  // Handle mouse move for thumbnail preview
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const time = percentage * duration;

    setThumbnailPosition({ x: e.clientX, time });
    setShowThumbnail(true);
  };

  const handleMouseLeave = () => {
    setShowThumbnail(false);
    setHoveredMarker(null);
  };

  // Get thumbnail for specific time
  const getThumbnailForTime = (time: number): string | null => {
    if (!thumbnails.length) return null;

    // Find closest thumbnail
    const closest = thumbnails.reduce((prev, curr) => 
      Math.abs(curr.timestamp - time) < Math.abs(prev.timestamp - time) ? curr : prev
    );

    return closest.url;
  };

  // Get marker color classes
  const getMarkerColorClass = (type: string, color: string): string => {
    const baseClasses = "absolute top-0 bottom-0 opacity-70 hover:opacity-90 transition-opacity cursor-pointer";
    
    switch (type) {
      case 'search':
        return `${baseClasses} bg-blue-500 border-l-2 border-blue-400`;
      case 'scene':
        return `${baseClasses} bg-green-500 border-l-2 border-green-400`;
      case 'moment':
        return `${baseClasses} bg-purple-500 border-l-2 border-purple-400`;
      case 'bookmark':
        return `${baseClasses} bg-yellow-500 border-l-2 border-yellow-400`;
      default:
        return `${baseClasses} bg-gray-500 border-l-2 border-gray-400`;
    }
  };

  // Calculate marker positions
  const getMarkerStyle = (marker: ClipMarker) => {
    const startPercent = (marker.startTime / duration) * 100;
    const widthPercent = ((marker.endTime - marker.startTime) / duration) * 100;
    
    return {
      left: `${startPercent}%`,
      width: `${Math.max(0.5, widthPercent)}%` // Minimum width for visibility
    };
  };

  return (
    <div className="bg-gray-900/90 rounded-lg border border-gray-700 p-4">
      {/* Video Controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {/* Play/Pause */}
          <button
            onClick={onPlayPause}
            className="w-10 h-10 bg-blue-600 hover:bg-blue-500 rounded-full flex items-center justify-center transition-colors"
          >
            {isPlaying ? (
              <Pause className="w-5 h-5 text-white" />
            ) : (
              <Play className="w-5 h-5 text-white ml-0.5" />
            )}
          </button>

          {/* Skip Controls */}
          <button
            onClick={() => onSeek(Math.max(0, currentTime - 10))}
            className="w-8 h-8 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center transition-colors"
          >
            <SkipBack className="w-4 h-4 text-white" />
          </button>

          <button
            onClick={() => onSeek(Math.min(duration, currentTime + 10))}
            className="w-8 h-8 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center transition-colors"
          >
            <SkipForward className="w-4 h-4 text-white" />
          </button>

          {/* Time Display */}
          <div className="text-sm text-gray-300 font-mono">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Volume Control */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => onVolumeChange(volume > 0 ? 0 : 1)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              {volume > 0 ? (
                <Volume2 className="w-5 h-5" />
              ) : (
                <VolumeX className="w-5 h-5" />
              )}
            </button>
            
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
              className="w-20 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
            />
          </div>

          {/* Markers Legend */}
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-gray-400" />
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-blue-500 rounded-sm"></div>
              <span className="text-xs text-gray-400">Search</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-500 rounded-sm"></div>
              <span className="text-xs text-gray-400">Scene</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-purple-500 rounded-sm"></div>
              <span className="text-xs text-gray-400">Moment</span>
            </div>
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="relative">
        {/* Timeline Track */}
        <div
          ref={timelineRef}
          className="relative h-6 bg-gray-700 rounded-lg cursor-pointer overflow-hidden"
          onClick={handleTimelineInteraction}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Progress Bar */}
          <div
            ref={progressRef}
            className="absolute top-0 left-0 h-full bg-blue-600 rounded-lg transition-all duration-100"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          />

          {/* Clip Markers */}
          {clipMarkers.map((marker) => (
            <div
              key={marker.id}
              className={getMarkerColorClass(marker.type, marker.color)}
              style={getMarkerStyle(marker)}
              onClick={(e) => {
                e.stopPropagation();
                onMarkerClick?.(marker);
                onSeek(marker.startTime);
              }}
              onMouseEnter={() => setHoveredMarker(marker)}
              onMouseLeave={() => setHoveredMarker(null)}
            />
          ))}

          {/* Current Time Indicator */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg z-10"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          >
            <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-white rounded-full shadow-lg" />
          </div>
        </div>

        {/* Thumbnail Preview */}
        {showThumbnail && (
          <div
            className="absolute bottom-full mb-2 pointer-events-none z-20"
            style={{ 
              left: `${thumbnailPosition.x}px`,
              transform: 'translateX(-50%)'
            }}
          >
            <div className="bg-black rounded-lg p-2 shadow-xl border border-gray-600">
              <div className="w-32 h-18 bg-gray-800 rounded overflow-hidden mb-2">
                {getThumbnailForTime(thumbnailPosition.time) ? (
                  <img
                    src={getThumbnailForTime(thumbnailPosition.time)!}
                    alt="Preview"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Play className="w-6 h-6 text-gray-400" />
                  </div>
                )}
              </div>
              <div className="text-xs text-white text-center font-mono">
                {formatTime(thumbnailPosition.time)}
              </div>
            </div>
          </div>
        )}

        {/* Marker Tooltip */}
        {hoveredMarker && (
          <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 pointer-events-none z-20">
            <div className="bg-black rounded-lg p-3 shadow-xl border border-gray-600 max-w-xs">
              <div className="text-sm font-medium text-white mb-1">
                {hoveredMarker.title}
              </div>
              <div className="text-xs text-gray-400 mb-2">
                {formatTime(hoveredMarker.startTime)} - {formatTime(hoveredMarker.endTime)}
              </div>
              <div className="flex items-center justify-between">
                <span className={`text-xs px-2 py-1 rounded ${
                  hoveredMarker.type === 'search' ? 'bg-blue-500/20 text-blue-300' :
                  hoveredMarker.type === 'scene' ? 'bg-green-500/20 text-green-300' :
                  hoveredMarker.type === 'moment' ? 'bg-purple-500/20 text-purple-300' :
                  'bg-yellow-500/20 text-yellow-300'
                }`}>
                  {hoveredMarker.type}
                </span>
                <span className="text-xs text-gray-400">
                  {Math.round(hoveredMarker.confidence * 100)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Timeline Markers Summary */}
      <div className="mt-3 flex items-center justify-between text-xs text-gray-400">
        <div className="flex items-center gap-4">
          <span>{clipMarkers.filter(m => m.type === 'search').length} search results</span>
          <span>{clipMarkers.filter(m => m.type === 'scene').length} scene changes</span>
          <span>{clipMarkers.filter(m => m.type === 'moment').length} key moments</span>
        </div>
        <div>
          Click markers to jump to specific moments
        </div>
      </div>
    </div>
  );
};
