import React, { useState, useEffect, useCallback } from 'react';
import { Search, Eye, Clock, Zap, Film, ChevronDown, ChevronUp, Users, Target, Play, ChevronLeft, ChevronRight, Grid3X3, BarChart3 } from 'lucide-react';
import { SearchResult, ClipResult, apiService } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

interface VisualSearchProps {
  videoId: number;
  videoUrl: string;
  onTimeJump: (time: number) => void;
}

const VisualSearch: React.FC<VisualSearchProps> = ({ videoId, onTimeJump }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchClips, setSearchClips] = useState<ClipResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [viewMode, setViewMode] = useState<'clips' | 'frames' | 'timeline'>('clips');
  const [directAnswer, setDirectAnswer] = useState<string>('');
  const [queryType, setQueryType] = useState<string>('');
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());
  const [currentClipIndex, setCurrentClipIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const { toast } = useToast();

  // Add keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!hasSearched || searchClips.length === 0) return;
      
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          navigateToClip(Math.max(0, currentClipIndex - 1));
          break;
        case 'ArrowRight':
          e.preventDefault();
          navigateToClip(Math.min(searchClips.length - 1, currentClipIndex + 1));
          break;
        case ' ':
          e.preventDefault();
          if (searchClips[currentClipIndex]) {
            onTimeJump(searchClips[currentClipIndex].start_time);
            setIsPlaying(!isPlaying);
          }
          break;
        case 'Enter':
          e.preventDefault();
          if (searchClips[currentClipIndex]) {
            onTimeJump(searchClips[currentClipIndex].start_time);
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentClipIndex, searchClips, hasSearched, isPlaying, onTimeJump]);

  const navigateToClip = useCallback((index: number) => {
    if (index >= 0 && index < searchClips.length) {
      setCurrentClipIndex(index);
      onTimeJump(searchClips[index].start_time);
    }
  }, [searchClips, onTimeJump]);

  const formatDuration = (seconds: number): string => {
    return `${Math.round(seconds * 10) / 10}s`;
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      // Clear results if search is empty
      setSearchResults([]);
      setSearchClips([]);
      setDirectAnswer('');
      setQueryType('');
      setHasSearched(false);
      setCurrentClipIndex(0);
      return;
    }

    setIsSearching(true);
    setHasSearched(true);
    setExpandedResults(new Set()); // Reset expanded results

    try {
      // Log the search attempt for debugging
      console.log(`🔍 Starting visual search for: "${searchQuery}" on video ID: ${videoId}`);
      
      // Use real API call to search video content with the native search enabled
      const response = await apiService.visualSearch(videoId, searchQuery.trim(), 10);
      
      console.log('📊 Visual search response:', response);
      
      // Add detailed debugging for "people" searches
      if (searchQuery.toLowerCase().includes('people') || searchQuery.toLowerCase().includes('person')) {
        console.log('👥 People search debug:', {
          query: searchQuery,
          videoId,
          processingMethod: response.processing_method,
          totalResults: response.total_results,
          hasClips: !!response.clips && response.clips.length > 0,
          hasResults: !!response.results && response.results.length > 0,
          clipsData: response.clips?.slice(0, 2), // First 2 clips for inspection
          resultsData: response.results?.slice(0, 2) // First 2 results for inspection
        });
      }

      setSearchResults(response.results || []);
      setSearchClips(response.clips || []);
      setDirectAnswer(response.direct_answer || '');
      setQueryType(response.query_type || '');
      setCurrentClipIndex(0); // Reset to first clip

      const hasResults = (response.results || []).length > 0 || (response.clips || []).length > 0;

      if (!hasResults) {
        // Provide more specific guidance based on processing method
        let processingMessage = "No visual content matching your query was found.";
        
        if (response.processing_method === "direct_frame_analysis") {
          processingMessage = `Direct frame analysis completed but found no matches for "${searchQuery}". Try more general terms or check if the content exists in your video.`;
        } else if (response.processing_method === "native_video_search") {
          processingMessage = "Native video search found no matches. The video may need more processing time or the content might not be visually detectable.";
        } else if (response.processing_method === "semantic_search") {
          processingMessage = "Semantic search found no matches. Try uploading the video again or use different keywords.";
        } else {
          processingMessage = "No matches found. Try keywords like 'person', 'microphone', 'red car', 'text', or 'background'.";
        }
        
        toast({
          title: "No results found", 
          description: processingMessage,
        });
        
        console.log('❌ No results found. Response details:', {
          totalResults: response.total_results,
          processingMethod: response.processing_method,
          clips: response.clips?.length || 0,
          results: response.results?.length || 0
        });
      } else {
        const totalResults = (response.clips || []).length + (response.results || []).length;
        const method = response.processing_method ? ` using ${response.processing_method}` : '';
        
        toast({
          title: "Search completed",
          description: `Found ${totalResults} ${totalResults === 1 ? 'result' : 'results'} for "${searchQuery}"${method}`,
        });
        
        console.log('✅ Search successful:', {
          query: searchQuery,
          totalResults,
          processingMethod: response.processing_method,
          clipsFound: response.clips?.length || 0,
          framesFound: response.results?.length || 0,
          directAnswer: response.direct_answer,
          queryType: response.query_type
        });
      }
    } catch (error) {
      console.error('❌ Visual search error:', error);
      
      // More detailed error handling
      let errorMessage = "Failed to search video content";
      if (error instanceof Error) {
        if (error.message.includes('404')) {
          errorMessage = "Video not found. Please make sure the video is uploaded and processed.";
        } else if (error.message.includes('500')) {
          errorMessage = "Server error during search. The video might need to be processed first.";
        } else {
          errorMessage = error.message;
        }
      }
      
      toast({
        title: "Search failed",
        description: errorMessage,
        variant: "destructive",
      });
      
      // Clear results on error
      setSearchResults([]);
      setSearchClips([]);
      setDirectAnswer('');
      setQueryType('');
    } finally {
      setIsSearching(false);
    }
  };

  // Add real-time search as user types (but only on manual submit, not auto-search)
  useEffect(() => {
    if (!searchQuery.trim()) {
      // Clear results if query is empty
      setSearchResults([]);
      setSearchClips([]);
      setDirectAnswer('');
      setQueryType('');
      setHasSearched(false);
      setCurrentClipIndex(0);
    }
  }, [searchQuery]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 90) return 'text-green-400';
    if (confidence >= 75) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const toggleExpandResult = (index: number) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedResults(newExpanded);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-lg font-semibold text-foreground mb-2">Visual Search</h3>
        <p className="text-sm text-muted-foreground">
          Search for objects, people, or scenes within video frames
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="space-y-4">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search for anything... (e.g., 'red car', 'tree', 'people', 'microphone')"
            className="w-full px-4 py-3 pl-12 pr-12 bg-input border border-border rounded-lg text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent transition-all"
            disabled={isSearching}
          />
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
          {searchQuery && (
            <button
              type="button"
              onClick={() => {
                setSearchQuery('');
                setSearchResults([]);
                setSearchClips([]);
                setDirectAnswer('');
                setQueryType('');
                setHasSearched(false);
                setCurrentClipIndex(0);
              }}
              className="absolute right-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground hover:text-foreground transition-colors"
              title="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        {/* Search Suggestions & Help */}
        {!hasSearched && (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {['red car', 'person', 'microphone', 'text on screen', 'background', 'people'].map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => setSearchQuery(suggestion)}
                  className="px-3 py-1.5 text-xs bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground rounded-full transition-all border border-border hover:border-primary/50"
                >
                  {suggestion}
                </button>
              ))}
            </div>
            
            <div className="bg-muted/30 rounded-lg p-3 text-xs text-muted-foreground">
              <div className="font-medium mb-2">💡 How Visual Search Works:</div>
              <div className="space-y-2">
                <div>• <strong>Native Video Search:</strong> Uses Gemini 2.5 to directly analyze video content</div>
                <div>• <strong>Object Detection:</strong> "car", "microphone", "phone", "person"</div>
                <div>• <strong>Color + Object:</strong> "red car", "blue shirt", "green background"</div>
                <div>• <strong>Scene Analysis:</strong> "background", "outdoor scene", "indoor setting"</div>
                <div>• <strong>Text Detection:</strong> "sign", "text on screen", "writing"</div>
                <div>• <strong>Counting:</strong> "how many people", "count cars"</div>
              </div>
              
              <div className="mt-3 pt-2 border-t border-border text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Powered by Gemini 2.5 Video Understanding</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={!searchQuery.trim() || isSearching}
          className="w-full bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed text-primary-foreground font-semibold py-3 px-6 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg flex items-center justify-center gap-2"
        >
          {isSearching ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-primary-foreground border-t-transparent"></div>
              Analyzing Frames...
            </>
          ) : (
            <>
              <Eye className="w-5 h-5" />
              Search Video
            </>
          )}
        </button>
      </form>

      {/* Search Results */}
      {hasSearched && (
        <div className="space-y-4">
          {isSearching ? (
            <div className="text-center py-8">
              <div className="animate-pulse space-y-4">
                <div className="w-16 h-16 bg-muted rounded-full mx-auto flex items-center justify-center">
                  <Zap className="w-8 h-8 text-muted-foreground" />
                </div>
                <p className="text-muted-foreground">
                  AI is analyzing video frames for "{searchQuery}"
                </p>
              </div>
            </div>
          ) : searchResults.length > 0 || searchClips.length > 0 ? (
            <>
              {/* Direct Answer Section */}
              {directAnswer && (
                <div className="mb-4 p-4 bg-muted/50 rounded-lg border border-border">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                      {queryType === 'counting' ? (
                        <Users className="w-4 h-4 text-primary-foreground" />
                      ) : (
                        <Target className="w-4 h-4 text-primary-foreground" />
                      )}
                    </div>
                    <div className="flex-1">
                      <h4 className="text-foreground font-medium mb-1">Direct Answer</h4>
                      <p className="text-muted-foreground text-sm leading-relaxed">{directAnswer}</p>
                      {queryType && (
                        <span className="inline-block mt-2 px-2 py-1 bg-muted rounded text-xs text-muted-foreground border border-border">
                          {queryType.replace('_', ' ')}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Results Header with View Toggle */}
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="text-foreground font-medium">
                    Found {searchClips.length > 0 ? `${searchClips.length} clips` : `${searchResults.length} matches`} for "{searchQuery}"
                  </h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    {searchClips.length > 0 
                      ? `${searchClips.length} video segments with ${searchResults.length} total frames detected`
                      : searchResults.length > 1 
                        ? 'Multiple instances detected throughout the video. Click any result to jump to that moment' 
                        : 'Click result to jump to that moment in the video'
                    }
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {searchClips.length > 0 && (
                    <button
                      onClick={() => setViewMode('clips')}
                      className={`px-3 py-1.5 rounded text-sm font-medium transition-all ${
                        viewMode === 'clips'
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      }`}
                    >
                      <Film className="w-4 h-4 inline mr-1" />
                      Clips ({searchClips.length})
                    </button>
                  )}
                  {searchClips.length > 0 && (
                    <button
                      onClick={() => setViewMode('timeline')}
                      className={`px-3 py-1.5 rounded text-sm font-medium transition-all ${
                        viewMode === 'timeline'
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      }`}
                    >
                      <BarChart3 className="w-4 h-4 inline mr-1" />
                      Timeline
                    </button>
                  )}
                  <button
                    onClick={() => setViewMode('frames')}
                    className={`px-3 py-1.5 rounded text-sm font-medium transition-all ${
                      viewMode === 'frames'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                    }`}
                  >
                    <Grid3X3 className="w-4 h-4 inline mr-1" />
                    Frames ({searchResults.length})
                  </button>
                </div>
              </div>

              {/* Clips View */}
              {viewMode === 'clips' && searchClips.length > 0 && (
                <div className="space-y-4">
                  {/* Clip Navigation */}
                  <div className="flex items-center justify-between bg-muted/30 rounded-lg p-3">
                    <div className="flex items-center gap-3">
                      <span className="text-sm text-muted-foreground">
                        Clip {currentClipIndex + 1} of {searchClips.length}
                      </span>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => navigateToClip(currentClipIndex - 1)}
                          disabled={currentClipIndex === 0}
                          className="p-1 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          title="Previous clip (←)"
                        >
                          <ChevronLeft className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => navigateToClip(currentClipIndex + 1)}
                          disabled={currentClipIndex === searchClips.length - 1}
                          className="p-1 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          title="Next clip (→)"
                        >
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Use ← → keys to navigate • Space to play/pause • Enter to jump
                    </div>
                  </div>

                  {/* Clips Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-500/50 scrollbar-track-transparent">
                    {searchClips.map((clip, index) => (
                      <div
                        key={index}
                        className={`group bg-card rounded-lg border transition-all duration-300 cursor-pointer ${
                          index === currentClipIndex
                            ? 'border-primary bg-primary/5 ring-2 ring-primary/20'
                            : 'border-border hover:border-primary/50 hover:bg-muted/30'
                        }`}
                        onClick={() => {
                          setCurrentClipIndex(index);
                          onTimeJump(clip.start_time);
                        }}
                      >
                        {/* Clip Thumbnail */}
                        <div className="relative aspect-video bg-muted/50 rounded-t-lg overflow-hidden">
                          {/* Thumbnail Image */}
                          <img 
                            src={`https://picsum.photos/400/225?random=${index + Math.random()}`}
                            alt={`Clip ${index + 1} thumbnail`}
                            className="w-full h-full object-cover transition-transform group-hover:scale-105"
                            onError={(e) => {
                              const target = e.currentTarget;
                              target.style.display = 'none';
                            }}
                          />
                          
                          {/* Play button overlay */}
                          <div className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity">
                            <div className="bg-primary/90 rounded-full p-3 shadow-lg">
                              <Play className="w-6 h-6 text-primary-foreground" />
                            </div>
                          </div>
                          
                          {/* Clip Info Overlay */}
                          <div className="absolute top-2 left-2 flex gap-1">
                            <span className="bg-black/80 text-white text-xs px-2 py-1 rounded font-medium">
                              #{index + 1}
                            </span>
                            <span className="bg-primary/90 text-primary-foreground text-xs px-2 py-1 rounded font-medium">
                              {formatDuration(clip.end_time - clip.start_time)}
                            </span>
                          </div>
                          
                          {/* Confidence Badge */}
                          <div className="absolute top-2 right-2">
                            <span className={`text-xs px-2 py-1 rounded font-medium ${
                              clip.confidence >= 90 
                                ? 'bg-green-500/90 text-white' 
                                : clip.confidence >= 75 
                                  ? 'bg-yellow-500/90 text-white'
                                  : 'bg-orange-500/90 text-white'
                            }`}>
                              {clip.confidence}%
                            </span>
                          </div>
                        </div>

                        {/* Clip Details */}
                        <div className="p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Clock className="w-3 h-3" />
                              <span>{formatTime(clip.start_time)} - {formatTime(clip.end_time)}</span>
                            </div>
                            <span className="text-xs text-muted-foreground">
                              {clip.frame_count} frames
                            </span>
                          </div>
                          
                          <p className="text-sm text-foreground leading-relaxed overflow-hidden" style={{
                            display: '-webkit-box',
                            WebkitBoxOrient: 'vertical',
                            WebkitLineClamp: 2
                          }}>
                            {clip.description}
                          </p>
                          
                          {/* Clip Actions */}
                          <div className="flex gap-2 mt-3">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onTimeJump(clip.start_time);
                              }}
                              className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground px-3 py-2 rounded text-xs font-medium transition-colors flex items-center justify-center gap-1"
                            >
                              <Play className="w-3 h-3" />
                              Jump to clip
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Timeline View */}
              {viewMode === 'timeline' && searchClips.length > 0 && (
                <div className="space-y-4">
                  <div className="bg-muted/30 rounded-lg p-4">
                    <h5 className="text-sm font-medium text-foreground mb-3">Clip Timeline</h5>
                    <div className="relative">
                      {/* Timeline bar */}
                      <div className="h-2 bg-muted rounded-full relative overflow-hidden">
                        {searchClips.map((clip, index) => {
                          const totalDuration = Math.max(...searchClips.map(c => c.end_time));
                          const startPercent = (clip.start_time / totalDuration) * 100;
                          const widthPercent = ((clip.end_time - clip.start_time) / totalDuration) * 100;
                          
                          return (
                            <div
                              key={index}
                              className={`absolute h-full rounded transition-all cursor-pointer ${
                                index === currentClipIndex
                                  ? 'bg-primary'
                                  : 'bg-primary/60 hover:bg-primary/80'
                              }`}
                              style={{
                                left: `${startPercent}%`,
                                width: `${widthPercent}%`
                              }}
                              onClick={() => {
                                setCurrentClipIndex(index);
                                onTimeJump(clip.start_time);
                              }}
                              title={`Clip ${index + 1}: ${formatTime(clip.start_time)} - ${formatTime(clip.end_time)}`}
                            />
                          );
                        })}
                      </div>
                      
                      {/* Timeline markers */}
                      <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                        <span>0:00</span>
                        <span>{formatTime(Math.max(...searchClips.map(c => c.end_time)))}</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Current clip details */}
                  {searchClips[currentClipIndex] && (
                    <div className="bg-card rounded-lg border p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h6 className="font-medium text-foreground mb-1">
                            Clip {currentClipIndex + 1}: {formatTime(searchClips[currentClipIndex].start_time)} - {formatTime(searchClips[currentClipIndex].end_time)}
                          </h6>
                          <p className="text-sm text-muted-foreground">
                            {searchClips[currentClipIndex].description}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">
                            {formatDuration(searchClips[currentClipIndex].end_time - searchClips[currentClipIndex].start_time)}
                          </span>
                          <span className={`text-xs font-medium px-2 py-1 rounded ${getConfidenceColor(searchClips[currentClipIndex].confidence)} bg-muted`}>
                            {searchClips[currentClipIndex].confidence}%
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex gap-2">
                        <button
                          onClick={() => onTimeJump(searchClips[currentClipIndex].start_time)}
                          className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded text-sm font-medium transition-colors flex items-center gap-2"
                        >
                          <Play className="w-4 h-4" />
                          Play this clip
                        </button>
                        <button
                          onClick={() => navigateToClip(currentClipIndex - 1)}
                          disabled={currentClipIndex === 0}
                          className="bg-muted hover:bg-accent text-muted-foreground hover:text-accent-foreground px-3 py-2 rounded text-sm font-medium transition-colors flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <ChevronLeft className="w-4 h-4" />
                          Previous
                        </button>
                        <button
                          onClick={() => navigateToClip(currentClipIndex + 1)}
                          disabled={currentClipIndex === searchClips.length - 1}
                          className="bg-muted hover:bg-accent text-muted-foreground hover:text-accent-foreground px-3 py-2 rounded text-sm font-medium transition-colors flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Next
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Frames View */}
              {viewMode === 'frames' && (
                <div className="max-h-96 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-gray-500/50 scrollbar-track-transparent">
                  {searchResults.map((result, index) => (
                    <div
                      key={index}
                      className="bg-card rounded-lg border border-border p-4 hover:bg-muted/50 transition-all duration-300 cursor-pointer group"
                      onClick={() => onTimeJump(result.timestamp)}
                    >
                      <div className="flex gap-4">
                        {/* Frame Number & Thumbnail */}
                        <div className="flex flex-col items-center gap-2 flex-shrink-0">
                          <div className="w-20 h-12 bg-muted rounded flex items-center justify-center relative overflow-hidden">
                            {result.frame_path ? (
                              <img
                                src={result.frame_path}
                                alt={`Frame at ${formatTime(result.timestamp)}`}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                  const target = e.currentTarget as HTMLImageElement;
                                  target.style.display = 'none';
                                  const fallback = target.nextElementSibling as HTMLElement;
                                  if (fallback) fallback.style.display = 'flex';
                                }}
                              />
                            ) : null}
                            <div
                              className="w-full h-full flex items-center justify-center"
                              style={{ display: result.frame_path ? 'none' : 'flex' }}
                            >
                              <Eye className="w-6 h-6 text-muted-foreground" />
                            </div>
                          </div>
                          <span className="text-xs text-muted-foreground font-medium">#{index + 1}</span>
                        </div>

                        {/* Content */}
                        <div className="flex-1">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex-1 pr-2">
                              {/* Summary (concise) */}
                              {result.summary ? (
                                <p className="text-foreground text-sm font-medium mb-1">
                                  {result.summary}
                                </p>
                              ) : (
                                <p className="text-foreground text-sm leading-relaxed group-hover:text-muted-foreground transition-colors">
                                  {result.description}
                                </p>
                              )}

                              {/* Objects detected */}
                              {result.objects_detected && result.objects_detected.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                  {result.objects_detected.slice(0, 3).map((obj, objIndex) => (
                                    <span
                                      key={objIndex}
                                      className="inline-block px-2 py-1 bg-muted rounded text-xs text-muted-foreground border border-border"
                                    >
                                      {obj}
                                    </span>
                                  ))}
                                  {result.objects_detected.length > 3 && (
                                    <span className="text-xs text-muted-foreground">
                                      +{result.objects_detected.length - 3} more
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                            <div className="flex items-center gap-2 text-xs">
                              <span className={`font-medium px-2 py-1 rounded ${getConfidenceColor(result.confidence)} bg-muted`}>
                                {result.confidence}%
                              </span>
                            </div>
                          </div>

                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="w-3 h-3" />
                              <span className="font-medium">{formatTime(result.timestamp)}</span>
                              {result.people_count !== undefined && (
                                <>
                                  <span className="mx-1">•</span>
                                  <Users className="w-3 h-3" />
                                  <span>{result.people_count} person{result.people_count !== 1 ? 's' : ''}</span>
                                </>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              {/* Expand/Collapse detailed analysis */}
                              {result.detailed_analysis && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleExpandResult(index);
                                  }}
                                  className="opacity-0 group-hover:opacity-100 transition-opacity bg-muted hover:bg-accent text-muted-foreground hover:text-accent-foreground px-2 py-1 rounded text-xs font-medium flex items-center gap-1"
                                >
                                  {expandedResults.has(index) ? (
                                    <>
                                      <ChevronUp className="w-3 h-3" />
                                      Less
                                    </>
                                  ) : (
                                    <>
                                      <ChevronDown className="w-3 h-3" />
                                      Details
                                    </>
                                  )}
                                </button>
                              )}
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onTimeJump(result.timestamp);
                                }}
                                className="opacity-0 group-hover:opacity-100 transition-opacity bg-primary hover:bg-primary/90 text-primary-foreground px-3 py-1 rounded text-xs font-medium flex items-center gap-1"
                              >
                                <Eye className="w-3 h-3" />
                                Jump to frame
                              </button>
                            </div>
                          </div>

                          {/* Expanded detailed analysis */}
                          {expandedResults.has(index) && result.detailed_analysis && (
                            <div className="mt-3 pt-3 border-t border-border">
                              <p className="text-sm text-muted-foreground leading-relaxed">
                                {result.detailed_analysis}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <div className="max-w-md mx-auto">
                <Search className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium text-foreground mb-2">
                  No matches found for "{searchQuery}"
                </h3>
                <p className="text-sm text-muted-foreground mb-6 leading-relaxed">
                  Try different keywords or be more specific about visual elements. Search works best with objects, people, colors, or scenes.
                </p>
                
                {/* Quick suggestions for failed searches */}
                <div className="space-y-4">
                  <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
                    <h4 className="font-medium text-amber-800 dark:text-amber-200 mb-2">🔧 Troubleshooting</h4>
                    <div className="text-sm text-amber-700 dark:text-amber-300 space-y-2">
                      <div>• <strong>For uploaded videos:</strong> Make sure the video file exists and is properly processed</div>
                      <div>• <strong>For YouTube videos:</strong> Visual search works best with recently uploaded content</div>
                      <div>• <strong>Search tips:</strong> Try broader terms like "person" instead of specific names</div>
                      <div>• <strong>Best results:</strong> Clear, prominent objects work better than small details</div>
                    </div>
                    
                    <div className="mt-3 pt-3 border-t border-amber-200 dark:border-amber-700 space-y-2">
                      <button
                        onClick={async () => {
                          try {
                            setIsSearching(true);
                            await apiService.analyzeVideoFrames(videoId);
                            toast({
                              title: "Frame Analysis Started",
                              description: "Video frames are being analyzed. This may take a few minutes. Try your search again shortly.",
                            });
                          } catch (error) {
                            toast({
                              title: "Analysis Failed",
                              description: "Could not start frame analysis. Please try again.",
                              variant: "destructive",
                            });
                          } finally {
                            setIsSearching(false);
                          }
                        }}
                        disabled={isSearching}
                        className="w-full text-xs bg-amber-600 hover:bg-amber-700 text-white px-3 py-1.5 rounded font-medium transition-colors disabled:opacity-50"
                      >
                        {isSearching ? 'Processing...' : 'Analyze Video Frames'}
                      </button>
                      
                      <button
                        onClick={() => {
                          setSearchQuery('person');
                          handleSearch({ preventDefault: () => {} } as React.FormEvent);
                        }}
                        disabled={isSearching}
                        className="w-full text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded font-medium transition-colors disabled:opacity-50"
                      >
                        🧪 Test Search "person"
                      </button>
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-xs text-muted-foreground font-medium mb-2">Try these proven searches:</p>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {['person', 'people', 'microphone', 'background', 'speaker', 'text', 'screen', 'chair'].map((suggestion) => (
                        <button
                          key={suggestion}
                          type="button"
                          onClick={() => setSearchQuery(suggestion)}
                          className="px-3 py-1.5 text-xs bg-muted hover:bg-primary hover:text-primary-foreground text-muted-foreground rounded-full transition-all border border-border hover:border-primary"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VisualSearch;