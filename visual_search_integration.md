# Native Video Search Integration

## Overview
The `/visual` endpoint in `backend/app/api/routes/search.py` has been updated to integrate NativeVideoSearchService with intelligent fallback handling.

## Key Features

### 1. Native Video Search Priority
- **First Check**: NativeVideoSearchService is checked for availability
- **Upload Videos**: Native search is used for uploaded videos when `video.file_path` exists
- **Better Results**: Native search provides more accurate clip-based results

### 2. Query Parameter Control
- **`use_native`**: New boolean parameter (default: `true`) to control search method
- **Example**: `POST /api/search/visual?use_native=false` to force frame-based search

### 3. Intelligent Query Type Detection
The system automatically detects query types:
- **Counting**: "how many", "count", "number of"
- **Text Detection**: "text", "sign", "writing", "label"
- **Scene Analysis**: "scene", "background", "setting", "location"
- **Color Search**: "red", "blue", "green", "yellow", "color"
- **Object Detection**: Default for other queries

### 4. Clip-Based Results
Native search returns:
- **Start/End Times**: Precise clip boundaries
- **Thumbnails**: Preview images for each clip
- **Confidence Scores**: Accuracy of each match
- **Clip Duration**: Length of each relevant segment

### 5. Frame Endpoint
New endpoint added: `GET /api/search/{video_id}/frame?timestamp={timestamp}&size={size}`
- Serves frames at specific timestamps
- Supports multiple sizes: small, medium, large, timeline
- Uses ThumbnailService when available
- Falls back to direct frame extraction

### 6. Fallback Chain
1. **Native Video Search** (if available and enabled)
2. **Batch Video Processing** (Gemini video API)
3. **Enhanced Vector Search** (hybrid search)
4. **Standard Vector Search**
5. **Semantic Search** (final fallback)

## API Response Format

```json
{
  "query": "red car",
  "results": [
    {
      "timestamp": 15.5,
      "confidence": 92.5,
      "description": "Red car visible in the scene",
      "frame_path": "/api/search/1/frame?timestamp=15.5",
      "summary": "Clip 15.5s - 18.2s: Red car driving...",
      "detailed_analysis": "Full description..."
    }
  ],
  "clips": [
    {
      "start_time": 15.5,
      "end_time": 18.2,
      "confidence": 92.5,
      "description": "Red car driving on highway",
      "frame_count": 3,
      "frames": [...]
    }
  ],
  "total_results": 5,
  "direct_answer": "Object 'red car' detected in 5 clips",
  "query_type": "object",
  "processing_method": "native_video_search"
}
```

## Implementation Details

### Native Search Integration
```python
# Initialize native search service
native_search_service = NativeVideoSearchService()

# Perform search
clips = await native_search_service.search_visual_content(
    video_path=video.file_path,
    query=search_request.query,
    search_type=search_type
)
```

### Frame Generation
- Each clip generates 2-5 representative frames
- Frame timestamps are distributed evenly across clip duration
- Confidence scores decrease slightly for later frames in a clip

### Error Handling
- Graceful fallback if native search fails
- Detailed logging for debugging
- Maintains service availability

## Benefits

1. **Better Accuracy**: Native video understanding vs frame-by-frame analysis
2. **Faster Results**: Direct video analysis without frame extraction
3. **Richer Context**: Clips provide temporal context
4. **Flexible Control**: Query parameter allows method selection
5. **Robust Fallbacks**: Multiple search methods ensure reliability

## Usage Examples

### Basic Visual Search
```bash
curl -X POST "http://localhost:8000/api/search/visual" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": 1,
    "query": "person with microphone",
    "max_results": 10
  }'
```

### Force Frame-Based Search
```bash
curl -X POST "http://localhost:8000/api/search/visual?use_native=false" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": 1,
    "query": "red car",
    "max_results": 5
  }'
```

### Get Frame at Timestamp
```bash
curl "http://localhost:8000/api/search/1/frame?timestamp=15.5&size=large"
```

## Future Enhancements

1. **Caching**: Cache native search results for repeated queries
2. **Batch Queries**: Support multiple queries in single request
3. **Advanced Filters**: Time range, confidence threshold filters
4. **Webhook Support**: Notify when search completes
5. **Export Options**: Download clips as video segments