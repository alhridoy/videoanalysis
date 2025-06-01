# Native Video Search API Documentation

This document describes the native video search endpoints that leverage Gemini 2.5's video understanding capabilities for advanced visual search.

## Overview

The Native Video Search API provides powerful video analysis capabilities:
- Direct video understanding without frame extraction
- Returns video clips (5-10 seconds) instead of individual frames
- Supports multiple search types: objects, counting, color combinations, text, and scenes
- Includes confidence scores and detailed descriptions
- Efficient video upload caching to avoid re-uploads

## Base URL

```
/api/search/native
```

## Endpoints

### 1. General Visual Search

**Endpoint:** `POST /native/search`

Search for any visual content in the video.

**Request Body:**
```json
{
  "video_id": 1,
  "query": "red car",
  "search_type": "general"  // Options: general, object, counting, color, text, scene
}
```

**Response:**
```json
{
  "query": "red car",
  "search_type": "general",
  "clips": [
    {
      "start_time": 15.5,
      "end_time": 20.5,
      "duration": 5.0,
      "description": "A red sedan car drives through the intersection",
      "confidence": 0.92,
      "match_type": "exact",
      "visual_elements": ["red color", "sedan", "intersection", "daytime"],
      "context": "Urban street scene with traffic",
      "timestamp_formatted": "00:15",
      "end_timestamp_formatted": "00:20"
    }
  ],
  "total_clips": 3,
  "processing_time": 4.2
}
```

### 2. Count Visual Elements

**Endpoint:** `POST /native/count`

Count occurrences of specific visual elements.

**Request Body:**
```json
{
  "video_id": 1,
  "element": "cars",
  "count_type": "unique"  // Options: unique (distinct instances), total (all appearances)
}
```

**Response:**
```json
{
  "element": "cars",
  "count_type": "unique",
  "total_count": 5,
  "instances": [
    {
      "instance_id": "car_1",
      "first_appearance": 10.5,
      "last_appearance": 45.2,
      "total_screen_time": 15.7,
      "appearances": [
        {"start": 10.5, "end": 15.2},
        {"start": 40.1, "end": 45.2}
      ],
      "description": "Red sedan, license plate visible",
      "distinguishing_features": ["red color", "4-door sedan", "sunroof"],
      "confidence": 0.88
    }
  ],
  "temporal_pattern": "Cars appear throughout the video with clusters during traffic scenes",
  "clips": [...],
  "confidence": 0.85
}
```

### 3. Color + Object Search

**Endpoint:** `POST /native/color-object`

Search for specific color and object combinations.

**Request Body:**
```json
{
  "video_id": 1,
  "color": "blue",
  "object_type": "shirt"
}
```

**Response:**
```json
{
  "query": "blue shirt",
  "clips": [
    {
      "start_time": 30.0,
      "end_time": 35.0,
      "duration": 5.0,
      "description": "Person wearing a navy blue button-up shirt",
      "confidence": 0.95,
      "match_type": "exact",
      "visual_elements": [
        "Color: navy blue",
        "Object: button-up shirt",
        "Prominence: foreground"
      ],
      "context": "Indoor office setting",
      "timestamp_formatted": "00:30",
      "end_timestamp_formatted": "00:35"
    }
  ],
  "total_clips": 2,
  "search_type": "color_object_combo"
}
```

### 4. Text Search

**Endpoint:** `POST /native/text-search`

Find text appearing in the video (signs, captions, UI elements).

**Request:**
```
POST /native/text-search?video_id=1&text=exit
```

**Response:**
```json
{
  "query": "exit",
  "clips": [
    {
      "start_time": 45.0,
      "end_time": 50.0,
      "duration": 5.0,
      "description": "Text: 'EXIT' - sign text",
      "confidence": 0.90,
      "match_type": "exact",
      "visual_elements": [
        "Position: top-right",
        "Type: sign",
        "Readability: 95.0%"
      ],
      "context": "Building entrance with exit sign visible",
      "timestamp_formatted": "00:45",
      "end_timestamp_formatted": "00:50"
    }
  ],
  "total_clips": 1,
  "search_type": "text_detection"
}
```

### 5. Scene Type Search

**Endpoint:** `POST /native/scene-search`

Search for specific types of scenes or settings.

**Request:**
```
POST /native/scene-search?video_id=1&scene_type=outdoor
```

**Response:**
```json
{
  "query": "outdoor",
  "clips": [
    {
      "start_time": 0.0,
      "end_time": 25.0,
      "duration": 25.0,
      "description": "Outdoor city street scene with buildings and traffic",
      "confidence": 0.88,
      "match_type": "exact",
      "visual_elements": ["buildings", "street", "cars", "pedestrians", "daylight"],
      "context": "Sunny day - Urban environment, busy street",
      "timestamp_formatted": "00:00",
      "end_timestamp_formatted": "00:25"
    }
  ],
  "total_clips": 3,
  "search_type": "scene_analysis"
}
```

### 6. Find All Occurrences

**Endpoint:** `POST /native/find-all`

Exhaustive search to find ALL occurrences of a visual element.

**Request:**
```
POST /native/find-all?video_id=1&element=phone
```

**Response:**
```json
{
  "element": "phone",
  "clips": [
    {
      "start_time": 12.0,
      "end_time": 15.0,
      "duration": 3.0,
      "description": "Person holding smartphone, checking messages",
      "confidence": 0.95,
      "match_type": "exact",
      "visual_elements": ["smartphone", "hand holding", "screen visible"],
      "context": "Close-up shot of phone usage",
      "timestamp_formatted": "00:12",
      "end_timestamp_formatted": "00:15"
    },
    {
      "start_time": 45.0,
      "end_time": 46.0,
      "duration": 1.0,
      "description": "Phone briefly visible on desk in background",
      "confidence": 0.65,
      "match_type": "partial",
      "visual_elements": ["phone on desk", "background", "partial view"],
      "context": "Office desk scene",
      "timestamp_formatted": "00:45",
      "end_timestamp_formatted": "00:46"
    }
  ],
  "total_occurrences": 5,
  "search_type": "exhaustive_search"
}
```

### 7. Cleanup Upload

**Endpoint:** `DELETE /native/cleanup/{video_id}`

Clean up uploaded video from Gemini to free resources.

**Response:**
```json
{
  "video_id": 1,
  "cleanup_success": true,
  "message": "Video upload cleaned up from Gemini"
}
```

## Search Types Explained

1. **general**: Comprehensive search considering all visual aspects
2. **object**: Focus on identifying specific objects
3. **counting**: Identify and count distinct instances
4. **color**: Pay special attention to color matching
5. **text**: Look for text, captions, signs, UI elements
6. **scene**: Analyze scene types and settings

## Clip Properties

Each clip result contains:

- `start_time`: Start time in seconds
- `end_time`: End time in seconds
- `duration`: Clip duration in seconds
- `description`: Detailed description of what's found
- `confidence`: Confidence score (0.0 - 1.0)
- `match_type`: "exact", "partial", or "related"
- `visual_elements`: List of specific visual details
- `context`: Scene context information
- `timestamp_formatted`: Human-readable start timestamp
- `end_timestamp_formatted`: Human-readable end timestamp

## Example Usage

### Python Example

```python
import requests

# Search for red cars
response = requests.post(
    "http://localhost:8082/api/search/native/search",
    json={
        "video_id": 1,
        "query": "red car",
        "search_type": "object"
    }
)

clips = response.json()["clips"]
for clip in clips:
    print(f"Found at {clip['timestamp_formatted']}: {clip['description']}")
```

### JavaScript Example

```javascript
// Count people in video
const response = await fetch('/api/search/native/count', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        video_id: 1,
        element: "people",
        count_type: "unique"
    })
});

const result = await response.json();
console.log(`Found ${result.total_count} unique people in the video`);
```

## Best Practices

1. **Query Specificity**: Be specific in queries for better results
   - Good: "red Toyota car", "person wearing blue shirt"
   - Less effective: "thing", "stuff"

2. **Search Type Selection**: Choose appropriate search type
   - Use `counting` for "how many" questions
   - Use `color_object` for color-specific searches
   - Use `text` for finding text/signs
   - Use `scene` for location/setting searches

3. **Resource Management**: Clean up uploads when done
   - Call cleanup endpoint after finishing searches
   - Uploads are cached for 23 hours automatically

4. **Performance Considerations**:
   - First search uploads video (slower)
   - Subsequent searches reuse upload (faster)
   - Clips are 5-10 seconds for context

## Error Handling

Common errors:

- `404`: Video not found
- `400`: Video file not accessible
- `500`: Processing error (check logs)

Example error response:
```json
{
  "detail": "Video file not found"
}
```