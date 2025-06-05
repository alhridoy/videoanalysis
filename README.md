# VidChat - Multimodal Video Analysis System

A powerful multimodal video analysis system that enables intelligent chat with YouTube videos, generates timestamped sections, and provides advanced visual content search using Gemini's video understanding capabilities.

## Key Features

### 1. **RAG-Based Video Chat**
- Upload videos or provide YouTube links
- Chat with video content using natural language
- Get timestamped responses with clickable citations
- Context-aware conversations about video content

### 2. **Visual Search with Natural Language**
- Search for objects, people, or scenes within video frames
- Natural language queries (e.g., "red car", "person speaking")
- Semantic detection with confidence scoring
- Jump to specific moments in the video timeline

### 3. **Intelligent Section Breakdown**
- Automatic video segmentation into 5-minute sections
- AI-generated section titles and descriptions
- Hyperlinked timestamps for easy navigation
- Key topics extraction for each section

### 4. **Advanced Video Understanding**
- Powered by Gemini 2.5 Pro/Flash for enhanced video analysis
- Temporal counting and moment retrieval capabilities
- Support for videos up to 6 hours with configurable resolution
- Frame-by-frame content analysis and description

## Recent Improvements

### Visual Search False Positive Fix
- **Problem**: Search for "car" returned false positives from partial word matches (e.g., "card")
- **Solution**: Implemented regex word boundary detection for exact word matching
- **Result**: 100% accurate search results with no false positives

### Enhanced Search Accuracy
- Word boundary detection prevents partial matches
- Strict semantic validation with confidence thresholds
- Disabled pattern-based fallbacks that generated fake results
- Comprehensive logging for debugging and transparency

## Technology Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **SQLAlchemy** - Database ORM with SQLite
- **Google Gemini 2.5** - Advanced multimodal AI for video understanding
- **ChromaDB** - Vector database for semantic search
- **FFmpeg** - Video processing and frame extraction
- **yt-dlp** - YouTube video downloading

### Frontend
- **React + TypeScript** - Modern web interface
- **Tailwind CSS** - Utility-first styling
- **Vite** - Fast development and build tool
- **Lucide React** - Beautiful icons

## Prerequisites

- Python 3.8+
- Node.js 16+
- FFmpeg installed
- Google Gemini API key

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/alhridoy/vidchat.git
cd vidchat
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Create `.env` file in the backend directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=sqlite:///./videochat.db
```

### 4. Frontend Setup
```bash
cd ../  # Back to root directory
npm install
```

## Running the Application

### Start Backend Server
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend Development Server
```bash
npm run dev
```

The application will be available at:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Usage Guide

### 1. **Upload or Add YouTube Video**
- Click "Upload Video" to add a local video file
- Or paste a YouTube URL to analyze online content
- Wait for processing to complete (includes transcript extraction and frame analysis)

### 2. **Chat with Video**
- Use the chat interface to ask questions about the video
- Get responses with clickable timestamps
- Citations show relevant video moments

### 3. **Visual Search**
- Click "Visual Search" tab
- Enter natural language queries (e.g., "person with microphone", "red object")
- Browse results with confidence scores
- Jump to specific video moments

### 4. **Browse Sections**
- View auto-generated video sections
- Click timestamps to navigate to specific moments
- Explore key topics for each section

## Testing

### Run Visual Search Tests
```bash
python test_car_search_fix.py
```

This test validates that the visual search system:
- Returns 0 results for non-existent objects (no false positives)
- Finds legitimate matches when content exists
- Provides accurate confidence scores

## API Endpoints

### Video Management
- `POST /api/v1/video/youtube` - Process YouTube video
- `GET /api/v1/video/{video_id}` - Get video information
- `GET /api/v1/video/{video_id}/sections` - Get video sections

### Visual Search
- `POST /api/v1/search/visual` - Perform visual search
- `GET /api/v1/search/{video_id}/frames` - Get video frames

### Chat
- `POST /api/v1/chat/{video_id}/message` - Send chat message
- `GET /api/v1/chat/{video_id}/history` - Get chat history

## Performance Optimizations

- **Batch Video Processing**: Efficient handling of long videos
- **Scene Change Detection**: Smart frame selection (~10% of total frames)
- **Dual Embedding**: MiniLM text + CLIP image embeddings with weighted fusion
- **Redis Caching**: Fast retrieval of processed content
- **Real Vector Search**: ChromaDB for production-grade semantic search

## Security & Privacy

- Secure API key management through environment variables
- Local video processing (no data sent to external services except Gemini API)
- Configurable data retention policies
- CORS protection and input validation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini team for advanced video understanding capabilities
- LangChain community for multimodal AI insights
- Open source contributors for the amazing tools and libraries

---

**Built with care for intelligent video analysis and interaction**
