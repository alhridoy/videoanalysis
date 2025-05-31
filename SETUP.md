# VideoChat AI - Setup Guide

## Overview
This is a multimodal video analysis system that allows you to:
1. **Chat with videos** using RAG (Retrieval-Augmented Generation)
2. **Navigate videos** with AI-generated sections and timestamps
3. **Search video content** using natural language queries for visual elements

## Prerequisites

### Required API Keys
1. **Gemini API Key** (Required)
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for later use

### System Requirements
- **Node.js** 18+ (for frontend)
- **Python** 3.8+ (for backend)
- **FFmpeg** (for video processing)

## Installation

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

**Required .env configuration:**
```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Database (SQLite for development)
DATABASE_URL=sqlite:///./videochat.db

# Application Settings
DEBUG=True
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
MAX_VIDEO_SIZE_MB=500
FRAME_EXTRACTION_INTERVAL=5

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd .

# Install dependencies
npm install

# Or if you prefer yarn
yarn install
```

## Running the Application

### 1. Start the Backend

```bash
# From backend directory
cd backend

# Activate virtual environment if not already active
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Start the server
python start.py
```

The backend will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

### 2. Start the Frontend

```bash
# From project root directory
npm run dev

# Or with yarn
yarn dev
```

The frontend will be available at: http://localhost:5173

## Features

### 1. Video Upload & Processing
- **File Upload**: Drag & drop video files or click to browse
- **YouTube Integration**: Paste YouTube URLs for automatic processing
- **Supported Formats**: MP4, WebM, AVI, MOV, and more

### 2. Chat with Videos
- **Natural Language Queries**: Ask questions about video content
- **Contextual Responses**: AI provides answers based on video transcript and content
- **Timestamp Citations**: Responses include clickable timestamps to jump to relevant sections

### 3. Video Sections
- **Auto-Generated Chapters**: AI creates logical video sections with timestamps
- **Interactive Timeline**: Click sections to jump to specific parts
- **Topic Extraction**: Key topics identified for each section

### 4. Visual Search
- **Frame-Based Search**: Search for objects, people, or scenes within video frames
- **Natural Language Queries**: "red car", "person speaking", "computer screen"
- **Confidence Scoring**: Results ranked by AI confidence levels

## API Endpoints

### Video Management
- `POST /api/v1/video/upload` - Upload video file
- `POST /api/v1/video/youtube` - Process YouTube video
- `GET /api/v1/video/{id}` - Get video information
- `GET /api/v1/video/{id}/sections` - Get video sections

### Chat
- `POST /api/v1/chat/message` - Send chat message
- `GET /api/v1/chat/{video_id}/history` - Get chat history

### Search
- `POST /api/v1/search/visual` - Visual content search
- `GET /api/v1/search/{video_id}/frames` - Get video frames

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY is required" Error**
   - Ensure you've set the GEMINI_API_KEY in your .env file
   - Verify the API key is valid and has proper permissions

2. **Video Upload Fails**
   - Check file size (default limit: 500MB)
   - Ensure video format is supported
   - Verify backend is running and accessible

3. **YouTube Processing Fails**
   - Check if the video has captions/transcript available
   - Verify the YouTube URL is valid and accessible
   - Some videos may have restricted access

4. **Frontend Can't Connect to Backend**
   - Ensure backend is running on port 8000
   - Check CORS settings in backend configuration
   - Verify no firewall blocking the connection

### Performance Tips

1. **Video Processing**
   - Smaller videos process faster
   - Videos with transcripts work better for chat
   - Frame extraction interval affects processing time

2. **Search Performance**
   - First search may be slower (model loading)
   - Subsequent searches are faster
   - More specific queries yield better results

## Development

### Project Structure
```
clip-quest-navigator-main/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Configuration
│   │   ├── models/         # Database models
│   │   └── services/       # Business logic
│   ├── requirements.txt
│   └── start.py
├── src/                    # React frontend
│   ├── components/         # UI components
│   ├── services/          # API client
│   └── pages/             # Application pages
└── package.json
```

### Adding New Features
1. Backend: Add routes in `app/api/routes/`
2. Frontend: Add components in `src/components/`
3. API Integration: Update `src/services/api.ts`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API documentation at http://localhost:8000/docs
3. Check browser console for frontend errors
4. Review backend logs for server errors
