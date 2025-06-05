
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


