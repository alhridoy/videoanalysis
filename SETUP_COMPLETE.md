# VideoChat AI - Complete Setup Guide

## 🚀 Quick Setup

Your VideoChat AI system is now fully implemented with real functionality! Follow these steps to get it running:

### 1. **Backend Setup**

```bash
cd backend

# Create the .env file with your API key
python setup_env.py

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. **Install System Dependencies**

For full functionality, install these system dependencies:

```bash
# macOS
brew install ffmpeg redis

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg redis-server

# Windows
# Download and install from:
# FFmpeg: https://ffmpeg.org/download.html
# Redis: https://redis.io/download/
```

### 3. **Frontend Setup**

```bash
# Return to root directory
cd ..

# Install frontend dependencies
npm install
# or
bun install
```

### 4. **Start the Application**

```bash
# Option 1: Use the start script
chmod +x start.sh
./start.sh

# Option 2: Manual start
# Terminal 1 - Backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2 - Frontend
npm run dev
```

## ✅ What's Been Implemented

### 1. **Real Visual Search**
- Vector embeddings using ChromaDB and Sentence Transformers
- Automatic frame analysis with Gemini Vision API
- Semantic search across video frames
- Confidence-scored results

### 2. **YouTube Video Processing**
- Full video download using yt-dlp
- Automatic frame extraction
- Transcript extraction and indexing
- Real frame analysis (not mock data)

### 3. **Enhanced Chat with RAG**
- Vector search for relevant transcript segments
- Context-aware responses using semantic search
- Better accuracy with focused context retrieval

### 4. **Automatic Processing Pipeline**
- Frames are automatically analyzed after extraction
- Embeddings are created for both frames and transcripts
- Vector indexes are built automatically

## 🎯 How to Use

### 1. **Upload a Video**
- Drag and drop a video file or click to browse
- Video will be processed automatically
- Frames will be extracted and analyzed

### 2. **Process YouTube Video**
- Paste any YouTube URL
- Video will be downloaded and processed
- Transcript will be extracted if available

### 3. **Chat with Video**
- Ask questions about the video content
- Responses will include timestamp citations
- Click timestamps to jump to specific moments

### 4. **Visual Search**
- Enter natural language queries like:
  - "red car"
  - "person speaking"
  - "computer screen"
  - "outdoor scene"
- Results show confidence scores and timestamps

## 📊 System Status Check

Run the test script to verify everything is working:

```bash
python test_system.py
```

## 🔧 Troubleshooting

### If visual search returns no results:
1. Make sure frames have been analyzed (check the `/analyze-frames` endpoint)
2. Verify ChromaDB is running and accessible
3. Check that embeddings were created successfully

### If YouTube download fails:
1. Ensure yt-dlp is installed: `pip install yt-dlp`
2. Check internet connection
3. Verify the YouTube URL is valid

### If frame analysis fails:
1. Check your Gemini API key is valid
2. Ensure you have API quota available
3. Verify image files exist in the uploads/frames directory

## 🚦 API Endpoints

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 💡 Tips

1. **For best results**: Upload videos with clear visual content
2. **Frame extraction**: Adjust `FRAME_EXTRACTION_INTERVAL` in .env (default: 5 seconds)
3. **Search accuracy**: More specific queries yield better results
4. **Performance**: First video processing may be slower as models are loaded

## 🎉 You're Ready!

Your VideoChat AI system is now fully functional with:
- ✅ Real vector search
- ✅ Actual YouTube video downloads
- ✅ Gemini Vision API integration
- ✅ Semantic RAG for chat
- ✅ Automatic frame analysis

Enjoy exploring your videos with AI! 