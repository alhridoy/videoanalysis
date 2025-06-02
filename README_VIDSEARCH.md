# 🎬 VidSearch - AI-Powered Video Analysis & Search

A sophisticated multimodal video analysis system that enables intelligent chat, visual search, and content discovery using Google's Gemini 2.5 AI model.

## ✨ Features

### 🎯 Core Capabilities
- **Multimodal Video Chat**: Ask questions about video content with timestamped citations
- **Visual Search**: Find objects, people, and scenes using natural language queries
- **Smart Sections**: Auto-generated video segments with intelligent breakdowns
- **YouTube Integration**: Direct processing of YouTube URLs with optimized analysis
- **Real-time Thumbnails**: Dynamic frame extraction and thumbnail generation

### 🧠 AI-Powered Analysis
- **Gemini 2.5 Flash**: Advanced video understanding with temporal reasoning
- **Moment Retrieval**: Precise timestamp-based content location
- **Temporal Counting**: Count objects and events across video timeline
- **Scene Detection**: Intelligent frame selection and analysis
- **Semantic Search**: Enhanced search with word boundary detection

### 🎨 User Experience
- **Clean Interface**: Modern React-based UI with dark theme
- **Real-time Processing**: Live progress updates during video analysis
- **Interactive Timeline**: Click timestamps to jump to specific moments
- **Responsive Design**: Works seamlessly across devices
- **Health Monitoring**: Real-time backend status indicators

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/alhridoy/vidsearch.git
cd vidsearch
```

2. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
# Create .env file in backend directory
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

4. **Frontend Setup**
```bash
npm install
```

5. **Start the Application**
```bash
# Terminal 1: Start Backend
cd backend && python main.py

# Terminal 2: Start Frontend
npm run dev
```

6. **Access the Application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🎯 Usage

### Video Upload & Processing
1. **Upload Local Video**: Drag & drop or select video files
2. **YouTube URLs**: Paste YouTube links for direct processing
3. **Wait for Analysis**: AI processes video (30-120 seconds)
4. **Start Exploring**: Chat, search, and navigate your content

### Visual Search Examples
```
"red car"           → Find all clips with red vehicles
"people talking"    → Locate conversation segments  
"microphone"        → Find recording equipment
"text on screen"    → Locate text/subtitle moments
"outdoor scene"     → Find exterior shots
```

### Chat Examples
```
"What is this video about?"
"Summarize the main points"
"When do they discuss AI?"
"How many people appear in the video?"
"What happens at 5:30?"
```

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **FastAPI**: High-performance async API framework
- **SQLAlchemy**: Database ORM with SQLite
- **Google Gemini**: AI video analysis and chat
- **yt-dlp**: YouTube video downloading
- **FFmpeg**: Video processing and frame extraction

### Frontend (React/TypeScript)
- **React 18**: Modern component-based UI
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Vite**: Fast build tool and dev server
- **Shadcn/UI**: Beautiful component library

### Key Services
- **Video Processor**: Handles upload and YouTube processing
- **Gemini Service**: AI analysis and chat responses
- **Search Engine**: Visual and semantic search capabilities
- **Thumbnail Service**: Dynamic frame extraction
- **Health Monitor**: System status and diagnostics

## 📊 Performance

### Optimizations
- **Smart Frame Selection**: ~10% of frames for efficient processing
- **Async Processing**: Non-blocking video analysis
- **Caching**: Database and API response caching
- **Batch Operations**: Efficient database writes
- **Progressive Loading**: Incremental content delivery

### Typical Processing Times
- **Short Videos (< 5 min)**: 30-60 seconds
- **Medium Videos (5-20 min)**: 1-3 minutes  
- **Long Videos (20+ min)**: 3-10 minutes
- **Visual Search**: 10-30 seconds per query
- **Chat Responses**: 2-5 seconds

## 🔧 Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional
DATABASE_URL=sqlite:///./videochat.db
UPLOAD_DIR=./uploads
MAX_VIDEO_SIZE=500MB
FRAME_EXTRACTION_INTERVAL=5.0
```

### Model Configuration
```python
# Gemini Model Options
GEMINI_MODEL = "gemini-2.5-flash"  # Default
# GEMINI_MODEL = "gemini-2.5-pro-preview-0506"  # Enhanced
# GEMINI_MODEL = "gemini-2.0-flash-exp"  # Experimental
```

## 🧪 Testing

### Run Tests
```bash
# Backend tests
cd backend && python -m pytest

# Frontend tests  
npm test

# Integration tests
npm run test:e2e
```

### Manual Testing
```bash
# Test video processing
python test_video_processing.py

# Test visual search
python test_visual_search.py

# Test API endpoints
python test_api.py
```

## 📈 Monitoring

### Health Checks
- **Database**: Connection status
- **Gemini API**: Service availability  
- **Video Processor**: Processing capacity
- **Storage**: Disk space monitoring

### Logging
- **Application Logs**: `backend/logs/app.log`
- **Error Logs**: `backend/logs/error.log`
- **Access Logs**: `backend/logs/access.log`

## 🔒 Security

### Data Protection
- **Local Processing**: Videos processed locally
- **API Key Security**: Environment-based configuration
- **Input Validation**: Comprehensive request validation
- **File Type Restrictions**: Safe upload filtering

### Privacy
- **No Data Retention**: Videos can be deleted after processing
- **Local Storage**: All data stored locally by default
- **Configurable Cleanup**: Automatic file cleanup options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: Advanced AI video understanding
- **LangChain**: Inspiration for multimodal approaches
- **FFmpeg**: Powerful video processing
- **React Community**: Excellent frontend ecosystem

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alhridoy/vidsearch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alhridoy/vidsearch/discussions)
- **Email**: [your-email@example.com]

---

**Built with ❤️ using Google Gemini 2.5 and modern web technologies**
