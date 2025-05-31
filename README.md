# VideoChat AI - Multimodal Video Analysis System

A powerful multimodal video analysis system that enables intelligent conversations with videos, automatic section generation, and advanced visual search capabilities using Google's Gemini 2.5 video understanding.

## 🎯 Features

### 🎬 **Core Capabilities**
- **RAG-based Video Chat**: Upload videos or YouTube links and have intelligent conversations with the content
- **Automatic Section Breakdown**: AI-powered video segmentation with hyperlinked timestamps and citations
- **Visual Search**: Natural language search within video frames (e.g., "find red cars", "count people")
- **Moment Retrieval**: Advanced scene detection and key moment identification using Gemini 2.5
- **Temporal Counting**: Precise counting and tracking of objects/actions across video timeline

### 🚀 **Advanced Features**
- **Native Video Analysis**: Direct video processing using Gemini 2.5's video understanding capabilities
- **Enhanced Timeline**: Dynamic timeline display supporting videos up to 1 hour (configurable)
- **Multimodal RAG**: Combines text, visual, and temporal information for accurate responses
- **Real-time Processing**: Optimized video processing with configurable quality settings
- **YouTube Integration**: Direct YouTube URL processing and analysis

## 🛠️ Tech Stack

### **Frontend**
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Recharts** for data visualization
- **Lucide React** for icons
- **React Router** for navigation

### **Backend**
- **FastAPI** with Python 3.8+
- **SQLAlchemy** for database ORM
- **Google Gemini 2.5** Pro/Flash for AI analysis
- **ChromaDB** for vector search
- **FFmpeg** for video processing
- **OpenCV** for frame extraction

### **AI & Processing**
- **Google Gemini 2.5 Pro/Flash** for video understanding
- **Vector embeddings** for semantic search
- **Hybrid search** (keyword + vector)
- **Scene detection** for optimal frame selection
- **Batch processing** for efficiency

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- FFmpeg installed
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/alhridoy/videochatai.git
cd videochatai
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your Gemini API key:
# GEMINI_API_KEY=your_api_key_here
```

4. **Frontend Setup**
```bash
cd ../
npm install
```

5. **Run the Application**
```bash
# Terminal 1: Backend
cd backend
python main.py

# Terminal 2: Frontend
npm run dev
```

Visit `http://localhost:5173` to access the application.

## 🎯 Usage Examples

### Video Chat
1. Upload a video or paste a YouTube URL
2. Wait for processing to complete
3. Ask questions about the video content
4. Get timestamped responses with citations

### Visual Search
1. Navigate to the "Search" tab
2. Enter natural language queries like:
   - "Find scenes with red cars"
   - "Count how many people appear"
   - "Show me text or signs"
3. Browse results with confidence scores and timestamps

### Section Navigation
1. Go to the "Sections" tab
2. View automatically generated video sections
3. Click on any section to jump to that timestamp
4. See the full video timeline with section markers

## 🔧 Configuration

### Video Processing Limits
```env
MAX_VIDEO_DURATION_SECONDS=3600  # 1 hour
MAX_FRAMES_PER_VIDEO=300
MAX_VIDEO_SIZE_MB=500
```

### AI Model Selection
```env
GEMINI_MODEL=gemini-2.5-flash  # or gemini-2.5-pro-preview-0506
```

## 📖 Documentation

- **[Setup Guide](SETUP.md)** - Detailed installation and configuration
- **[Performance Optimization](PERFORMANCE_OPTIMIZATION.md)** - System optimization tips
- **[Production Deployment](PRODUCTION_IMPROVEMENTS.md)** - Production deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Google Gemini team for advanced video understanding capabilities
- The open-source community for excellent tools and libraries
