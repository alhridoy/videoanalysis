#!/usr/bin/env python3
"""
Setup script to create .env file with necessary configuration
"""
import os

def create_env_file():
    """Create .env file with necessary configuration"""
    
    env_content = """# API Keys
GEMINI_API_KEY=AIzaSyBGz0jiD_oXZMFUNqYpst4IGs-71lNdhRw

# Database
DATABASE_URL=sqlite:///./videochat.db

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379

# Application Settings
DEBUG=True
MAX_VIDEO_SIZE_MB=500
FRAME_EXTRACTION_INTERVAL=5

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db

# YouTube Download Settings
YOUTUBE_DOWNLOAD_DIR=./uploads/youtube
"""
    
    # Check if .env already exists
    if os.path.exists('.env'):
        response = input(".env file already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with configuration")
    print("✅ Gemini API key has been configured")
    print("\nNote: Make sure to install yt-dlp for YouTube video downloads:")
    print("  pip install yt-dlp")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")

if __name__ == "__main__":
    create_env_file() 