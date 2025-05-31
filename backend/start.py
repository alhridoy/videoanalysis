#!/usr/bin/env python3
"""
Startup script for VideoChat AI Backend
"""
import os
import sys
import subprocess
import logging

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import fastapi
        import uvicorn
        import google.generativeai
        import cv2
        print("✅ Core packages are installed")

        # ChromaDB is optional for now
        try:
            import chromadb
            print("✅ ChromaDB is available")
        except ImportError:
            print("⚠️  ChromaDB not available (optional for vector search)")

        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup environment variables and directories"""
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("uploads/frames", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    # Check for .env file
    if not os.path.exists(".env"):
        print("⚠️  No .env file found. Please copy .env.example to .env and configure your API keys")
        return False

    print("✅ Environment setup complete")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting VideoChat AI Backend...")

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Setup environment
    if not setup_environment():
        sys.exit(1)

    # Start the server
    print("🌟 Starting FastAPI server on http://localhost:8000")
    print("📚 API documentation available at http://localhost:8000/docs")

    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
