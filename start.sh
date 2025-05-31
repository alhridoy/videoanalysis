#!/bin/bash

# VideoChat AI - Complete Startup Script
# This script sets up and starts both backend and frontend

set -e  # Exit on any error

echo "🚀 VideoChat AI - Starting Complete System"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "backend" ]; then
    print_error "Please run this script from the clip-quest-navigator-main directory"
    exit 1
fi

# Check for required tools
print_status "Checking system requirements..."

if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm and try again."
    exit 1
fi

print_success "System requirements check passed"

# Setup backend
print_status "Setting up backend..."

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv || python -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Check environment configuration
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Copying from .env.example..."
    cp .env.example .env
    print_warning "Please edit backend/.env and add your GEMINI_API_KEY before proceeding"
    print_warning "Get your API key from: https://makersuite.google.com/app/apikey"
    
    read -p "Press Enter after you've configured your .env file..."
fi

# Test backend setup
print_status "Testing backend setup..."
if python test_setup.py; then
    print_success "Backend setup test passed"
else
    print_error "Backend setup test failed. Please check the errors above."
    exit 1
fi

# Go back to root directory
cd ..

# Setup frontend
print_status "Setting up frontend..."

# Install Node.js dependencies
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
else
    print_status "Node.js dependencies already installed"
fi

print_success "Frontend setup complete"

# Start the applications
print_status "Starting applications..."

# Function to cleanup background processes
cleanup() {
    print_status "Shutting down applications..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend in background
print_status "Starting backend server..."
cd backend
source venv/bin/activate
python start.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    print_warning "Backend may not be fully ready yet. Continuing..."
fi

# Start frontend in background
print_status "Starting frontend development server..."
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 3

# Display access information
echo ""
echo "🎉 VideoChat AI is now running!"
echo "================================"
echo ""
echo "📱 Frontend Application:"
echo "   URL: http://localhost:5173"
echo "   Status: Running (PID: $FRONTEND_PID)"
echo ""
echo "🔧 Backend API:"
echo "   URL: http://localhost:8000"
echo "   Documentation: http://localhost:8000/docs"
echo "   Status: Running (PID: $BACKEND_PID)"
echo ""
echo "🔍 Features Available:"
echo "   ✅ Video Upload (drag & drop or file picker)"
echo "   ✅ YouTube Video Processing (paste URL)"
echo "   ✅ Chat with Videos (AI-powered Q&A)"
echo "   ✅ Video Sections (auto-generated chapters)"
echo "   ✅ Visual Search (natural language queries)"
echo ""
echo "💡 Usage Tips:"
echo "   • Upload a video file or paste a YouTube URL to get started"
echo "   • Try asking questions like 'What is this video about?'"
echo "   • Use visual search with queries like 'red car' or 'person speaking'"
echo "   • Click on timestamps in chat responses to jump to specific moments"
echo ""
echo "🛑 To stop the application, press Ctrl+C"
echo ""

# Wait for user to stop the application
wait
