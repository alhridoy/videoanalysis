from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
import shutil
import logging

from app.core.database import get_db
from app.models.video import Video, VideoFrame
from app.services.youtube_service import YouTubeService
# Enhanced services (require additional dependencies)
try:
    from app.services.optimized_video_processor import OptimizedVideoProcessor
    from app.services.fast_analysis_service import FastAnalysisService
    ENHANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSING_AVAILABLE = False
from app.core.config import settings
from .search import analyze_video_frames  # Import for automatic frame analysis

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a video file"""
    try:
        # Validate file size
        if file.size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.MAX_VIDEO_SIZE_MB}MB"
            )

        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="File must be a video"
            )

        # Create video record
        video = Video(
            title=file.filename,
            url="",
            video_type="upload",
            status="processing"
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        # Save uploaded file
        file_path = f"{settings.UPLOAD_DIR}/video_{video.id}.mp4"
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        video.file_path = file_path
        db.commit()

        # Process video in background (for now, process immediately)
        video_processor = request.app.state.video_processor
        result = await video_processor.process_uploaded_video(file_path, video.id)

        if result["status"] == "success":
            # Save frame information
            for frame_data in result["frames"]:
                frame = VideoFrame(
                    video_id=video.id,
                    timestamp=frame_data["timestamp"],
                    frame_path=frame_data["frame_path"]
                )
                db.add(frame)

            video.status = "completed"
            video.frame_count = result["frame_count"]
            video.duration = result["video_info"].get("duration")
            
            # Commit frames to database
            db.commit()
            
            # Trigger automatic frame analysis
            logger.info(f"Starting automatic frame analysis for uploaded video {video.id}...")
            analyze_response = await analyze_video_frames(request, video.id, db)
            logger.info(f"Frame analysis result: {analyze_response}")
        else:
            video.status = "failed"

        db.commit()

        return {
            "video_id": video.id,
            "status": video.status,
            "message": "Video uploaded and processed successfully" if result["status"] == "success" else "Video upload failed",
            "frame_count": video.frame_count or 0,
            "duration": video.duration
        }

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class YouTubeRequest(BaseModel):
    youtube_url: str

@router.post("/youtube")
async def process_youtube_video(
    request: Request,
    youtube_request: YouTubeRequest,
    db: Session = Depends(get_db)
):
    """Process a YouTube video"""
    try:
        youtube_service = YouTubeService()
        gemini_service = request.app.state.gemini_service
        video_processor = request.app.state.video_processor

        # Extract video ID
        video_id = youtube_service.extract_video_id(youtube_request.youtube_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # Get video info
        video_info = await youtube_service.get_video_info(video_id)

        # Create video record
        video = Video(
            title=video_info["title"],
            url=youtube_request.youtube_url,
            video_type="youtube",
            youtube_id=video_id,
            status="processing",
            duration=video_info.get("duration")
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        # Get transcript
        try:
            transcript_data = youtube_service.get_transcript(video_id)
            video.transcript = transcript_data["text"]

            # Generate sections using AI
            sections = await gemini_service.generate_video_sections(
                transcript_data["text"],
                video_info
            )
            video.sections = sections

        except Exception as e:
            logger.warning(f"Could not get transcript for {video_id}: {e}")
            video.transcript = None

        # Download and process the actual video
        try:
            logger.info(f"Downloading YouTube video {video_id}...")
            video_path = await youtube_service.download_video(video_id, video.id)
            
            if video_path:
                video.file_path = video_path
                
                # Process video to extract frames
                logger.info(f"Processing downloaded video {video_id}...")
                result = await video_processor.process_uploaded_video(video_path, video.id)
                
                if result["status"] == "success":
                    # Save frame information
                    for frame_data in result["frames"]:
                        frame = VideoFrame(
                            video_id=video.id,
                            timestamp=frame_data["timestamp"],
                            frame_path=frame_data["frame_path"]
                        )
                        db.add(frame)
                    
                    video.frame_count = result["frame_count"]
                    video.status = "completed"
                    
                    # Commit frames to database
                    db.commit()
                    
                    # Trigger automatic frame analysis
                    logger.info(f"Starting automatic frame analysis for video {video.id}...")
                    analyze_response = await analyze_video_frames(request, video.id, db)
                    logger.info(f"Frame analysis result: {analyze_response}")
                    
                    # Index transcript for vector search
                    if video.transcript:
                        from app.services.vector_service import VectorService
                        vector_service = VectorService()
                        if vector_service.available:
                            await vector_service.add_transcript_embedding(
                                video_id=str(video.id),
                                transcript=video.transcript,
                                metadata={
                                    "video_id": video.id,
                                    "title": video.title,
                                    "video_type": "youtube"
                                }
                            )
                else:
                    video.status = "failed"
                    logger.error(f"Failed to process video frames: {result}")
            else:
                # If download fails, still mark as completed if we have transcript
                if video.transcript:
                    video.status = "completed"
                    logger.warning("Video download failed, but transcript is available")
                else:
                    video.status = "failed"
                    logger.error("Both video download and transcript extraction failed")

        except Exception as e:
            logger.error(f"Error processing YouTube video file: {e}")
            # If video processing fails but we have transcript, still mark as completed
            if video.transcript:
                video.status = "completed"
            else:
                video.status = "failed"

        db.commit()

        return {
            "video_id": video.id,
            "status": video.status,
            "title": video.title,
            "has_transcript": bool(video.transcript),
            "sections_count": len(video.sections) if video.sections else 0,
            "frame_count": video.frame_count or 0,
            "duration": video.duration
        }

    except Exception as e:
        logger.error(f"Error processing YouTube video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}")
async def get_video(video_id: int, db: Session = Depends(get_db)):
    """Get video information"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "id": video.id,
        "title": video.title,
        "url": video.url,
        "video_type": video.video_type,
        "status": video.status,
        "duration": video.duration,
        "has_transcript": bool(video.transcript),
        "sections": video.sections,
        "frame_count": video.frame_count,
        "created_at": video.created_at
    }

@router.get("/{video_id}/sections")
async def get_video_sections(video_id: int, db: Session = Depends(get_db)):
    """Get video sections"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "video_id": video.id,
        "sections": video.sections or []
    }

@router.delete("/{video_id}")
async def delete_video(
    request: Request,
    video_id: int,
    db: Session = Depends(get_db)
):
    """Delete a video and its associated files"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        # Clean up files
        video_processor = request.app.state.video_processor
        video_processor.cleanup_video_files(video_id)

        # Delete from database
        db.query(VideoFrame).filter(VideoFrame.video_id == video_id).delete()
        db.delete(video)
        db.commit()

        return {"message": "Video deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-fast", response_model=dict)
async def upload_video_fast(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Optimized video upload with fast processing"""
    if not ENHANCED_PROCESSING_AVAILABLE:
        raise HTTPException(status_code=501, detail="Enhanced processing features not available. Please install additional dependencies.")

    try:
        # Validate file
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="Invalid video format")

        # Create video record
        video = Video(
            title=file.filename,
            video_type="upload",
            status="processing"
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        # Save uploaded file
        upload_dir = f"./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = f"{upload_dir}/{video.id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        video.file_path = file_path
        db.commit()

        # Initialize optimized services
        optimized_processor = OptimizedVideoProcessor()

        # Process video with optimizations
        logger.info(f"Starting optimized processing for video {video.id}...")
        result = await optimized_processor.quick_process_video(file_path, video.id)

        if result["status"] == "success":
            # Save frame information
            for frame_data in result["frames"]:
                frame = VideoFrame(
                    video_id=video.id,
                    timestamp=frame_data["timestamp"],
                    frame_path=frame_data["frame_path"]
                )
                db.add(frame)

            video.frame_count = result["frame_count"]
            video.duration = result["video_info"].get("duration")
            video.status = "completed"
            video.embedding_status = "pending"  # Will be processed separately

            db.commit()

            return {
                "message": "Video processed successfully with optimizations",
                "video_id": video.id,
                "processing_time": result.get("processing_time", 0),
                "frames_extracted": result["frame_count"],
                "optimizations_applied": result.get("optimization_applied", False),
                "status": "completed",
                "next_step": "AI analysis will be processed in background"
            }
        else:
            video.status = "failed"
            db.commit()
            raise HTTPException(status_code=500, detail=f"Video processing failed: {result.get('error', 'Unknown error')}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast video upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{video_id}/fast-analysis")
async def trigger_fast_analysis(
    request: Request,
    video_id: int,
    db: Session = Depends(get_db)
):
    """Trigger fast AI analysis for uploaded video"""
    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        if not video.file_path or not os.path.exists(video.file_path):
            raise HTTPException(status_code=400, detail="Video file not found")

        # Get frames
        frames = db.query(VideoFrame)\
            .filter(VideoFrame.video_id == video_id)\
            .order_by(VideoFrame.timestamp.asc())\
            .all()

        if not frames:
            raise HTTPException(status_code=400, detail="No frames available for analysis")

        # Initialize fast analyzer
        fast_analyzer = FastAnalysisService(request.app.state.gemini_api_key)

        # Perform fast analysis
        frame_paths = [frame.frame_path for frame in frames]
        analysis_result = await fast_analyzer.fast_analyze_video(
            video.file_path, video.id, frame_paths, video.title
        )

        if analysis_result.status == "success":
            # Update frame descriptions
            for frame_desc in analysis_result.frame_descriptions:
                timestamp = frame_desc.get("timestamp", 0)
                description = frame_desc.get("description", "")

                # Find corresponding frame
                frame = next((f for f in frames if abs(f.timestamp - timestamp) < 1.0), None)
                if frame and description:
                    frame.description = description

            # Store analysis metadata
            video.metadata = {
                "analysis_method": analysis_result.analysis_method,
                "confidence_score": analysis_result.confidence_score,
                "processing_time": analysis_result.processing_time,
                "summary": analysis_result.summary,
                "key_moments": analysis_result.key_moments[:5],
                "cached": analysis_result.cached
            }

            video.embedding_status = "completed"
            db.commit()

            return {
                "message": "Fast analysis completed successfully",
                "video_id": video.id,
                "analysis_time": analysis_result.processing_time,
                "analysis_method": analysis_result.analysis_method,
                "confidence_score": analysis_result.confidence_score,
                "cached_analysis": analysis_result.cached,
                "frames_analyzed": len(analysis_result.frame_descriptions),
                "key_moments_found": len(analysis_result.key_moments)
            }
        else:
            video.embedding_status = "failed"
            db.commit()
            raise HTTPException(status_code=500, detail=f"Analysis failed: {analysis_result.summary}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
