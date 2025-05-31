
import React, { useState, useRef } from 'react';
import { Upload, Link, Play, AlertCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiService } from '@/services/api';

interface VideoUploadProps {
  onVideoUpload: (videoId: number, videoUrl: string, title: string) => void;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoUpload }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const videoFile = files.find(file => file.type.startsWith('video/'));

    if (videoFile) {
      handleFileUpload(videoFile);
    } else {
      toast({
        title: "Invalid file type",
        description: "Please upload a video file",
        variant: "destructive",
      });
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsProcessing(true);

    try {
      const result = await apiService.uploadVideo(file);

      if (result.status === 'completed') {
        onVideoUpload(result.video_id, URL.createObjectURL(file), file.name);
        toast({
          title: "Video uploaded successfully!",
          description: "You can now chat with your video",
        });
      } else {
        toast({
          title: "Video processing...",
          description: "Your video is being processed. Please wait.",
        });
      }
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload video",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleYouTubeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!youtubeUrl.trim()) return;

    setIsProcessing(true);

    try {
      const result = await apiService.processYouTubeVideo(youtubeUrl);

      onVideoUpload(result.video_id, youtubeUrl, result.title);
      toast({
        title: "YouTube video loaded!",
        description: `Processing complete. ${result.has_transcript ? 'Transcript available.' : 'No transcript found.'}`,
      });
    } catch (error) {
      toast({
        title: "Failed to process YouTube video",
        description: error instanceof Error ? error.message : "Please check the URL and try again",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const extractYouTubeVideoId = (url: string): string | null => {
    const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
    const match = url.match(regex);
    return match ? match[1] : null;
  };

  if (isProcessing) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-card border border-border rounded-2xl p-12 text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-foreground border-t-transparent mx-auto mb-6"></div>
          <h2 className="text-2xl font-bold text-foreground mb-2">Processing Video...</h2>
          <p className="text-muted-foreground">
            We're analyzing your video and preparing it for AI chat
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* File Upload Area */}
      <div
        className={`relative bg-card rounded-2xl border-2 border-dashed transition-all duration-300 p-12 text-center ${
          isDragging
            ? 'border-foreground bg-muted scale-105'
            : 'border-border hover:border-foreground hover:bg-muted/50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        <Upload className="w-16 h-16 text-foreground mx-auto mb-6" />
        <h2 className="text-2xl font-bold text-foreground mb-4">
          Upload Your Video
        </h2>
        <p className="text-muted-foreground mb-6 max-w-md mx-auto">
          Drag and drop your video file here, or click to browse.
          Supports MP4, WebM, AVI, and more.
        </p>

        <button
          onClick={() => fileInputRef.current?.click()}
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg"
        >
          Choose File
        </button>
      </div>

      {/* Divider */}
      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-border"></div>
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-4 bg-background text-muted-foreground">
            or
          </span>
        </div>
      </div>

      {/* YouTube URL Input */}
      <div className="bg-card border border-border rounded-2xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <Link className="w-8 h-8 text-foreground" />
          <h3 className="text-xl font-bold text-foreground">
            Add YouTube Video
          </h3>
        </div>

        <form onSubmit={handleYouTubeSubmit} className="space-y-4">
          <div>
            <input
              type="url"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              className="w-full px-4 py-3 bg-input border border-border rounded-lg text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
            />
          </div>

          <button
            type="submit"
            disabled={!youtubeUrl.trim()}
            className="w-full bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed text-primary-foreground font-semibold py-3 px-6 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg flex items-center justify-center gap-2"
          >
            <Play className="w-5 h-5" />
            Load YouTube Video
          </button>
        </form>

        <div className="mt-4 p-3 bg-muted rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-muted-foreground mt-0.5 flex-shrink-0" />
          <p className="text-sm text-muted-foreground">
            YouTube videos will be processed for transcript extraction and AI analysis
          </p>
        </div>
      </div>
    </div>
  );
};

export default VideoUpload;
