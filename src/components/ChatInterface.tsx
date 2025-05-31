
import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Clock } from 'lucide-react';
import { apiService } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  citations?: { text: string; time: number }[];
}

interface ChatInterfaceProps {
  videoId: number;
  videoUrl: string;
  onTimeJump: (time: number) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ videoId, videoUrl, onTimeJump }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hello! I've analyzed your video and I'm ready to answer questions about its content. What would you like to know?",
      sender: 'ai',
      timestamp: new Date(),
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newMessage]);
    const currentMessage = inputMessage;
    setInputMessage('');
    setIsTyping(true);

    try {
      const response = await apiService.sendChatMessage(videoId, currentMessage);

      const aiResponse: Message = {
        id: response.message_id.toString(),
        content: response.response,
        sender: 'ai',
        timestamp: new Date(),
        citations: response.citations
      };

      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      toast({
        title: "Chat Error",
        description: error instanceof Error ? error.message : "Failed to send message",
        variant: "destructive",
      });

      // Add error message
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: "I'm sorry, I encountered an error while processing your message. Please try again.",
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsTyping(false);
    }
  };

  const generateAIResponse = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();

    if (lowerMessage.includes('about') || lowerMessage.includes('summary')) {
      return "This video covers video analysis and AI-powered chat capabilities. The speaker discusses building applications that can process video content, extract meaningful information, and enable natural language interactions with video data. Key topics include RAG (Retrieval-Augmented Generation) for video content, timestamp-based navigation, and visual search capabilities.";
    }

    if (lowerMessage.includes('project') || lowerMessage.includes('features')) {
      return "The project involves three main features: 1) Chat with videos using RAG to answer questions about content, 2) Section breakdown with timestamp hyperlinks for navigation, and 3) Visual search to find specific content within video frames using natural language queries.";
    }

    if (lowerMessage.includes('technology') || lowerMessage.includes('tech')) {
      return "The technology stack likely includes video processing APIs, AI/ML models for content analysis, natural language processing for chat capabilities, and computer vision for visual search functionality. The speaker mentions working at companies like Microsoft and Google, bringing experience from Big Tech.";
    }

    return "Based on the video content, I can help you understand the concepts discussed. The speaker covers video analysis, AI integration, and practical applications for processing video content. Could you be more specific about what aspect you'd like to explore?";
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex gap-3 max-w-[85%] ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
              {/* Avatar */}
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.sender === 'user'
                  ? 'bg-primary'
                  : 'bg-muted'
              }`}>
                {message.sender === 'user' ? (
                  <User className="w-4 h-4 text-primary-foreground" />
                ) : (
                  <Bot className="w-4 h-4 text-muted-foreground" />
                )}
              </div>

              {/* Message Content */}
              <div className={`rounded-lg p-3 ${
                message.sender === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-foreground'
              }`}>
                <p className="text-sm leading-relaxed">{message.content}</p>

                {/* Citations */}
                {message.citations && (
                  <div className="mt-3 pt-3 border-t border-border">
                    <p className="text-xs text-muted-foreground mb-2">References:</p>
                    <div className="space-y-1">
                      {message.citations.map((citation, index) => (
                        <button
                          key={index}
                          onClick={() => onTimeJump(citation.time)}
                          className="flex items-center gap-2 text-xs bg-accent hover:bg-accent/80 rounded px-2 py-1 transition-colors"
                        >
                          <Clock className="w-3 h-3" />
                          <span>{citation.text}</span>
                          <span className="text-muted-foreground">
                            [{formatTime(citation.time)}]
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <p className="text-xs text-muted-foreground mt-2">
                  {message.timestamp.toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </p>
              </div>
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[85%]">
              <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="bg-muted text-foreground rounded-lg p-3">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-foreground rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSendMessage} className="mt-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask anything about this video..."
            className="flex-1 px-4 py-3 bg-input border border-border rounded-lg text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
            disabled={isTyping}
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isTyping}
            className="px-4 py-3 bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed text-primary-foreground rounded-lg transition-all duration-300 transform hover:scale-105"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
