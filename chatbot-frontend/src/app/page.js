"use client";
import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, RotateCcw, Heart } from 'lucide-react';

const EnglishPracticeChatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hi there! I'm here to help you practice English in a fun way. What would you like to chat about today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [practiceMode, setPracticeMode] = useState('chat');
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize WebSocket connection
    const ws = new WebSocket('ws://localhost:8000/ws/chat');
    
    ws.onopen = () => {
      console.log('Connected to WebSocket');
      setIsConnected(true);
      setSocket(ws);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.error) {
          console.error('Server error:', data.error);
          return;
        }

        // Add the bot's reply to messages
        const botMessage = {
          id: Date.now() + 1,
          text: data.reply,
          sender: 'bot',
          timestamp: new Date()
        };

        setMessages(prev => [...prev, botMessage]);
        setIsTyping(false);
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket');
      setIsConnected(false);
      setSocket(null);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  const modes = [
    { id: 'chat', label: '💬 Chat', emoji: '💬' },
    { id: 'grammar', label: '📝 Grammar', emoji: '📝' },
  ];

  const handleSend = () => {
    if (!inputText.trim() || !socket || !isConnected) return;

    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    // Send message to WebSocket
    const payload = {
      message: inputText
    };
    socket.send(JSON.stringify(payload));
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([{
      id: 1,
      text: "Hi there! 🌟 I'm here to help you practice English in a fun way. What would you like to chat about today?",
      sender: 'bot',
      timestamp: new Date()
    }]);
  };

  const changeMode = (mode) => {
    setPracticeMode(mode);
    const modeMessages = {
      chat: "Let's have a lovely conversation! Tell me about your favorite hobby 💫",
      grammar: "Time for some grammar fun! Write a sentence and I'll help make it perfect ✨",
    };

    setTimeout(() => {
      const botMessage = {
        id: Date.now(),
        text: modeMessages[mode],
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    }, 300);
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-gray-900 via-black to-purple-900">
      {/* Dark Header with Neon Accents */}
      <div className="bg-black/90 backdrop-blur-sm border-b border-purple-500/20">
        <div className="max-w-2xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 rounded-full flex items-center justify-center shadow-lg shadow-purple-500/50">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  English Buddy
                </h1>
                <p className="text-sm text-gray-400">
                  {isConnected ? 'Connected ✨' : 'Connecting...'}
                </p>
              </div>
            </div>
            <button
              onClick={clearChat}
              className="p-2 hover:bg-gray-800/50 rounded-full transition-colors group border border-gray-700/50"
              title="Start fresh"
            >
              <RotateCcw className="w-5 h-5 text-gray-400 group-hover:text-purple-400 transition-colors" />
            </button>
          </div>
        </div>
      </div>

      {/* Dark Mode Pills with Neon Glow */}
      <div className="bg-black/50 backdrop-blur-sm border-b border-gray-800/50">
        <div className="max-w-2xl mx-auto px-6 py-3">
          <div className="flex justify-center space-x-2">
            {modes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => changeMode(mode.id)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 border ${
                  practiceMode === mode.id
                    ? 'bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 text-white shadow-lg shadow-purple-500/50 scale-105 border-transparent'
                    : 'bg-gray-900/80 text-gray-300 hover:bg-gray-800/80 hover:scale-102 border-gray-700/50 hover:border-purple-500/50'
                }`}
              >
                {mode.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Dark Chat Area */}
      <div className="flex-1 overflow-y-auto px-6">
        <div className="max-w-2xl mx-auto py-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className="flex items-end space-x-2 max-w-xs">
                {message.sender === 'bot' && (
                  <div className="w-8 h-8 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 rounded-full flex items-center justify-center shadow-sm shadow-purple-500/50">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                )}
                <div
                  className={`px-4 py-3 rounded-2xl shadow-lg ${
                    message.sender === 'user'
                      ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-br-md shadow-blue-500/20'
                      : 'bg-gray-900/90 border border-gray-700/50 text-gray-100 rounded-bl-md shadow-black/20'
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.text}</p>
                </div>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex justify-start">
              <div className="flex items-end space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 rounded-full flex items-center justify-center shadow-sm shadow-purple-500/50">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
                <div className="bg-gray-900/90 border border-gray-700/50 px-4 py-3 rounded-2xl rounded-bl-md shadow-lg shadow-black/20">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Dark Input with Neon Accent */}
      <div className="bg-black/80 backdrop-blur-sm border-t border-gray-800/50">
        <div className="max-w-2xl mx-auto px-6 py-4">
          <div className="flex items-end space-x-3">
            <div className="flex-1 relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={isConnected ? "Type something lovely... ✨" : "Connecting to server..."}
                className="w-full resize-none rounded-2xl border border-gray-700/50 px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 bg-gray-900/90 backdrop-blur-sm placeholder-gray-500 text-gray-100"
                rows="1"
                style={{ minHeight: '48px', maxHeight: '120px' }}
                disabled={!isConnected}
              />
              <button
                onClick={handleSend}
                disabled={!inputText.trim() || !isConnected}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 hover:from-pink-600 hover:via-purple-600 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-700 text-white p-2 rounded-full transition-all duration-200 shadow-lg shadow-purple-500/30 hover:scale-110 disabled:hover:scale-100 hover:shadow-purple-500/50"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnglishPracticeChatbot;