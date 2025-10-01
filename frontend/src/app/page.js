"use client";
import React, { useState, useRef, useEffect } from "react";
import { Send, Sparkles, Heart, RotateCcw } from "lucide-react";

const BotAvatar = () => (
  <div className="w-8 h-8 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 rounded-full flex items-center justify-center shadow-sm shadow-purple-500/50">
    <Sparkles className="w-4 h-4 text-white" />
  </div>
);

const MessageBubble = ({ text, sender }) => (
  <div className={`flex ${sender === "user" ? "justify-end" : "justify-start"}`}>
    <div className="flex items-end space-x-2 max-w-xs">
      {sender === "bot" && <BotAvatar />}
      <div
        className={`px-4 py-3 rounded-2xl shadow-lg ${
          sender === "user"
            ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-br-md shadow-blue-500/20"
            : "bg-gray-900/90 border border-gray-700/50 text-gray-100 rounded-bl-md shadow-black/20"
        }`}
      >
        <p className="text-sm leading-relaxed">{text}</p>
      </div>
    </div>
  </div>
);

const TypingIndicator = () => (
  <div className="flex justify-start">
    <div className="flex items-end space-x-2">
      <BotAvatar />
      <div className="bg-gray-900/90 border border-gray-700/50 px-4 py-3 rounded-2xl rounded-bl-md shadow-lg shadow-black/20">
        <div className="flex space-x-1">
          {["pink", "purple", "cyan"].map((c, i) => (
            <div
              key={c}
              className={`w-2 h-2 bg-${c}-400 rounded-full animate-bounce`}
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  </div>
);

const EnglishPracticeChatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Time for some grammar fun! Write a sentence and I'll help make it perfect ✨",
      sender: "bot",
    },
  ]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/chat");
    ws.onopen = () => {
      setIsConnected(true);
      setSocket(ws);
    };
    ws.onmessage = ({ data }) => {
      try {
        const { reply, error } = JSON.parse(data);
        if (error) return console.error("Server error:", error);
        setMessages((p) => [...p, { id: Date.now(), text: reply, sender: "bot" }]);
        setIsTyping(false);
      } catch (e) {
        console.error("Message parse error:", e);
      }
    };
    ws.onclose = () => {
      setIsConnected(false);
      setSocket(null);
    };
    ws.onerror = () => setIsConnected(false);

    return () => ws.readyState === WebSocket.OPEN && ws.close();
  }, []);

  const handleSend = () => {
    if (!inputText.trim() || !socket || !isConnected) return;
    const msg = { id: Date.now(), text: inputText, sender: "user" };
    setMessages((p) => [...p, msg]);
    setInputText("");
    setIsTyping(true);
    socket.send(JSON.stringify({ message: inputText }));
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-gray-900 via-black to-purple-900">
      {/* Header */}
      <div className="bg-black/90 backdrop-blur-sm border-b border-purple-500/20">
        <div className="max-w-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 rounded-full flex items-center justify-center shadow-lg shadow-purple-500/50">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Grammar Buddy
              </h1>
              <p className="text-sm text-gray-400">{isConnected ? "Connected ✨" : "Connecting..."}</p>
            </div>
          </div>
          <button
            onClick={() =>
              setMessages([
                {
                  id: 1,
                  text: "Time for some grammar fun! Write a sentence and I'll help make it perfect ✨",
                  sender: "bot",
                },
              ])
            }
            className="p-2 hover:bg-gray-800/50 rounded-full border border-gray-700/50 group"
          >
            <RotateCcw className="w-5 h-5 text-gray-400 group-hover:text-purple-400 transition-colors" />
          </button>
        </div>
      </div>

      {/* Chat */}
      <div className="flex-1 overflow-y-auto px-6">
        <div className="max-w-2xl mx-auto py-6 space-y-4">
          {messages.map((m) => (
            <MessageBubble key={m.id} {...m} />
          ))}
          {isTyping && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-black/80 backdrop-blur-sm border-t border-gray-800/50">
        <div className="max-w-2xl mx-auto px-6 py-4 flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), handleSend())}
              placeholder={isConnected ? "Type a sentence for grammar correction... ✨" : "Connecting to server..."}
              rows="1"
              disabled={!isConnected}
              className="w-full resize-none rounded-2xl border border-gray-700/50 px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 bg-gray-900/90 placeholder-gray-500 text-gray-100"
              style={{ minHeight: "48px", maxHeight: "120px" }}
            />
            <button
              onClick={handleSend}
              disabled={!inputText.trim() || !isConnected}
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 hover:from-pink-600 hover:via-purple-600 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-700 text-white p-2 rounded-full shadow-lg transition-all hover:scale-110 disabled:hover:scale-100"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnglishPracticeChatbot;