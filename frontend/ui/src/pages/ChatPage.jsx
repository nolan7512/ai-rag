import React, { useState } from "react";
import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";
import {
  MainContainer,
  ChatContainer,
  MessageList,
  MessageInput,
  Message,
  TypingIndicator,
  MessageSeparator,
  Avatar,
} from "@chatscope/chat-ui-kit-react";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (text) => {
    if (!text.trim()) return;

    const userMessage = text.trim();

    // Thêm tin nhắn người dùng vào
    setMessages((prev) => [
      ...prev,
      {
        type: "text",
        message: userMessage,
        sender: "user",
        direction: "outgoing",
      },
    ]);

    setIsTyping(true);

    try {
      const form = new FormData();
      form.append("question", userMessage);

      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        body: form,
      });

      const data = await res.json();

      const botReply = data?.answer || "❌ Không có phản hồi từ hệ thống.";

      // Thêm tin nhắn bot
      setMessages((prev) => [
        ...prev,
        {
          type: "text",
          message: botReply,
          sender: "bot",
          direction: "incoming",
        },
      ]);
    } catch (err) {
      console.error("Lỗi gọi API:", err);
      setMessages((prev) => [
        ...prev,
        {
          type: "text",
          message: "⚠️ Lỗi khi gọi backend.",
          sender: "bot",
          direction: "incoming",
        },
      ]);
    }

    setIsTyping(false);
  };

  return (
    <div
      style={{
        height: "100vh",
        backgroundColor: "#f7f8fa",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        className="chat-header"
        style={{
          padding: "8px 16px",
          borderBottom: "1px solid #ddd",
          background: "#fff",
        }}
      >
        <h1 >💬 Trợ lý doanh nghiệp</h1>
      </div>

      <MainContainer style={{ flex: 1 }}>
        <ChatContainer>
          <MessageList
            typingIndicator={
              isTyping ? (
                <TypingIndicator content="Đang xử lý..." />
              ) : null
            }
          >
            <MessageSeparator content="Hôm nay" />
            {messages.map((m, i) => (
              <Message
                key={i}
                model={{
                  message: m.message,
                  sender: m.sender,
                  direction: m.direction,
                }}
              >
                {m.sender === "bot" && <Avatar name="Bot" src="/bot.jpg" />}
                {m.sender === "user" && (
                  <Avatar name="User" src="/user.jpg" />
                )}
              </Message>
            ))}
          </MessageList>

          <MessageInput
            placeholder="Nhập câu hỏi..."
            onSend={handleSend}
          />
        </ChatContainer>
      </MainContainer>
    </div>
  );
}
