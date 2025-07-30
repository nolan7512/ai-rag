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
    function stripHtml(html) {
        const temp = document.createElement("div");
        temp.innerHTML = html;
        return temp.textContent || temp.innerText || "";
    }
    const handleSend = async (text) => {
        if (!text.trim()) return;

        const cleanedQuestion = stripHtml(text.trim());

        setMessages((prev) => [
            ...prev,
            {
                type: "text",
                message: cleanedQuestion,
                sender: "user",
                direction: "outgoing",
            },
        ]);

        setIsTyping(true);

        const newBotMessage = {
            type: "text",
            message: "",
            sender: "bot",
            direction: "incoming",
        };

        setMessages((prev) => [...prev, newBotMessage]);
        const messageIndex = messages.length + 1;

        try {
            const form = new FormData();
            form.append("question", cleanedQuestion);

            const res = await fetch("http://localhost:8000/ask", {
                method: "POST",
                body: form,
            });

            if (!res.body) throw new Error("Không có phản hồi từ server");

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let result = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                result += chunk;

                // Cập nhật tin nhắn bot khi đang stream
                setMessages((prev) => {
                    const updated = [...prev];
                    updated[messageIndex] = {
                        ...updated[messageIndex],
                        message: result,
                    };
                    return updated;
                });
            }
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
                <h3>💬 AI</h3>
            </div>

            <MainContainer style={{ flex: 1 }}>
                <ChatContainer>
                    <MessageList
                        typingIndicator={
                            isTyping ? <TypingIndicator content="Đang xử lý..." /> : null
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
                                {m.sender === "user" && <Avatar name="User" src="/user.jpg" />}
                            </Message>
                        ))}
                    </MessageList>

                    <MessageInput
                        placeholder="Nhập câu hỏi..."
                        onSend={handleSend}
                        attachButton={false}
                    />
                </ChatContainer>
            </MainContainer>
        </div>
    );
}
