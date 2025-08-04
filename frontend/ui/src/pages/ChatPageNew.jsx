import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import he from "he";
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

function normalizeMarkdown(text = "") {
  try {
    return he.decode(
      text
        .replaceAll(/\\"/g, '"')     // \" => "
        .replaceAll(/\\'/g, "'")     // \' => '
        .replaceAll(/\\n/g, "\n")    // \n => newline
        .replaceAll(/\\t/g, "\t")    // \t => tab
        .replaceAll(/\\\\/g, "\\")   // \\ => \
        .replaceAll(/\\u003c/g, "<") // \u003c => <
        .replaceAll(/\\u003e/g, ">") // \u003e => >
        .replaceAll(/\\u0026/g, "&") // \u0026 => &
    );
  } catch {
    return text;
  }
}


function extractTextFromReactNode(node) {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }

  if (Array.isArray(node)) {
    return node.map(extractTextFromReactNode).join("");
  }

  if (React.isValidElement(node)) {
    return extractTextFromReactNode(node.props.children);
  }

  return "";
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const abortControllerRef = useRef(null);
  const messageListRef = useRef(null);
  const [useStream, setUseStream] = useState(true);
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [messages]);

  const stripHtml = (html) => {
    const temp = document.createElement("div");
    temp.innerHTML = html;
    return temp.textContent || temp.innerText || "";
  };

  const handleSend = async (text) => {
    if (!text.trim()) return;
    const cleaned = stripHtml(text.trim());

    setMessages((prev) => [
      ...prev,
      { sender: "user", direction: "outgoing", message: cleaned },
    ]);
    setIsTyping(true);

    const newBotMsg = {
      sender: "bot",
      direction: "incoming",
      message: "",
      isStreaming: useStream,
    };
    const msgIndex = messages.length + 1;
    setMessages((prev) => [...prev, newBotMsg]);

    try {
      const form = new FormData();
      form.append("question", cleaned);
      const memory = messages
        .map((m) => `${m.sender === "user" ? "Người dùng" : "Bot"}: ${m.message}`)
        .join("\n");
      form.append("history", memory);
      form.append("stream", useStream.toString());

      abortControllerRef.current = new AbortController();
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        body: form,
        signal: abortControllerRef.current.signal,
      });

      if (useStream) {
        // Stream mode
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          setMessages((prev) => {
            const updated = [...prev];
            if (!updated[msgIndex]) return updated;
            updated[msgIndex].message = buffer;
            return updated;
          });
        }

        setMessages((prev) => {
          const updated = [...prev];
          if (!updated[msgIndex]) return updated;
          updated[msgIndex].isStreaming = false;
          return updated;
        });

      } else {
        // Non-stream mode: receive full response
        const result = await res.json();
        setMessages((prev) => {
          const updated = [...prev];
          if (!updated[msgIndex]) return updated;
          updated[msgIndex].message = result.answer || "Không có câu trả lời.";
          updated[msgIndex].isStreaming = false;
          return updated;
        });
      }

    } catch (err) {
      console.error("Stream error:", err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          direction: "incoming",
          message: "⚠️ Đã xảy ra lỗi khi gọi backend.",
          isStreaming: false,
        },
      ]);
    }

    setIsTyping(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="p-4 border-b bg-white shadow">
        <h1 className="text-lg font-semibold text-gray-800">💬 Trợ lý doanh nghiệp</h1>
      </header>

      <div className="flex-1 flex flex-col overflow-hidden">
        <MainContainer className="flex-1 overflow-hidden">
          <ChatContainer>
            <MessageList
              typingIndicator={isTyping ? <TypingIndicator content="Đang xử lý..." /> : null}
              ref={messageListRef}
              className="overflow-y-auto"
            >
              <MessageSeparator content="Hôm nay" />
              {messages.filter((m) => m.message.trim() !== "").map((m, i) => (
                <Message
                  key={i}
                  model={{ sender: m.sender, direction: m.direction }}
                >
                  <Message.CustomContent>
                    {m.isStreaming ? (
                      <pre className="rounded-md bg-gray-100 p-2 whitespace-pre-wrap break-words">
                        {/* {normalizeMarkdown(m.message)} */}
                        {m.message}
                      </pre>
                    ) : (
                      <ReactMarkdown
                        children={normalizeMarkdown(m.message)}
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                        components={{
                          table({ children }) {
                            return (
                              <div className="overflow-x-auto">
                                <table className="table-auto border-collapse border border-gray-300 w-full">
                                  {children}
                                </table>
                              </div>
                            );
                          },
                          th({ children }) {
                            return (
                              <th className="border border-gray-300 px-2 py-1 bg-gray-100 text-left">
                                {children}
                              </th>
                            );
                          },
                          td({ children }) {
                            return (
                              <td className="border border-gray-300 px-2 py-1 whitespace-pre-wrap">
                                {children}
                              </td>
                            );
                          },
                          code({ node, inline, className, children, ...props }) {
                            const content = extractTextFromReactNode(children);
                            return (
                              <div className="relative group">
                                <pre className="rounded-md bg-gray-100 p-2 whitespace-pre-wrap break-words">
                                  <code className={className} {...props}>
                                    {children}
                                  </code>
                                </pre>
                                <button
                                  onClick={() => navigator.clipboard.writeText(content)}
                                  className="absolute top-1 right-2 text-xs text-gray-500 bg-white px-2 py-1 border rounded opacity-0 group-hover:opacity-100"
                                >
                                  📋 Copy
                                </button>
                              </div>
                            );
                          },
                        }}

                      />
                    )}
                  </Message.CustomContent>
                  <Avatar
                    name={m.sender === "user" ? "User" : "Bot"}
                    src={`/${m.sender}.jpg`}
                  />
                </Message>
              ))}
            </MessageList>
            <MessageInput
              placeholder="Nhập câu hỏi..."
              onSend={handleSend}
              attachButton={true}
            />
          </ChatContainer>
        </MainContainer>

      </div>
    </div>
  );
}
