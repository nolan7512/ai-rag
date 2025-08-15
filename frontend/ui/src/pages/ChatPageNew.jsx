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

const VITE_SERVER_API = import.meta.env.VITE_SERVER_API || "http://localhost:8000";
// Loại bỏ HTML thô người dùng dán vào
function stripHtml(html) {
  if (!html) return "";
  const tmp = document.createElement("div");
  tmp.innerHTML = html;
  return (tmp.textContent || tmp.innerText || "").trim();
}
function normalizeMarkdown(text = "") {
  try {
    return he.decode(
      text
        .replaceAll(/\\"/g, '"')     // \" => "
        .replaceAll(/\\'/g, "'")     // \' => '
        .replaceAll(/\\n/g, "\n")    // \n => newline
        .replaceAll(/\\t/g, "\t")    // \t => tab
        .replaceAll(/\\r/g, "\r")    // 
        // .replaceAll(/\\s/g, "\s")    // \s => space
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
// Xây history JSON messages ổn định cho backend
function buildHistoryJSON(messages) {
  const msgs = messages
    .filter((m) => m.message && m.message.trim() !== "")
    .map((m) => ({
      role: m.sender === "user" ? "user" : "assistant",
      content: m.message,
    }));
  return JSON.stringify({ messages: msgs });
}

/* ------------------------- Component ------------------------- */

export default function ChatPageNew() {
  const [messages, setMessages] = useState([
    // { sender: "bot", direction: "incoming", message: "Xin chào! Mình là Ani 👋", isStreaming: false }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [useStream, setUseStream] = useState(true);

  const abortControllerRef = useRef(null);
  const messageListRef = useRef(null);

  // Auto scroll xuống cuối khi có tin nhắn mới
  useEffect(() => {
    const el = messageListRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const appendUserMessage = (text) => {
    setMessages((prev) => [
      ...prev,
      { sender: "user", direction: "outgoing", message: text },
    ]);
  };

  const appendBotMessageShell = () => {
    const newBotMsg = {
      sender: "bot",
      direction: "incoming",
      message: "",
      isStreaming: useStream,
    };
    setMessages((prev) => [...prev, newBotMsg]);
  };

  const updateLastBotMessage = (patch) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].sender === "bot") {
          next[i] = { ...next[i], ...patch };
          break;
        }
      }
      return next;
    });
  };

  const quickTranslate = async (target) => {
    // target: "tiếng Anh" | "tiếng Việt" | "tiếng Trung"
    const cmd = `dịch câu trả lời sang ${target}`;
    await handleSend(cmd);
  };

  const stopStreaming = () => {
    try {
      abortControllerRef.current?.abort();
    } catch { }
  };

  const handleSend = async (raw) => {
    const text = stripHtml((raw ?? input).trim());
    if (!text || isGenerating) return;

    // đẩy tin người dùng
    appendUserMessage(text);
    setInput("");
    setIsTyping(true);
    setIsGenerating(true);

    // tạo ô tin của bot (rỗng) để stream đổ về
    appendBotMessageShell();

    try {
      const form = new FormData();
      form.append("question", text);
      form.append("stream", String(useStream));
      form.append("history", buildHistoryJSON(messages));

      abortControllerRef.current = new AbortController();
      const res = await fetch(`${VITE_SERVER_API}/ask`, {
        method: "POST",
        body: form,
        signal: abortControllerRef.current.signal,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      // STREAM MODE
      if (useStream && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          // Ghép chunk vào buffer hiện tại
          buffer += chunk;
          // const live = buffer.replace(/\\n/g, "\n"); // unescape live
          // updateLastBotMessage({ message: live, isStreaming: true });
          updateLastBotMessage({ message: buffer, isStreaming: true });
        }

        // Kết thúc stream
        updateLastBotMessage({
          message: normalizeMarkdown(buffer),
          isStreaming: false,
        });
      } else {
        // NON-STREAM MODE (backend trả JSON)
        const data = await res.json();
        const answer = data?.answer || "⚠️ Không có nội dung.";
        updateLastBotMessage({
          message: normalizeMarkdown(answer),
          isStreaming: false,
        });
      }
    } catch (err) {
      console.error("Stream error:", err);
      updateLastBotMessage({
        message:
          "⚠️ Đã xảy ra lỗi khi gọi backend. Hãy thử lại hoặc kiểm tra server.",
        isStreaming: false,
      });
    } finally {
      setIsTyping(false);
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="h-screen w-full flex flex-col bg-gray-50">
      {/* Header */}
      <header className="p-4 border-b bg-white shadow flex items-center gap-3 flex-wrap">
        <h1 className="text-lg font-semibold text-gray-800 mr-auto">
          💬 Trợ lý doanh nghiệp
        </h1>

        <label className="text-sm flex items-center gap-2">
          <input
            type="checkbox"
            checked={useStream}
            onChange={(e) => setUseStream(e.target.checked)}
          />
          Stream
        </label>

        <div className="flex items-center gap-2">
          <button
            onClick={() => quickTranslate("tiếng Anh")}
            className="px-2 py-1 border rounded text-sm"
            title="Dịch câu trả lời gần nhất sang tiếng Anh"
            disabled={isGenerating}
          >
            EN
          </button>
          <button
            onClick={() => quickTranslate("tiếng Việt")}
            className="px-2 py-1 border rounded text-sm"
            title="Dịch câu trả lời gần nhất sang tiếng Việt"
            disabled={isGenerating}
          >
            VI
          </button>
          <button
            onClick={() => quickTranslate("tiếng Trung")}
            className="px-2 py-1 border rounded text-sm"
            title="Dịch câu trả lời gần nhất sang tiếng Trung"
            disabled={isGenerating}
          >
            ZH
          </button>
        </div>

        <button
          onClick={stopStreaming}
          disabled={!isGenerating}
          className="px-2 py-1 border rounded text-sm disabled:opacity-50"
          title="Dừng sinh nội dung"
        >
          ⏹ Stop
        </button>
      </header>

      {/* Message list */}
      <div
        ref={messageListRef}
        className="flex-1 overflow-y-auto p-4 space-y-3"
      >
        {messages.map((m, idx) => {
          const isUser = m.sender === "user";
          const bubbleCls = isUser
            ? "bg-blue-600 text-white"
            : "bg-white text-gray-900 border";
          return (
            <div
              key={idx}
              className={`w-full flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-3 py-2 shadow-sm ${bubbleCls}`}
              >
                {/* Khi đang stream thì hiển thị dạng <pre> để xuống dòng tức thì */}
                {m.isStreaming ? (
                  <pre className="whitespace-pre-wrap text-sm">
                    {m.message}
                  </pre>
                ) : (
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown
                      children={normalizeMarkdown(m.message)}
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        table({ children }) {
                          return (
                            <div className="overflow-x-auto my-2">
                              <table className="table-auto border-collapse border border-gray-300 w-full">
                                {children}
                              </table>
                            </div>
                          );
                        },
                        thead({ children }) {
                          return <thead className="bg-gray-100">{children}</thead>;
                        },
                        th({ children }) {
                          return (
                            <th className="border border-gray-300 px-2 py-1 text-left">
                              {children}
                            </th>
                          );
                        },
                        td({ children }) {
                          return (
                            <td className="border border-gray-300 px-2 py-1 whitespace-pre-wrap align-top">
                              {children}
                            </td>
                          );
                        },
                        code({ node, inline, className, children, ...props }) {
                          const content = extractTextFromReactNode(children);
                          // Inline code -> span
                          if (inline) {
                            return (
                              <code className={`px-1 py-0.5 rounded bg-gray-100 ${className || ""}`} {...props}>
                                {children}
                              </code>
                            );
                          }
                          // Code block -> pre + copy button
                          return (
                            <div className="relative group my-2">
                              <pre className="rounded-md bg-gray-100 p-2 whitespace-pre-wrap break-words overflow-x-auto">
                                <code className={className} {...props}>
                                  {children}
                                </code>
                              </pre>
                              <button
                                type="button"
                                onClick={() => copyToClipboard(content)}
                                aria-label="Copy code"
                                className="absolute top-1 right-2 text-xs text-gray-600 bg-white/80 backdrop-blur px-2 py-1 border rounded opacity-0 group-hover:opacity-100"
                              >
                                📋 Copy
                              </button>
                            </div>
                          );
                        },
                      }}
                    />
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {isTyping && (
          <div className="text-sm text-gray-500">Ani đang soạn…</div>
        )}
      </div>

      {/* Input */}
      <div className="border-t bg-white p-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center gap-2"
        >
          <input
            className="flex-1 border rounded-lg px-3 py-2 outline-none focus:ring w-full"
            placeholder="Nhập câu hỏi… (vd: xin chào Ani)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isGenerating}
          />
          <button
            type="submit"
            className="px-4 py-2 rounded-lg bg-blue-600 text-white disabled:opacity-50"
            disabled={!input.trim() || isGenerating}
          >
            Gửi
          </button>
        </form>
      </div>
    </div>
  );
}
