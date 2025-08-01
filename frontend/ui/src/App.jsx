import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import ChatPageNew from "./pages/ChatPageNew";
import UploadPage from "./pages/UploadPage";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat-new" element={<ChatPageNew />} />
        <Route path="/upload" element={<UploadPage />} />
      </Routes>
    </Router>
  );
}
