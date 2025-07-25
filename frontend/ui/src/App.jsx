
import { useState } from 'react';
import axios from 'axios';

function App() {
  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const uploadDocs = async () => {
    const formData = new FormData();
    files.forEach(f => formData.append("files", f));
    await axios.post("http://localhost:8000/upload", formData);
    alert("Tải file thành công!");
  };

  const handleAsk = async () => {
    const form = new FormData();
    form.append("question", question);
    const res = await axios.post("http://localhost:8000/ask", form);
    setAnswer(res.data.answer);
  };

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold mb-4">Trợ lý văn bản nội bộ</h1>
      <input type="file" multiple onChange={e => setFiles([...e.target.files])} />
      <button onClick={uploadDocs}>Tải lên</button>
      <div className="my-4">
        <input type="text" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Câu hỏi..." />
        <button onClick={handleAsk}>Hỏi</button>
      </div>
      <div><strong>Trả lời:</strong> {answer}</div>
    </div>
  );
}

export default App;
