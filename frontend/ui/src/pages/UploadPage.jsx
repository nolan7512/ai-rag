import { useState } from "react";
import axios from "axios";

export default function UploadPage() {
    const [files, setFiles] = useState([]);
    const [tags, setTags] = useState("");

    const handleUpload = async () => {
        const formData = new FormData();
        files.forEach(f => formData.append("files", f));
        tags.split(",").forEach(tag => formData.append("tags", tag.trim()));
        await axios.post("http://localhost:8000/upload", formData);
        alert("Tải file thành công!");
    };

    return (
        <div className="p-6 space-y-4">
            <h1 className="text-xl font-bold">📤 Upload tài liệu</h1>
            <input type="file" multiple onChange={e => setFiles([...e.target.files])} />
            <input type="text" value={tags} onChange={e => setTags(e.target.value)} placeholder="Nhập tag, cách nhau bằng dấu phẩy..." className="border p-2 w-full" />
            <button onClick={handleUpload} className="bg-blue-600 text-white px-4 py-2 rounded">Tải lên</button>
        </div>
    );
}
