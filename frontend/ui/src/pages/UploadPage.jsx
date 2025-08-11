import { useEffect, useState } from "react";
import axios from "axios";
import ModalEditText from "./ModalEditText"; // đường dẫn tuỳ bạn
const VITE_SERVER_API = import.meta.env.VITE_SERVER_API || "http://localhost:8000";

export default function UploadPage() {
    const [files, setFiles] = useState([]);
    const [tags, setTags] = useState("");
    const [importFile, setImportFile] = useState(null);
    const [list, setList] = useState([]);
    const [refreshFlag, setRefreshFlag] = useState(false);
    const [textContent, setTextContent] = useState("");
    const [editingId, setEditingId] = useState(null);
    const [showEditModal, setShowEditModal] = useState(false);
    const [editingTags, setEditingTags] = useState([]);
    const fetchList = async () => {
        const res = await axios.get(`${VITE_SERVER_API}/list_files`);
        setList(res.data);
    };

    useEffect(() => {
        fetchList();
    }, [refreshFlag]);

    const handleEditText = async (file_id, tags = []) => {
        const form = new FormData();
        form.append("file_id", file_id);
        const res = await axios.post(`${VITE_SERVER_API}/get_text_by_file_id`, form);
        setTextContent(res.data.text);
        setEditingId(file_id);
        setEditingTags((tags || []).join(", "));
        setShowEditModal(true);
    };


    const handleUploadFile = async () => {
        const formData = new FormData();
        files.forEach(f => formData.append("files", f));
        tags.split(",").forEach(tag => formData.append("tags", tag.trim()));
        await axios.post(`${VITE_SERVER_API}/upload`, formData);
        alert("Tải file thành công!");
        setRefreshFlag(!refreshFlag);
    };

    const handleUploadText = async () => {
        const formData = new FormData();
        formData.append("text", textContent);
        tags.split(",").forEach(tag => formData.append("tags", tag.trim()));
        await axios.post(`${VITE_SERVER_API}/upload_text`, formData);
        alert("Tải text thành công!");
        setTextContent("");
        setTags("");
        setRefreshFlag(!refreshFlag);
    };

    const handleDelete = async (file_id) => {
        const form = new FormData();
        form.append("file_id", file_id);
        await axios.post(`${VITE_SERVER_API}/delete_by_id`, form); // dùng chung cho file/text
        alert("Đã xoá file/text.");
        setRefreshFlag(!refreshFlag);
    };



    const handleUpdateText = async () => {
        try {
            const form = new FormData();
            form.append("file_id", editingId);
            form.append("content", textContent); // 👉 dùng state text đang edit

            editingTags.split(",").forEach(tag => {
                const trimmed = tag.trim();
                if (trimmed) form.append("tags", trimmed);
            });

            await axios.post(`${VITE_SERVER_API}/update_text`, form);

            alert("Đã cập nhật!");
            setShowEditModal(false);
            setEditingId(null);
            setTextContent("");
            setEditingTags("");
            setRefreshFlag(!refreshFlag);
        } catch (err) {
            console.error("Lỗi cập nhật:", err);
            alert("❌ Có lỗi khi cập nhật.");
        }
    };

    const handleUpdateFile = async (file, file_id) => {
        const form = new FormData();
        form.append("file", file);
        form.append("file_id", file_id);
        tags.split(",").forEach(tag => form.append("tags", tag.trim()));
        await axios.post(`${VITE_SERVER_API}/update_file`, form);
        alert("Đã cập nhật file!");
        setRefreshFlag(!refreshFlag);
    };

    return (
        <div className="p-6 space-y-4">
            <h1 className="text-xl font-bold">📤 Quản lý tài liệu</h1>

            {/* Upload file */}
            <input type="file" multiple onChange={e => setFiles([...e.target.files])} />
            <textarea value={tags} onChange={e => setTags(e.target.value)} placeholder="Tag (phân cách bằng dấu phẩy)" className="border p-2 w-full" />
            <button onClick={handleUploadFile} className="bg-blue-600 text-white px-4 py-2 rounded">Tải lên file</button>

            {/* Upload text */}
            <textarea value={textContent} onChange={e => setTextContent(e.target.value)} placeholder="Nhập nội dung..." className="border p-2 w-full h-[150px]" />
            {editingId ? (
                <button onClick={handleUpdateText} className="bg-green-600 text-white px-4 py-2 rounded">💾 Lưu chỉnh sửa</button>
            ) : (
                <button onClick={handleUploadText} className="bg-purple-600 text-white px-4 py-2 rounded">✍️ Tải lên Text</button>
            )}

            {/* DB actions */}
            <div className="flex flex-col gap-4 mt-4">
                <div className="flex flex-1 gap-4">
                    <button onClick={async () => {
                        const res = await fetch("http://localhost:8000/clear_db", { method: "POST" });
                        const result = await res.json();
                        alert(result.message || result.error);
                        setRefreshFlag(!refreshFlag);
                    }} className="bg-red-500 text-white px-4 py-2 rounded">🗑️ Xoá toàn bộ DB</button>
                </div>
                <div className="flex flex-1 gap-4">
                    <input type="file" onChange={e => setImportFile(e.target.files[0])} />

                </div>
                <div className="flex flex-1 gap-4">
                    <button onClick={async () => {
                        const form = new FormData();
                        form.append("file", importFile);
                        await axios.post("http://localhost:8000/import_db", form);
                        alert("Import thành công!");
                    }} className="bg-blue-600 text-white px-4 py-2 rounded">📥 Nhập DB</button>

                    <a
                        href="http://localhost:8000/export_db"
                        download="qdrant_export.json"
                        className="bg-blue-600 text-white px-4 py-2 rounded"
                    >
                        📤 Xuất DB
                    </a>

                </div>

            </div>

            {/* Table list */}
            <table className="w-full border mt-6">
                <thead>
                    <tr className="bg-gray-100">
                        <th className="border px-2 py-2">Loại</th>
                        <th className="border px-2 py-2">Tên</th>
                        <th className="border px-2 py-2">Tags</th>
                        <th className="border px-2 py-2">Thời gian</th>
                        <th className="border px-2 py-2">Hành động</th>
                    </tr>
                </thead>
                <tbody>
                    {list.map(row => (
                        <tr key={row.file_id} className="text-sm">
                            <td className="border px-2 py-2 text-center">{row.source}</td>
                            <td className="border px-2 py-2">{row.filename}</td>
                            <td className="border px-2 py-2">{(row.tags || []).join(", ")}</td>
                            <td className="border px-2 py-2">{row.uploaded_at?.slice(0, 19).replace("T", " ")}</td>
                            <td className="border px-2 py-2 space-x-2">
                                <div className="flex flex-col md:flex-row gap-2">

                                    {row.source === "text" ? (
                                        <button
                                            onClick={() => handleEditText(row.file_id, row.tags)}
                                            className="bg-yellow-400 px-2 py-2 rounded"
                                        >
                                            ✏️ Sửa
                                        </button>
                                    ) : (
                                        <input
                                            type="file"
                                            onChange={e => handleUpdateFile(e.target.files[0], row.file_id)}
                                            className="text-sm"
                                        />
                                    )}
                                    <button onClick={() => handleDelete(row.file_id)} className="bg-red-500 text-white px-2 py-2 rounded">Xoá</button>
                                </div>

                            </td>
                        </tr>

                    ))}
                </tbody>
            </table>
            <ModalEditText
                isOpen={showEditModal}
                onClose={() => {
                    setShowEditModal(false);
                    setEditingId(null);
                }}
                onSave={async () => {
                    await handleUpdateText();
                    setShowEditModal(false);
                }}
                text={textContent}
                tags={editingTags}
                setText={setTextContent}
                setTags={setEditingTags}
            />

        </div>

    );

}
