import { useEffect } from "react";

export default function ModalEditText({
    isOpen,
    onClose,
    onSave,
    text,
    tags,
    setText,
    setTags
}) {
    useEffect(() => {
        setText(text || "");
        setTags(tags || "");
    }, [text, tags, setText, setTags]);


    if (!isOpen) return null;

    return (
        <div className="fixed inset-0  bg-opacity-30 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded shadow-md w-[600px] space-y-4">
                <h2 className="text-lg font-semibold">✏️ Sửa nội dung Text</h2>

                <textarea
                    value={text}
                    onChange={e => setText(e.target.value)}
                    className="w-full h-[200px] border p-2"
                />

                <input
                    type="text"
                    value={tags}
                    onChange={e => setTags(e.target.value)}
                    placeholder="Nhập tags (cách nhau bằng dấu phẩy)"
                    className="border p-2 w-full"
                />

                <div className="flex justify-end space-x-3">
                    <button
                        onClick={onSave}
                        className="bg-green-600 text-white px-4 py-2 rounded"
                    >
                        💾 Lưu
                    </button>
                    <button
                        onClick={onClose}
                        className="bg-gray-400 text-white px-4 py-2 rounded"
                    >
                        ❌ Hủy
                    </button>
                </div>
            </div>
        </div>
    );
}
