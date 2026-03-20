import { useState, useRef } from 'react';
import { uploadDocument } from '../utils/api';
import './UploadPanel.css';

export default function UploadPanel({ onUploadSuccess }) {
    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadResult, setUploadResult] = useState(null);
    const [error, setError] = useState(null);
    const [documents, setDocuments] = useState([]);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const files = Array.from(e.dataTransfer.files);
        const pdfFile = files.find((f) => f.name.toLowerCase().endsWith('.pdf'));
        if (pdfFile) handleUpload(pdfFile);
        else setError('Please upload a PDF file.');
    };

    const handleFileSelect = (e) => {
        const file = e.target.files?.[0];
        if (file) handleUpload(file);
    };

    const handleUpload = async (file) => {
        setUploading(true);
        setError(null);
        setUploadResult(null);

        try {
            const result = await uploadDocument(file);
            setUploadResult(result);
            setDocuments((prev) => [
                { id: result.document_id, title: result.title, chunks: result.chunks_created, pages: result.total_pages, time: result.processing_time_ms },
                ...prev,
            ]);
            onUploadSuccess?.(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="upload-panel">
            <div className="upload-panel-header">
                <div className="upload-panel-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                    </svg>
                </div>
                <h2 className="upload-panel-title">Documents</h2>
                <span className="badge badge-accent">{documents.length}</span>
            </div>

            {/* Drop Zone */}
            <div
                className={`drop-zone ${isDragging ? 'drop-zone--active' : ''} ${uploading ? 'drop-zone--uploading' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !uploading && fileInputRef.current?.click()}
                id="upload-dropzone"
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                    id="file-input"
                />

                {uploading ? (
                    <div className="drop-zone-content">
                        <div className="spinner" />
                        <span className="drop-zone-text">Processing document...</span>
                    </div>
                ) : (
                    <div className="drop-zone-content">
                        <svg className="drop-zone-icon" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <span className="drop-zone-text">Drop PDF here or click to browse</span>
                        <span className="drop-zone-hint">Supports .pdf files</span>
                    </div>
                )}
            </div>

            {/* Status messages */}
            {error && (
                <div className="upload-status upload-status--error animate-fade-in">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" /></svg>
                    <span>{error}</span>
                </div>
            )}

            {uploadResult && (
                <div className="upload-status upload-status--success animate-fade-in">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>
                    <span>{uploadResult.chunks_created} chunks indexed in {(uploadResult.processing_time_ms / 1000).toFixed(1)}s</span>
                </div>
            )}

            {/* Document list */}
            <div className="document-list">
                {documents.map((doc) => (
                    <div key={doc.id} className="document-item glass-card animate-fade-in" id={`doc-${doc.id}`}>
                        <div className="document-item-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                        </div>
                        <div className="document-item-info">
                            <span className="document-item-title">{doc.title}</span>
                            <span className="document-item-meta">{doc.pages} pages · {doc.chunks} chunks</span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
