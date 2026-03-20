import { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import { queryAssistant } from '../utils/api';
import './ChatInterface.css';

export default function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const query = input.trim();
        if (!query || isLoading) return;

        // Add user message
        const userMsg = { id: Date.now(), role: 'user', content: query };
        const loadingMsg = { id: Date.now() + 1, role: 'assistant', loading: true };

        setMessages((prev) => [...prev, userMsg, loadingMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const result = await queryAssistant(query);
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === loadingMsg.id
                        ? {
                            ...msg,
                            loading: false,
                            content: result.answer,
                            citations: result.citations,
                            metrics: result.metrics,
                        }
                        : msg
                )
            );
        } catch (err) {
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === loadingMsg.id
                        ? {
                            ...msg,
                            loading: false,
                            content: '',
                            error: err.message || 'An error occurred.',
                        }
                        : msg
                )
            );
        } finally {
            setIsLoading(false);
            inputRef.current?.focus();
        }
    };

    return (
        <div className="chat-interface">
            {/* Header */}
            <div className="chat-header">
                <div className="chat-header-info">
                    <div className="chat-header-avatar">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 2a4 4 0 0 1 4 4v1a2 2 0 0 1 2 2v1a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V9a2 2 0 0 1 2-2V6a4 4 0 0 1 4-4z" />
                            <path d="M9 18h6" />
                            <path d="M10 22h4" />
                        </svg>
                    </div>
                    <div>
                        <h1 className="chat-header-title">Enterprise RAG Assistant</h1>
                        <p className="chat-header-subtitle">AI-powered document Q&A with grounded retrieval</p>
                    </div>
                </div>
                <span className="badge badge-success">Online</span>
            </div>

            {/* Messages */}
            <div className="chat-messages" id="chat-messages">
                {messages.length === 0 ? (
                    <div className="chat-empty">
                        <div className="chat-empty-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                            </svg>
                        </div>
                        <h2 className="chat-empty-title">Ask a question</h2>
                        <p className="chat-empty-text">
                            Upload documents using the panel on the left, then ask questions about their content. Answers are grounded in your documents with full citations.
                        </p>
                        <div className="chat-empty-suggestions">
                            {[
                                'What are the key findings in the report?',
                                'Summarize the main conclusions',
                                'What metrics are discussed?',
                            ].map((suggestion, i) => (
                                <button
                                    key={i}
                                    className="btn btn-ghost chat-suggestion"
                                    onClick={() => { setInput(suggestion); inputRef.current?.focus(); }}
                                    id={`suggestion-${i}`}
                                >
                                    {suggestion}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="chat-input-container">
                <form className="chat-input-form" onSubmit={handleSubmit}>
                    <input
                        ref={inputRef}
                        type="text"
                        className="chat-input"
                        placeholder="Ask a question about your documents..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        disabled={isLoading}
                        id="chat-input"
                        autoFocus
                    />
                    <button
                        type="submit"
                        className="btn btn-primary chat-send-btn"
                        disabled={!input.trim() || isLoading}
                        id="send-button"
                    >
                        {isLoading ? (
                            <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                        ) : (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13" />
                                <polygon points="22 2 15 22 11 13 2 9 22 2" />
                            </svg>
                        )}
                    </button>
                </form>
                <p className="chat-disclaimer">Responses are grounded in uploaded documents. Always verify critical information.</p>
            </div>
        </div>
    );
}
