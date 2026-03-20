import CitationCard from './CitationCard';
import './MessageBubble.css';

export default function MessageBubble({ message }) {
    const isUser = message.role === 'user';

    return (
        <div className={`message-bubble ${isUser ? 'message-bubble--user' : 'message-bubble--assistant'} animate-fade-in`}>
            {/* Avatar */}
            <div className={`message-avatar ${isUser ? 'message-avatar--user' : 'message-avatar--assistant'}`}>
                {isUser ? (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                        <circle cx="12" cy="7" r="4" />
                    </svg>
                ) : (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 2a4 4 0 0 1 4 4v1a2 2 0 0 1 2 2v1a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V9a2 2 0 0 1 2-2V6a4 4 0 0 1 4-4z" />
                        <path d="M9 18h6" />
                        <path d="M10 22h4" />
                    </svg>
                )}
            </div>

            {/* Content */}
            <div className="message-content">
                <div className="message-header">
                    <span className="message-role">{isUser ? 'You' : 'RAG Assistant'}</span>
                    {message.metrics && (
                        <span className="message-timing badge badge-accent">
                            {message.metrics.total_latency_ms.toFixed(0)}ms
                        </span>
                    )}
                </div>

                <div className="message-text">
                    {message.loading ? (
                        <div className="message-loading">
                            <div className="spinner" />
                            <span>Analyzing documents...</span>
                        </div>
                    ) : (
                        <p>{message.content}</p>
                    )}
                </div>

                {/* Citations */}
                {message.citations && message.citations.length > 0 && (
                    <div className="message-citations">
                        <span className="message-citations-label">Sources</span>
                        <div className="message-citations-grid">
                            {message.citations.map((cite, i) => (
                                <CitationCard key={i} citation={cite} />
                            ))}
                        </div>
                    </div>
                )}

                {/* Metrics bar */}
                {message.metrics && (
                    <div className="message-metrics">
                        <span className="metric-item" title="Retrieval time">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
                            {message.metrics.retrieval_latency_ms.toFixed(0)}ms
                        </span>
                        <span className="metric-item" title="Generation time">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
                            {message.metrics.generation_latency_ms.toFixed(0)}ms
                        </span>
                        <span className="metric-item" title="Chunks retrieved">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2" /><rect x="2" y="14" width="20" height="8" rx="2" ry="2" /></svg>
                            {message.metrics.chunks_retrieved} chunks
                        </span>
                        <span className="metric-item" title="Average similarity">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
                            {(message.metrics.avg_similarity * 100).toFixed(1)}%
                        </span>
                    </div>
                )}

                {/* Error */}
                {message.error && (
                    <div className="message-error">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
                        {message.error}
                    </div>
                )}
            </div>
        </div>
    );
}
