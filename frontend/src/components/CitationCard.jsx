import './CitationCard.css';

export default function CitationCard({ citation }) {
    const scorePercent = Math.round(citation.score * 100);

    return (
        <div className="citation-card glass-card" id={`citation-${citation.chunk_id}`}>
            <div className="citation-card-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                </svg>
                <span className="citation-card-doc">{citation.document}</span>
            </div>
            <div className="citation-card-details">
                <span className="badge badge-accent">Page {citation.page}</span>
                <span className="citation-score" title="Similarity score">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
                    {scorePercent}%
                </span>
            </div>
        </div>
    );
}
