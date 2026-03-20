const API_BASE = 'http://localhost:8000/api';

/**
 * Upload a PDF document for ingestion.
 * @param {File} file - The PDF file to upload.
 * @param {string} [title] - Optional document title.
 * @returns {Promise<object>} Upload response with document_id, chunks, etc.
 */
export async function uploadDocument(file, title) {
  const formData = new FormData();
  formData.append('file', file);
  if (title) {
    formData.append('title', title);
  }

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Query the knowledge base.
 * @param {string} query - User question.
 * @param {object} [options] - Optional { top_k, score_threshold }.
 * @returns {Promise<object>} Query response with answer, citations, metrics.
 */
export async function queryAssistant(query, options = {}) {
  const body = { query, ...options };

  const response = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Query failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Stream a query response via SSE.
 * @param {string} query - User question.
 * @param {function} onToken - Callback for each token.
 * @param {function} onCitations - Callback for citation data.
 * @param {function} onDone - Callback when streaming is complete.
 * @param {function} onError - Callback on error.
 */
export async function queryAssistantStream(query, { onToken, onCitations, onDone, onError }) {
  try {
    const response = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Stream failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.type === 'token') {
              onToken?.(payload.data);
            } else if (payload.type === 'citations') {
              onCitations?.(payload.data);
            } else if (payload.type === 'done') {
              onDone?.();
            }
          } catch {
            // skip malformed lines
          }
        }
      }
    }

    onDone?.();
  } catch (error) {
    onError?.(error);
  }
}

/**
 * Health check.
 * @returns {Promise<object>} Health status.
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Health check failed');
  return response.json();
}
