"""Baseline source chunk preview, not an embedding or retrieval implementation."""


def chunk_pages(pages, max_chars=900, overlap=150):
    if not 0 <= overlap < max_chars:
        raise ValueError('INVALID_CHUNK_WINDOW')
    chunks = []
    for page, body in enumerate(pages, 1):
        start = 0
        while start < len(body):
            end = min(start + max_chars, len(body))
            if end < len(body):
                boundary = body.rfind('\n', start + max_chars // 2, end)
                if boundary > start:
                    end = boundary + 1
            text = body[start:end]
            if text.strip():
                chunks.append({'chunkIndex': len(chunks), 'page': page, 'start': start,
                               'end': end, 'body': text})
            if end == len(body):
                break
            start = max(start + 1, end - overlap)
    return chunks
