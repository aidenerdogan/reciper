def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100):
    # Very rough placeholder; will replace with tokenizer-based chunks
    words = text.split()
    step = max(1, max_tokens - overlap)
    chunks = []
    for i in range(0, len(words), step):
        part = " ".join(words[i:i+max_tokens])
        if part:
            chunks.append(part)
    return chunks
