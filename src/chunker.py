import re
from typing import List, Dict


def clean(text: str) -> str:
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def split_sentences(text: str) -> List[str]:
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc: dict,
    max_tokens: int = 500,
    overlap: int = 50,
    min_tokens: int = 500,
    clean_text: bool = True,
) -> List[Dict]:
    text = doc.get("text", "")
    if clean_text:
        text = clean(text)

    token_count = doc.get("token_count")
    if token_count is None:
        token_count = estimate_tokens(text)

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0
    char_pos = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens > max_tokens and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_start = char_pos - len(chunk_text)

            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "token_count": current_tokens,
                    "char_range": (chunk_start, char_pos),
                    "sentence_count": len(current_chunk),
                    "doc_id": doc.get("id"),
                    "url": doc.get("url"),
                }
            )

            overlap_buffer = []
            overlap_tokens = 0

            for sent in reversed(current_chunk):
                sent_tokens = estimate_tokens(sent)
                if overlap_tokens + sent_tokens <= overlap:
                    overlap_buffer.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break

            current_chunk = overlap_buffer
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += sent_tokens
        char_pos += len(sentence) + 1

    if current_chunk and current_tokens >= min_tokens:
        chunk_text = " ".join(current_chunk)
        chunk_start = char_pos - len(chunk_text)

        chunks.append(
            {
                "chunk_id": len(chunks),
                "text": chunk_text,
                "token_count": current_tokens,
                "char_range": (chunk_start, char_pos),
                "sentence_count": len(current_chunk),
                "doc_id": doc.get("id"),
                "url": doc.get("url"),
            }
        )
    elif current_chunk and chunks:
        last_chunk = chunks[-1]
        merged_text = last_chunk["text"] + " " + " ".join(current_chunk)
        last_chunk["text"] = merged_text
        last_chunk["token_count"] = estimate_tokens(merged_text)
        last_chunk["sentence_count"] += len(current_chunk)
        last_chunk["char_range"] = (last_chunk["char_range"][0], char_pos)

    return chunks
