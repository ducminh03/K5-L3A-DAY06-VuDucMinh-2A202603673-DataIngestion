"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split page text into fixed-size windows and retain its metadata."""
    if size <= 0:
        raise ValueError("size phải lớn hơn 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap dài hơn size, không hợp lệ")

    return [
        Chunk(
            chunk_id=f"{page.document_id}:p{page.page}:fixed:{size}:{overlap}:{start}",
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=page.text[start:start + size],
            metadata=dict(page.metadata),
        )
        for start in range(0, len(page.text), size - overlap)
        if (start == 0 or start + overlap < len(page.text))
        and page.text[start:start + size].strip()
    ]

def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Split page text by paragraphs, splitting oversized paragraphs evenly."""
    import math
    import re
    if max_size <= 0:
        raise ValueError("max_size phải lớn hơn 0")

    chunks: list[Chunk] = []

    for match in re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", page.text):
        paragraph = match.group()
        paragraph_start = match.start()

        parts_count = math.ceil(len(paragraph) / max_size)
        base_size, extra = divmod(len(paragraph), parts_count)

        offset = 0
        for part_index in range(parts_count):
            part_size = base_size + (1 if part_index < extra else 0)
            raw_text = paragraph[offset:offset + part_size]
            text = raw_text.strip()

            if text:
                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{page.document_id}:p{page.page}:"
                            f"by_paragraph:{max_size}:{paragraph_start + offset}"
                        ),
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=text,
                        metadata=dict(page.metadata),
                    )
                )

            offset += part_size

    return chunks