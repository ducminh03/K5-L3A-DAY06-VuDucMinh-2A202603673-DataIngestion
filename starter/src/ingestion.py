"""Read and normalize source data. TODO: implement functions."""
import json
from pathlib import Path

from .models import DocumentPage


def load_pdf() -> list[DocumentPage]:
    """Load every PDF listed in the data manifest and flatten its pages."""
    import pypdf

    manifest_path = Path(__file__).resolve().parent.parent / "data" / "manifest.jsonl"
    pages: list[DocumentPage] = []

    with manifest_path.open(encoding="utf-8") as manifest:
        for line in manifest:
            if not line.strip():
                continue
            document = json.loads(line)
            path = manifest_path.parent / document["source_path_or_url"]
            reader = pypdf.PdfReader(str(path))
            document_pages: list[DocumentPage] = []
            for idx, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = normalize_text(raw_text)
                if not cleaned:
                    print(f"[Ingestion] Cảnh báo: Trang {idx} rỗng, đã bỏ qua.")
                    continue
                document_pages.append(
                    DocumentPage(
                        document_id=document["source_id"],
                        source=document["source_path_or_url"],
                        page=idx,
                        text=cleaned,
                        metadata=document["metadata"],
                    )
                )
            print(
                f"[Ingestion] Đã nạp thành công {len(document_pages)} trang có nội dung "
                f"từ {path.name}"
            )
            pages.extend(document_pages)

    return pages


def normalize_text(text: str) -> str:
    import re
    import unicodedata
    """Normalize Unicode and whitespace while preserving paragraph structure."""
    # Chuẩn hóa các dạng Unicode tương đương, nhưng không biến đổi ký tự có nghĩa.
    text = unicodedata.normalize("NFC", text)

    # Đồng nhất kiểu xuống dòng.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Đổi các khoảng trắng Unicode không phải xuống dòng thành space thường.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Bỏ space thừa ở đầu/cuối từng dòng nhưng giữ các dòng và paragraph.
    text = re.sub(r" *\n *", "\n", text)

    # Giữ tối đa một dòng trống giữa các paragraph.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
