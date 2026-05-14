"""Document format handlers for dataset generation.

Supports saving documents in native formats: txt, md.
Note: PDF support is only for native PDFs (OpenRAGBench), not text-to-PDF conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DocumentFormat = Literal["txt", "md", "pdf"]


def save_document(
    content: str,
    output_path: Path,
    format: DocumentFormat = "txt",
    metadata: dict | None = None,
) -> Path:
    """Save document content in the specified format.

    Args:
        content: Document text content
        output_path: Output path (extension will be replaced based on format)
        format: Output format - "txt" or "md"
        metadata: Optional metadata dict (used for markdown frontmatter)

    Returns:
        Path to the created file

    Note:
        PDF format is not supported for text conversion.
        Use format="pdf" only with native PDF downloads (OpenRAGBench).
    """
    if format == "txt":
        return _save_txt(content, output_path)
    elif format == "md":
        return _save_markdown(content, output_path, metadata)
    elif format == "pdf":
        raise ValueError(
            "PDF text conversion is not supported. "
            "Use format='txt' or 'md'. "
            "For OpenRAGBench, native PDFs are downloaded directly from ArXiv."
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'txt' or 'md'")


def _save_txt(content: str, output_path: Path) -> Path:
    """Save as plain text file."""
    txt_path = output_path.with_suffix(".txt")
    txt_path.write_text(content, encoding="utf-8")
    return txt_path


def _save_markdown(content: str, output_path: Path, metadata: dict | None = None) -> Path:
    """Save as markdown file with optional YAML frontmatter."""
    md_content = content

    if metadata:
        # Add YAML frontmatter
        frontmatter = "---\n"
        for key, value in metadata.items():
            # Escape special YAML characters in values
            value_str = str(value).replace('"', '\\"')
            frontmatter += f'{key}: "{value_str}"\n'
        frontmatter += "---\n\n"
        md_content = frontmatter + content

    md_path = output_path.with_suffix(".md")
    md_path.write_text(md_content, encoding="utf-8")
    return md_path


def get_file_extension(format: DocumentFormat) -> str:
    """Get file extension for a given format.

    Args:
        format: Document format

    Returns:
        File extension including the dot (e.g., ".txt", ".md", ".pdf")
    """
    return f".{format}"


__all__ = [
    "DocumentFormat",
    "save_document",
    "get_file_extension",
]
