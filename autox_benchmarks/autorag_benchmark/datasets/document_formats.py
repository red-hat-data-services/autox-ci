"""Document format handlers for dataset generation.

Supports saving documents in native formats: txt, md, pptx, and images (png, jpg).
Note: PDF/PPTX support is only for native files, not text conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DocumentFormat = Literal["txt", "md", "pdf", "pptx", "png", "jpg"]


def save_document(
    content: str,
    output_path: Path,
    format: DocumentFormat = "txt",
    metadata: dict | None = None,
) -> Path:
    """Save document text content in the specified format.

    Args:
        content: Document text content
        output_path: Output path (extension will be replaced based on format)
        format: Output format - "txt" or "md"
        metadata: Optional metadata dict (used for markdown frontmatter)

    Returns:
        Path to the created file

    Note:
        This function is for text content only. For binary formats (PDF, PPTX, images),
        use save_binary_document() instead.
    """
    if format == "txt":
        return _save_txt(content, output_path)
    elif format == "md":
        return _save_markdown(content, output_path, metadata)
    elif format in ("pdf", "pptx", "png", "jpg"):
        raise ValueError(
            f"Format '{format}' requires binary data. "
            f"Use save_binary_document() for {format} files, or use format='txt'/'md' for text content."
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


def save_binary_document(
    binary_data: bytes,
    output_path: Path,
    format: DocumentFormat,
) -> Path:
    """Save binary document data (PDF, PPTX, images).

    Args:
        binary_data: Binary file content
        output_path: Output path (extension will be replaced based on format)
        format: Output format - "pdf", "pptx", "png", or "jpg"

    Returns:
        Path to the created file

    Raises:
        ValueError: If format is not a binary format
    """
    if format not in ("pdf", "pptx", "png", "jpg"):
        raise ValueError(
            f"Format '{format}' is not a binary format. "
            f"Use save_document() for text formats like 'txt' or 'md'."
        )

    output_file = output_path.with_suffix(f".{format}")
    output_file.write_bytes(binary_data)
    return output_file


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
    "save_binary_document",
    "get_file_extension",
]
