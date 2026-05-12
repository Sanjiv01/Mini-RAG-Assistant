"""Load PDF/TXT/MD files and split them into heading-aware overlapping chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .config import settings


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit. `text` is what gets embedded *and* shown."""

    text: str
    source: str          # filename or label
    chunk_id: str        # stable hash, used for dedupe
    section: str = ""    # nearest preceding heading, if any
    context: str = ""    # optional Contextual-Retrieval preamble (prepended at embed time)

    @property
    def embed_text(self) -> str:
        """Text used at embedding/rerank time. Adds source+section as cheap
        structural context (a deterministic stand-in for LLM-based Contextual
        Retrieval) and prepends the LLM-generated preamble if present."""
        prefix_bits = []
        if self.section:
            prefix_bits.append(f"[{self.source} — {self.section}]")
        elif self.source:
            prefix_bits.append(f"[{self.source}]")
        if self.context:
            prefix_bits.append(self.context)
        prefix = "\n".join(prefix_bits)
        return f"{prefix}\n\n{self.text}".strip() if prefix else self.text


# --- file loading -----------------------------------------------------------

def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # Some malformed PDFs raise on a single page; skip it rather than the file.
            continue
    return "\n\n".join(pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


_LOADERS = {
    ".pdf": _read_pdf,
    ".txt": _read_text,
    ".md": _read_text,
    ".markdown": _read_text,
}


def load_document(path: Path) -> str:
    """Return raw text from a supported file, or '' on failure."""
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    try:
        return loader(path)
    except Exception as exc:  # surface a soft failure; caller decides
        raise RuntimeError(f"Failed to parse {path.name}: {exc}") from exc


# --- chunking ---------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 _\-]{3,}$)", re.MULTILINE)
_SPLIT_PRIORITY = ["\n\n", "\n", ". ", " "]


def _normalize(text: str) -> str:
    # Collapse Windows newlines, trim trailing whitespace per line, drop NULs.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _split_with_positions(text: str, max_len: int, start: int = 0) -> list[tuple[int, str]]:
    """Greedy split returning (start_offset_in_text, chunk_text) pairs."""
    if len(text) <= max_len:
        return [(start, text)]
    for sep in _SPLIT_PRIORITY:
        if sep in text:
            parts = text.split(sep)
            out: list[tuple[int, str]] = []
            buf = ""
            buf_start = start
            cursor = start
            for part in parts:
                candidate = (buf + sep + part) if buf else part
                if len(candidate) <= max_len:
                    if not buf:
                        buf_start = cursor
                    buf = candidate
                else:
                    if buf:
                        out.append((buf_start, buf))
                    if len(part) > max_len:
                        out.extend(_split_with_positions(part, max_len, start=cursor))
                        buf = ""
                    else:
                        buf = part
                        buf_start = cursor
                cursor += len(part) + len(sep)
            if buf:
                out.append((buf_start, buf))
            return out
    # Last resort: hard cut.
    return [(start + i, text[i : i + max_len]) for i in range(0, len(text), max_len)]


def _nearest_heading(text_so_far: str) -> str:
    matches = list(_HEADING_RE.finditer(text_so_far))
    if not matches:
        return ""
    return matches[-1].group(0).lstrip("#").strip()


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping chunks, attaching nearest heading as `section`."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    text = _normalize(text)
    if not text:
        return []

    raw = _split_with_positions(text, chunk_size)

    chunks: list[Chunk] = []
    for i, (start, piece) in enumerate(raw):
        # Apply overlap by prepending the tail of the previous raw chunk.
        if i > 0 and overlap > 0:
            prev_tail = raw[i - 1][1][-overlap:]
            piece_text = prev_tail + piece
        else:
            piece_text = piece
        section = _nearest_heading(text[: start])
        if not section:
            # Chunk starts before any heading — look inside the chunk for one.
            m = _HEADING_RE.search(text[start : start + len(piece)])
            if m:
                section = m.group(0).lstrip("#").strip()
        chunk_id = hashlib.sha1(f"{source}::{piece_text}".encode("utf-8")).hexdigest()[:16]
        chunks.append(
            Chunk(text=piece_text.strip(), source=source, chunk_id=chunk_id, section=section)
        )
    return chunks


def load_and_chunk(paths: Iterable[Path]) -> list[Chunk]:
    """Convenience: parse each file and return the concatenated chunk list."""
    chunks: list[Chunk] = []
    for p in paths:
        text = load_document(p)
        chunks.extend(chunk_text(text, source=p.name))
    return chunks
