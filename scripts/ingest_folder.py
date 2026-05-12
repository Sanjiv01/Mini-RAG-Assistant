"""CLI: ingest every supported file in a folder into the persistent cache.

Usage:
    python scripts/ingest_folder.py path/to/folder [--no-contextual]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.pipeline import build_rag  # noqa: E402

SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--no-contextual", action="store_true",
                    help="Skip Contextual-Retrieval preamble generation.")
    args = ap.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Not a directory: {args.folder}")

    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No supported files in {args.folder}")

    print(f"Found {len(files)} files. Loading LLM and building RAG...")
    rag = build_rag()
    added = rag.ingest_paths(
        files, contextual=not args.no_contextual,
        progress=lambda p: print(f"  contextualising... {p:.0%}", end="\r"),
    )
    rag.store.save()
    print(f"\nIndexed {added} chunks. Saved to cache.")


if __name__ == "__main__":
    main()
