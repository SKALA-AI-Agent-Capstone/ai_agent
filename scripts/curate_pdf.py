#!/usr/bin/env python3
"""Generate a page-by-page RAG Markdown draft from a PDF, then verify image text.

Example:
  uv run --all-packages --frozen python scripts/curate_pdf.py ../backend/data/sources/S04.pdf

The output defaults to ../backend/data/curated/<PDF stem>.md. Existing reviewed files
are protected unless --force is passed. The adjacent .qa.json lists uncertain pages.
"""
import argparse
import json
from pathlib import Path

from capstone_rag_mcp.pdf_curator import write_markdown
from local_config import load_env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    load_env(args.env_file)
    target = args.output or args.pdf.parent.parent / "curated" / f"{args.pdf.stem}.md"
    report = write_markdown(args.pdf, target, force=args.force)
    print(json.dumps({"output": str(target), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
