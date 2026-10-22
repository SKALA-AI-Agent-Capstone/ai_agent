#!/usr/bin/env python3
"""Load the public RAG corpus from the host. Same code as the rag-ingest compose service.

  uv run --all-packages --frozen python scripts/ingest_rag.py all
  uv run --all-packages --frozen python scripts/ingest_rag.py lawapi --dry-run

Routes and options: python -m capstone_rag_mcp.batch --help
"""
import os
import sys

from capstone_rag_mcp.batch import main
from local_config import ROOT, load_env

if __name__ == "__main__":
    env_file = None
    if "--env-file" in sys.argv:
        index = sys.argv.index("--env-file")
        env_file = sys.argv[index + 1]
        del sys.argv[index:index + 2]
    load_env(env_file)
    os.environ.setdefault("RAG_DATA_DIR", str(ROOT / "data"))
    raise SystemExit(main(sys.argv[1:]))
