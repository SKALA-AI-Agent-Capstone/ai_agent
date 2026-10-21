#!/usr/bin/env python3
"""Load the public RAG corpus from the host. Same code as the rag-ingest compose service.

  uv run --all-packages --frozen python scripts/ingest_rag.py all
  uv run --all-packages --frozen python scripts/ingest_rag.py lawapi --dry-run

Routes and options: python -m capstone_rag_mcp.batch --help
"""
import os
import shutil
import subprocess
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
    if not os.getenv("GITHUB_TOKEN", "").strip() and shutil.which("gh"):
        # legalize-kr history uses the GitHub API (60 anonymous calls per hour). Reuse a local gh login if present.
        token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
        if token:
            os.environ["GITHUB_TOKEN"] = token
    raise SystemExit(main(sys.argv[1:]))
