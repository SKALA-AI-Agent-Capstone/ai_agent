"""Load explicitly selected local settings without echoing their values."""
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]


def load_env(path=None):
    if path and Path(path).resolve() != (ROOT / '.env').resolve():
        load_env()
    explicit = path is not None
    path = Path(path) if path else ROOT / '.env'
    if not explicit and not path.is_file():
        path = ROOT.parent / 'backend' / '.env'
    if not path.is_file():
        raise SystemExit('Local env file not found. Run scripts/bootstrap.py first.')
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.removeprefix('export ').split('=', 1)
        tokens = shlex.split(value, comments=True, posix=True)
        value = tokens[0] if tokens else ''
        if explicit:
            os.environ[key.strip()] = value
        else:
            os.environ.setdefault(key.strip(), value)
    # Respect explicit process and file settings before choosing the sibling data directory.
    os.environ.setdefault('RAG_DATA_DIR', str(path.parent / 'data'))
