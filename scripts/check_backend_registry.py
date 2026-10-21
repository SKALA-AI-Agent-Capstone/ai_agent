"""Verify or update Spring resource snapshots from the AI-owned registries."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--backend", type=Path, required=True)
p.add_argument("--write", action="store_true", help="Explicitly update the backend snapshots")
args = p.parse_args()
for name in ("reference-sources.json", "external-api-catalog.json"):
    source = ROOT / "resources" / name
    target = args.backend / "services/api/src/main/resources" / name
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if not target.is_file() or source.read_bytes() != target.read_bytes():
        raise SystemExit(f"Stale backend registry snapshot: {name}. Re-run with --write and review both repos.")
print("Backend registry snapshots match the AI sources.")
