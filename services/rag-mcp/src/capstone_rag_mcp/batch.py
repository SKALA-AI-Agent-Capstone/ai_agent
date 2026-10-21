"""Corpus batch: fetch every source in reference-sources.json by its acquisition route and write knowledge.*.

Routes
  legalize  화장품법 법률, 시행령, 시행규칙 from legalize-kr (S05, S06 조문)
  lawapi    국가법령정보 OpenAPI with LAW_OC (S06 별표 5, S07, S08, S09, S10)
  pdf       Official PDFs (S01~S04). Missing files are downloaded from the registry URL first.
  links     Rebuild knowledge.article_links from the loaded articles
  all       Every route above, then links. `--if-empty` skips when the corpus already has rows.

Runs as a batch (manually or as the rag-ingest compose service), never per review request.
"""
import argparse
import asyncio
from datetime import date
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

import httpx
import psycopg

from .embedding import load_embedder
from .ingest import embed_missing_all, from_law, from_lawapi, from_lawapi_statute, from_pdf, read_pdf_pages, write
from .lawapi import LawApi
from .legalize import fetch_versions
from .links import rebuild_links

DOWNLOAD_HOSTS = {"www.mfds.go.kr", "mfds.go.kr", "kcia.or.kr", "www.kcia.or.kr", "www.law.go.kr", "law.go.kr"}
MAX_PDF_BYTES = 32 * 1024 * 1024


def registry_path() -> Path:
    configured = os.getenv("RAG_REGISTRY", "").strip()
    if configured:
        return Path(configured)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "resources/reference-sources.json"
        if candidate.is_file():
            return candidate
    raise SystemExit("reference-sources.json not found. Set RAG_REGISTRY.")


def data_dirs() -> list[Path]:
    root = Path(os.getenv("RAG_DATA_DIR", "data"))
    return [root / "sources", root / "source-audit"]


def sources() -> list[dict]:
    return json.loads(registry_path().read_text(encoding="utf-8"))


def legalize_paths() -> list[str]:
    paths = []
    for source in sources():
        if source["acquisition"]["method"].startswith("legalize_kr"):
            paths += [p for p in source["acquisition"]["paths"] if p not in paths]
    return paths


def find_pdf(source_id: str) -> Path | None:
    return next((d / f"{source_id}.pdf" for d in data_dirs() if (d / f"{source_id}.pdf").is_file()), None)


async def download_pdf(client: httpx.AsyncClient, source: dict) -> Path | None:
    url = source.get("verified_direct_download_url") or source.get("candidate_download_url")
    if not url or urlsplit(url).hostname not in DOWNLOAD_HOSTS:
        return None
    response = await client.get(url, follow_redirects=True)
    if response.status_code != 200 or not response.content.lstrip().startswith(b"%PDF-") or len(response.content) > MAX_PDF_BYTES:
        return None
    target = data_dirs()[0] / f"{source['id']}.pdf"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
    except PermissionError:
        # Linux bind mounts retain host ownership. Keep the non-root container user
        # and use a temporary download cache when the source folder is read-only.
        with tempfile.NamedTemporaryFile(prefix=f"rag-{source['id']}-", suffix=".pdf", delete=False) as cached:
            cached.write(response.content)
            target = Path(cached.name)
    return target


async def collect_legalize(as_of: date) -> list:
    documents = []
    for path in legalize_paths():
        for version in await fetch_versions(path, as_of=as_of):
            documents.append(from_law(version))
    return documents


# legalize-kr path -> (official name, document key used when the Law API supplies the same statute)
STATUTE_FALLBACK = {
    "kr/화장품법/법률.md": ("화장품법", "lawapi:화장품법/법률"),
    "kr/화장품법/시행령.md": ("화장품법 시행령", "lawapi:화장품법/시행령"),
    "kr/화장품법/시행규칙.md": ("화장품법 시행규칙", "lawapi:화장품법/시행규칙"),
}


async def collect_statutes_from_lawapi() -> list:
    """Same statutes as legalize-kr, current version only, from the Law API (needs LAW_OC)."""
    oc = os.getenv("LAW_OC", "").strip()
    if not oc:
        return []
    documents = []
    async with httpx.AsyncClient(timeout=60) as client:
        api = LawApi(oc, client)
        for path in legalize_paths():
            if path in STATUTE_FALLBACK:
                name, key = STATUTE_FALLBACK[path]
                doc = await api.law(name, annex_numbers=set())
                documents.append(from_lawapi_statute(key, doc, f"https://www.law.go.kr/법령/{name.replace(' ', '')}"))
    return documents


async def collect_lawapi() -> tuple[list, list[dict]]:
    oc = os.getenv("LAW_OC", "").strip()
    picked = [s for s in sources() if "lawapi" in s["acquisition"]["method"].split("+")]
    if not oc:
        return [], [{"id": s["id"], "reason": "LAW_OC_NOT_CONFIGURED"} for s in picked]
    documents, skipped = [], []
    async with httpx.AsyncClient(timeout=60) as client:
        api = LawApi(oc, client)
        for source in picked:
            spec = source["acquisition"]["lawapi"]
            annexes = None if spec.get("annexes") is None else set(spec["annexes"])
            try:
                if spec["target"] == "admrul":
                    doc = await api.admrul(spec["name"], annex_numbers=annexes)
                else:
                    doc = await api.law(spec["name"], annex_numbers=annexes, articles=spec.get("articles", True))
                documents.append(from_lawapi(source, doc))
            except Exception as error:
                skipped.append({"id": source["id"], "reason": type(error).__name__})
    return documents, skipped


async def collect_pdf() -> tuple[list, list[dict]]:
    documents, skipped = [], []
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": "BeautyReview-Corpus-Batch/1.0"}) as client:
        for source in sources():
            acquisition = source["acquisition"]
            if "pdf" not in acquisition["method"].split("+"):
                continue
            path = find_pdf(source["id"])
            if path is None:
                try:
                    path = await download_pdf(client, source)
                except httpx.HTTPError:
                    path = None
            if path is None:
                skipped.append({"id": source["id"], "reason": "PDF_NOT_FOUND",
                                "expected": f"data/sources/{source['id']}.pdf", "page": source["official_page"]})
                continue
            if acquisition["collection"] == "mixed_rules_cases":
                ranges = acquisition.get("page_ranges")
                if not ranges:
                    skipped.append({"id": source["id"], "reason": "PAGE_RANGES_REQUIRED_FOR_MIXED_DOCUMENT"})
                    continue
                pages = read_pdf_pages(path)
                for collection, (first, last) in ranges.items():
                    documents.append(from_pdf(source, path, pages=pages, page_range=(first, last),
                                              collection=collection, key_suffix=f":{collection}"))
                continue
            documents.append(from_pdf(source, path))
    return documents, skipped


def corpus_rows(dsn: str) -> int:
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        return conn.execute("SELECT count(*) FROM knowledge.chunks").fetchone()[0]


async def collect(command: str, as_of: date) -> tuple[list, list[dict]]:
    documents, skipped = [], []
    if command in ("legalize", "all"):
        try:
            documents += await collect_legalize(as_of)
        except Exception as error:
            skipped.append({"route": "legalize", "reason": type(error).__name__})
            try:
                fallback = await collect_statutes_from_lawapi()
                documents += fallback
                if fallback:
                    skipped.append({"route": "legalize", "fallback": "lawapi", "documents": len(fallback)})
            except Exception as fallback_error:
                skipped.append({"route": "legalize-fallback", "reason": type(fallback_error).__name__})
    if command in ("lawapi", "all"):
        found, missed = await collect_lawapi()
        documents += found; skipped += missed
    if command in ("pdf", "all"):
        found, missed = await collect_pdf()
        documents += found; skipped += missed
    return documents, skipped


def run(command: str, *, as_of: date, corpus_version: str, dry_run=False, embed=False, if_empty=False) -> int:
    dsn = os.environ["RAG_INGEST_DATABASE_URL"]
    if command == "embed" or (if_empty and embed and corpus_rows(dsn) > 0):
        # Chunks already stored: give them vectors without downloading sources again.
        embedder = load_embedder()
        if embedder is None:
            raise SystemExit("Set RAG_EMBEDDING_MODEL and install the ingest extra to embed.")
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            print(json.dumps({"embed": embed_missing_all(conn, embedder, log=lambda row: print(json.dumps(row), flush=True))}))
        return 0
    if if_empty and corpus_rows(dsn) > 0:
        print(json.dumps({"skipped": "corpus already loaded", "command": command}))
        return 0
    documents, skipped = ([], []) if command == "links" else asyncio.run(collect(command, as_of))
    for doc in documents:
        print(json.dumps({"document": doc.document_key, "version": doc.version_label, "collection": doc.collection,
                          "effective": [str(doc.effective_from), str(doc.effective_until)], "chunks": len(doc.chunks)},
                         ensure_ascii=False))
    for item in skipped:
        print(json.dumps({"skipped": item}, ensure_ascii=False))
    if dry_run:
        print("Dry run: nothing written.")
        return 0
    embedder = load_embedder() if embed else None
    if embed and embedder is None:
        raise SystemExit("Set RAG_EMBEDDING_MODEL and install the ingest extra to embed.")
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        for doc in documents:
            status = write(conn, doc, corpus_version=corpus_version, embedder=embedder)
            print(json.dumps({"document": doc.document_key, "version": doc.version_label, "status": status},
                             ensure_ascii=False))
        if command in ("links", "all"):
            print(json.dumps({"links": rebuild_links(conn)}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["legalize", "lawapi", "pdf", "links", "all", "embed"],
                        help="embed: vectorize stored chunks that have none (no download)")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(),
                        help="Keep statute versions back to the one in force on this date.")
    parser.add_argument("--corpus-version", default=f"{date.today().isoformat()}.1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--if-empty", action="store_true", help="Do nothing when knowledge.chunks already has rows.")
    args = parser.parse_args(argv)
    return run(args.command, as_of=args.as_of, corpus_version=args.corpus_version,
               dry_run=args.dry_run, embed=args.embed, if_empty=args.if_empty)


if __name__ == "__main__":
    raise SystemExit(main())
