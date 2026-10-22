"""Corpus batch: fetch every source in reference-sources.json by its acquisition route and write knowledge.*.

Routes
  lawapi    국가법령정보 OpenAPI with LAW_OC: 화장품법, 시행령, 시행규칙 현행본 (S05), S06 별표 5, S07, S08, S09, S10
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
from .ingest import embed_missing_all, from_lawapi, from_lawapi_statute, from_pdf, retire_replaced, write
from .lawapi import LawApi
from .curated import CuratedMismatch, curated_path, parse_curated_text, read_curated_pages
from .pdf_vision import read_pages_for_index
from .pdf_tables import s01_annex_rows
from .links import rebuild_links

# Model reading of image pages misses or misreads too much (S04: 58% of reviewed key phrases).
# A PDF whose indexed pages are at least this share images is loaded only from reviewed MD.
IMAGE_HEAVY_SHARE = 0.3


class CuratedMdRequired(Exception):
    """An image-heavy PDF has no reviewed MD that matches it, so it is left out of the index."""

    def __init__(self, image_pages: list[int], total: int):
        super().__init__("PDF_NEEDS_CURATED_MD")
        self.image_pages, self.total = image_pages, total


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
                if spec.get("statutes"):
                    # One source, several statutes (화장품법, 시행령, 시행규칙): each is its own document.
                    for statute in spec["statutes"]:
                        doc = await api.law(statute["name"], annex_numbers=set())
                        documents.append(from_lawapi_statute(
                            statute["document_key"], doc,
                            f"https://www.law.go.kr/법령/{statute['name'].replace(' ', '')}"))
                    continue
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
            ranges = acquisition.get("page_ranges") if acquisition["collection"] == "mixed_rules_cases" else None
            if acquisition["collection"] == "mixed_rules_cases" and not ranges:
                skipped.append({"id": source["id"], "reason": "PAGE_RANGES_REQUIRED_FOR_MIXED_DOCUMENT"})
                continue
            page_numbers = ({page for first, last in ranges.values() for page in range(first, last + 1)}
                            if ranges else None)
            page_warnings = []
            try:
                pages, curated = read_source_pages(path, page_numbers, page_warnings.append)
            except CuratedMdRequired as error:
                page_warnings.append({"reason": str(error), "image_pages": error.image_pages,
                                      "pages": error.total, "expected": f"data/curated/{source['id']}.md"})
                continue
            finally:
                skipped.extend({"id": source["id"], **warning} for warning in page_warnings)
            if acquisition["collection"] == "mixed_rules_cases":
                for collection, (first, last) in ranges.items():
                    documents.append(from_pdf(source, path, pages=pages, page_range=(first, last),
                                              collection=collection, key_suffix=f":{collection}", curated=curated))
                continue
            annex_rows = s01_annex_rows(path) if source["id"] == "S01" else None
            documents.append(from_pdf(source, path, pages=pages, extra_chunks=annex_rows, curated=curated))
    return documents, skipped


def read_source_pages(path: Path, page_numbers: set[int] | None, on_warning) -> tuple[list[str], bool]:
    """Prefer reviewed MD, then optional in-memory auto check, then basic PDF vision.

    An image-heavy PDF without a matching MD raises CuratedMdRequired instead of being read by a model.
    """
    md_path = curated_path(path)
    if md_path.is_file():
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
        try:
            trust = {}
            pages = read_curated_pages(md_path, path, page_count, on_warning=on_warning, trust=trust)
            pages = _in_range(pages, page_numbers)
            on_warning(_trust_summary("curated_md", trust, page_numbers, md_path.name))
            return pages, True
        except CuratedMismatch as error:
            on_warning({"file": md_path.name, "reason": str(error), "fallback": "pdf"})
    from .pdf_curator import image_page_numbers

    image_pages, total = image_page_numbers(path, page_numbers)
    if total and len(image_pages) / total >= IMAGE_HEAVY_SHARE:
        raise CuratedMdRequired(image_pages, total)
    if os.getenv("RAG_PDF_AUTO_VERIFY", "false").lower() == "true":
        import pdfplumber
        from .pdf_curator import build_markdown

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
        markdown, report = build_markdown(path, page_numbers=page_numbers)
        trust = {}
        pages = _in_range(parse_curated_text(markdown, path, page_count, on_warning=on_warning, trust=trust),
                          page_numbers)
        for issue in report["issues"]:
            if issue.get("corrections") and not issue.get("uncertain"):
                on_warning({"page": issue["page"], "reason": "AUTO_PDF_CHECK_NOTES",
                            "count": len(issue["corrections"])})
            elif issue.get("reasons"):
                on_warning({"page": issue["page"], "reason": "AUTO_PDF_CHECK_FAILED", "checks": issue["reasons"]})
        for page in report.get("page_reports", []):
            if page["trust"] == "native_text" and page.get("reasons"):
                on_warning({"page": page["page"], "reason": "AUTO_PDF_TEXT_PAGE_NOTE", "checks": page["reasons"]})
        on_warning(_trust_summary("auto_verify", trust, page_numbers, write_qa_report(path, report)))
        return pages, True
    return read_pages_for_index(path, page_numbers=page_numbers, on_warning=on_warning), False


def _in_range(pages: list[str], page_numbers: set[int] | None) -> list[str]:
    if page_numbers is None:
        return pages
    return [page if number in page_numbers else "" for number, page in enumerate(pages, 1)]


def _trust_summary(mode: str, trust: dict[int, str], page_numbers: set[int] | None, report: str | None) -> dict:
    """One line per PDF so an operator sees which pages are human-, model- or not verified."""
    pages: dict[str, list[int]] = {}
    for number, level in sorted(trust.items()):
        if page_numbers is None or number in page_numbers:
            pages.setdefault(level, []).append(number)
    return {"reason": "PDF_PAGE_TRUST", "mode": mode, "pages": pages, "report": report}


def write_qa_report(path: Path, report: dict) -> str | None:
    """Keep the auto check's corrections and excluded pages; never touches data/curated."""
    folder = Path(os.getenv("RAG_PDF_QA_DIR", "").strip() or Path(os.getenv("RAG_DATA_DIR", "data")) / "pdf-qa")
    target = folder / f"{path.stem}.{report['sha256'][:12]}.auto.json"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return None  # read-only data mount: the batch output still lists every excluded page
    return str(target)


def corpus_rows(dsn: str) -> int:
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        return conn.execute("SELECT count(*) FROM knowledge.chunks").fetchone()[0]


async def collect(command: str) -> tuple[list, list[dict]]:
    documents, skipped = [], []
    if command in ("lawapi", "all"):
        found, missed = await collect_lawapi()
        documents += found; skipped += missed
    if command in ("pdf", "all"):
        found, missed = await collect_pdf()
        documents += found; skipped += missed
    return documents, skipped


def run(command: str, *, corpus_version: str, dry_run=False, embed=False, if_empty=False) -> int:
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
    documents, skipped = ([], []) if command == "links" else asyncio.run(collect(command))
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
            retired = retire_replaced(conn, doc)
            if retired:
                print(json.dumps({"document": doc.document_key, "retired_legacy_versions": retired}, ensure_ascii=False))
        if command in ("links", "all"):
            print(json.dumps({"links": rebuild_links(conn)}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["lawapi", "pdf", "links", "all", "embed"],
                        help="embed: vectorize stored chunks that have none (no download)")
    parser.add_argument("--corpus-version", default=f"{date.today().isoformat()}.1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--if-empty", action="store_true", help="Do nothing when knowledge.chunks already has rows.")
    args = parser.parse_args(argv)
    return run(args.command, corpus_version=args.corpus_version,
               dry_run=args.dry_run, embed=args.embed, if_empty=args.if_empty)


if __name__ == "__main__":
    raise SystemExit(main())
