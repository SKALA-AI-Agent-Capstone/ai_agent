"""Write public source documents into knowledge.* as rag_ingest.

An existing (document_key, version_label, content_sha256) row is never rewritten.
Only its effective range may move when a newer statute version arrives. The one exception is the same
statute version (same label and effective date) stored again with a new chunk layout: it replaces the
old copy, which could not be given an end date before its own start.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re

import psycopg

from .embedding import Embedder
from .lawapi import LawApiDocument
from .links import LEGACY_KEYS
from capstone_shared.reading import READER_VERSION

from .pdf_vision import AUTO_OCR, AUTO_OCR_UNCHECKED, AUTO_TRANSCRIPTION

ISSUERS = {"mfds": "식품의약품안전처", "kcia": "대한화장품협회", "law": "법제처 국가법령정보센터"}


@dataclass
class Document:
    document_key: str
    version_label: str
    title: str
    issuer: str
    collection: str
    source_type: str
    source_url: str
    content_sha256: str
    storage_key: str
    published_on: date | None = None
    effective_from: date | None = None
    effective_until: date | None = None
    chunks: list[dict] = field(default_factory=list)


def law_chunks(doc: LawApiDocument) -> list[dict]:
    return [{"section": c.section, "context": c.context, "body": c.body, "page_from": None,
             "children": [{"section": k.section, "context": k.context, "body": k.body, "page_from": None}
                          for k in c.children]} for c in doc.chunks]


def law_sha256(doc: LawApiDocument) -> str:
    """Source text plus chunk layout: re-chunking the same text (parent-child split) must give a new version."""
    layout = json.dumps([[c.section, c.body, [k.section for k in c.children]] for c in doc.chunks],
                        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(doc.raw + layout).hexdigest()


def from_lawapi(source: dict, doc: LawApiDocument) -> Document:
    acquisition = source["acquisition"]
    kind = acquisition["source_type"] if acquisition["method"] == "lawapi" else "regulation"
    return Document(
        document_key=f"lawapi:{source['id']}",
        version_label=doc.version_label,
        title=doc.title,
        issuer=doc.issuer,
        collection="rules",
        source_type=kind,
        source_url=source["official_page"],
        content_sha256=law_sha256(doc),
        storage_key=f"lawapi:{doc.target}:{doc.law_id}",
        published_on=doc.published_on,
        effective_from=doc.effective_from,
        chunks=law_chunks(doc),
    )


def from_lawapi_statute(document_key: str, doc: LawApiDocument, source_url: str) -> Document:
    """화장품법 법률, 시행령, 시행규칙 (S05), current version from the Law API."""
    return Document(
        document_key=document_key, version_label=doc.version_label, title=doc.title, issuer=doc.issuer,
        collection="rules", source_type="law" if doc.kind == "법률" else "regulation", source_url=source_url,
        content_sha256=law_sha256(doc), storage_key=f"lawapi:{doc.target}:{doc.law_id}", published_on=doc.published_on,
        effective_from=doc.effective_from,
        chunks=law_chunks(doc),
    )


HEADING = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\d{1,2}\.\s|[가-하]\.\s|\[별표\s*\d*\]|제\d+[장절조])")
TOP_HEADING = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\[별표\s*\d*\]|제\d+장)")
FRONT_MATTER = ("점검표", "개정 이력", "개정이력", "이력서", "목 차", "목차", "CONTENTS")
MAX_PDF_CHUNK = 1200


def chunk_pdf_sections(pages: list[str], first_page: int = 1, *, slide_mode: bool = False) -> list[dict]:
    """Split guideline PDFs at their own headings (Ⅰ., 1., 가., [별표]) and keep the heading as section.

    Slide decks (short pages) are kept one slide per chunk with the slide title as section.
    Front matter such as revision checklists and tables of contents is skipped.
    """
    texts = [(first_page + i, p or "") for i, p in enumerate(pages)]
    texts = [(n, p) for n, p in texts if p.strip() and not any(k in p[:80] for k in FRONT_MATTER)]
    if not texts:
        return []
    if slide_mode or sum(len(p) for _, p in texts) / len(texts) < 500:
        return [{"section": p.strip().split("\n")[0][:60], "body": p.strip(), "page_from": n, "page_until": n}
                for n, p in texts]
    chunks: list[dict] = []
    top = heading = None
    buffer: list[str] = []
    start = None

    def flush(end_page):
        body = "\n".join(buffer).strip()
        if len(body) > 20:
            section = " > ".join(h for h in (top, heading) if h) or None
            chunks.append({"section": section[:90] if section else None, "body": body,
                           "page_from": start, "page_until": end_page})

    page = first_page
    for page, text in texts:
        for line in text.split("\n"):
            line = line.strip()
            if not line or re.fullmatch(r"-?\s*\d+\s*-?", line):
                continue  # 쪽 번호
            starts_section = len(line) < 70 and (TOP_HEADING.match(line) or re.match(r"^\d{1,2}\.\s", line))
            if starts_section:
                flush(page)
                buffer, start = [], page
                if TOP_HEADING.match(line):
                    top, heading = line[:50], None
                else:
                    heading = line[:50]
            if start is None:
                start = page
            buffer.append(line)
            if sum(len(b) for b in buffer) > MAX_PDF_CHUNK:
                flush(page)
                buffer, start = [], page
    flush(page)
    return chunks


def read_pdf_pages(path: Path) -> list[str]:
    """Text layer of every page, by the shared page reader (no OCR; picture pages are handled in batch)."""
    from .pdf_vision import read_public_pdf

    return [page.text for page in read_public_pdf(path).pages]


def from_pdf(source: dict, path: Path, *, pages: list[str] | None = None,
             page_range: tuple[int, int] | None = None, collection: str | None = None,
             key_suffix: str = "", extra_chunks: list[dict] | None = None, curated: bool = False) -> Document:
    acquisition = source["acquisition"]
    raw = path.read_bytes()
    pages = pages if pages is not None else read_pdf_pages(path)
    first, last = page_range or (1, len(pages))
    selected_pages = pages[first - 1:last]
    chunks = [{"section": c["section"], "context": source["title"], "body": c["body"], "page_from": c["page_from"],
               "page_until": c["page_until"]} for c in chunk_pdf_sections(selected_pages, first,
                                                            slide_mode=acquisition["collection"] == "mixed_rules_cases")]
    if source["id"] == "S01" and extra_chunks:
        # Raw text extraction runs table columns together; rows are the precise search units.
        table_end = max(chunk["page_until"] for chunk in extra_chunks)
        chunks = [chunk for chunk in chunks if not ((chunk["section"] or "").startswith("[별표 ")
                  and chunk["page_until"] <= table_end)]
    chunks.extend({**chunk, "context": source["title"]} for chunk in (extra_chunks or []))
    # Reviewed markdown can change while the PDF stays the same; its text must give a new version.
    derived = "\n".join(selected_pages).encode("utf-8") if curated or any(
        AUTO_TRANSCRIPTION in p for p in selected_pages) else b""
    # OCR text depends on the reader: a new reader version gives the OCR pages a new document version.
    if any(AUTO_OCR in p or AUTO_OCR_UNCHECKED in p for p in selected_pages):
        derived += "\n".join(selected_pages).encode("utf-8") + READER_VERSION.encode("ascii")
    if extra_chunks:
        derived += json.dumps(extra_chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return Document(
        document_key=f"pdf:{source['id']}{key_suffix}",
        version_label=str(source.get("reported_version_date") or "unverified"),
        title=source["title"],
        issuer=ISSUERS.get(source.get("provider", ""), source.get("provider", "unknown")),
        collection=collection or acquisition["collection"],
        source_type="case" if (collection or acquisition["collection"]) == "cases" else acquisition["source_type"],
        source_url=acquisition.get("annex_pdf_url") or source["official_page"],
        content_sha256=hashlib.sha256(raw + derived).hexdigest(),
        storage_key=f"file:{path.name}",
        chunks=chunks,
    )


def _date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def write(conn: psycopg.Connection, doc: Document, *, corpus_version: str,
          embedder: Embedder | None = None) -> str:
    """Return inserted, exists or range_updated. Commits on success."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT document_version_id, effective_from, effective_until FROM knowledge.document_versions
               WHERE document_key = %s AND version_label = %s AND content_sha256 = %s""",
            (doc.document_key, doc.version_label, doc.content_sha256))
        existing = cur.fetchone()
        if existing:
            status = "exists"
            if existing[1] != doc.effective_from:
                cur.execute("UPDATE knowledge.document_versions SET effective_from = %s WHERE document_version_id = %s",
                            (doc.effective_from, existing[0]))
                status = "range_updated"
            recompute_ranges(cur, doc.document_key)
            if embedder:
                _embed_missing(cur, existing[0], embedder)
            conn.commit()
            return status
        replace_rechunked(cur, doc)
        cur.execute(
            """INSERT INTO knowledge.document_versions
               (document_key, version_label, title, issuer, collection_key, source_type, source_url,
                source_verified, published_on, effective_from, effective_until, content_sha256,
                storage_key, corpus_version)
               VALUES (%s,%s,%s,%s,%s,%s,%s,false,%s,%s,%s,%s,%s,%s) RETURNING document_version_id""",
            (doc.document_key, doc.version_label, doc.title, doc.issuer, doc.collection, doc.source_type,
             doc.source_url, doc.published_on, doc.effective_from, doc.effective_until,
             doc.content_sha256, doc.storage_key, corpus_version))
        version_id = cur.fetchone()[0]
        index = 0
        for chunk in doc.chunks:
            parent_id = None
            for row in [chunk, *chunk.get("children", [])]:
                cur.execute(
                    """INSERT INTO knowledge.chunks (document_version_id, chunk_index, parent_chunk_id, section, page_from,
                       page_until, body, context) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING chunk_id""",
                    (version_id, index, parent_id, row["section"], row["page_from"],
                     row.get("page_until", row["page_from"]), row["body"], row["context"]))
                chunk_id = cur.fetchone()[0]
                parent_id = parent_id or chunk_id  # children of this chunk point at it
                index += 1
        if embedder:
            _embed_missing(cur, version_id, embedder)
        recompute_ranges(cur, doc.document_key)
    conn.commit()
    return "inserted"


def retire_replaced(conn: psycopg.Connection, doc: Document) -> int:
    """End the legacy legalize-kr versions of a statute the day before its Law API version takes effect.

    Older versions keep answering for past review dates; none overlaps the Law API version, so a
    reload does not leave the same statute in search twice. Returns the number of versions changed.
    """
    legacy = LEGACY_KEYS.get(doc.document_key)
    if legacy is None:
        return 0
    until = (doc.effective_from or date.today()) - timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute("""UPDATE knowledge.document_versions SET effective_until = %s
                       WHERE document_key = %s AND (effective_until IS NULL OR effective_until > %s)""",
                    (until, legacy, until))
        changed = cur.rowcount
    conn.commit()
    return changed


def replace_rechunked(cur, doc: Document) -> int:
    """Delete stored copies of this exact statute version (same label and effective date, other content hash).

    They hold the same legal text cut into different chunks; keeping both would put the article in search twice.
    PDF guides have no effective date and keep their older versions as before.
    """
    if doc.effective_from is None:
        return 0
    cur.execute("""SELECT document_version_id FROM knowledge.document_versions
                   WHERE document_key = %s AND version_label = %s AND effective_from = %s AND content_sha256 <> %s""",
                (doc.document_key, doc.version_label, doc.effective_from, doc.content_sha256))
    stale = [row[0] for row in cur.fetchall()]
    if stale:
        cur.execute("DELETE FROM knowledge.chunks WHERE document_version_id = ANY(%s) AND parent_chunk_id IS NOT NULL", (stale,))
        cur.execute("DELETE FROM knowledge.chunks WHERE document_version_id = ANY(%s)", (stale,))
        cur.execute("DELETE FROM knowledge.document_versions WHERE document_version_id = ANY(%s)", (stale,))
    return len(stale)


def recompute_ranges(cur, document_key: str) -> None:
    """Every version of a document ends the day before the next one (by effective date) starts.

    A batch may fetch only the newest version, so ranges are rebuilt from all stored versions.
    """
    cur.execute("""SELECT document_version_id, effective_from FROM knowledge.document_versions
                   WHERE document_key = %s AND effective_from IS NOT NULL
                   ORDER BY effective_from, published_on NULLS FIRST, retrieved_at""", (document_key,))
    rows = cur.fetchall()
    for (version_id, _), following in zip(rows, rows[1:] + [None]):
        until = following[1] - timedelta(days=1) if following else None
        cur.execute("UPDATE knowledge.document_versions SET effective_until = %s WHERE document_version_id = %s",
                    (until, version_id))


def embed_missing_all(conn: psycopg.Connection, embedder: Embedder, log=print) -> dict:
    """Embed every stored chunk that has no vector yet, without fetching sources again.
    One commit per document version, so an interrupted run keeps what it finished and resumes from there.
    Vectors made by another model are cleared first: documents and queries must share one model."""
    with conn.cursor() as cur:
        cur.execute("""UPDATE knowledge.chunks SET embedding = NULL, embedding_model = NULL, embedding_revision = NULL
                       WHERE embedding IS NOT NULL AND embedding_model IS DISTINCT FROM %s""", (embedder.model_name,))
        if cur.rowcount:
            log({"cleared_other_model": cur.rowcount})
        conn.commit()
    with conn.cursor() as cur:
        cur.execute("""SELECT document_version_id, count(*) FROM knowledge.chunks
                       WHERE embedding IS NULL GROUP BY 1 ORDER BY 1""")
        pending = cur.fetchall()
    done = 0
    for version_id, count in pending:
        with conn.cursor() as cur:
            _embed_missing(cur, version_id, embedder)
        conn.commit()
        done += count
        log({"embedded": str(version_id), "chunks": count, "progress": f"{done}/{sum(c for _, c in pending)}"})
    return {"versions": len(pending), "chunks": done, "model": embedder.model_name, "revision": embedder.revision}


def _embed_missing(cur, version_id, embedder: Embedder, batch: int = 16) -> None:
    cur.execute("""SELECT chunk_id, coalesce(section, '') || ' ' || body FROM knowledge.chunks
                   WHERE document_version_id = %s AND embedding IS NULL ORDER BY chunk_index""", (version_id,))
    rows = cur.fetchall()
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        for (chunk_id, _), vector in zip(part, embedder.embed([text for _, text in part])):
            cur.execute("""UPDATE knowledge.chunks SET embedding = %s::vector, embedding_model = %s,
                           embedding_revision = %s WHERE chunk_id = %s""",
                        (str(vector), embedder.model_name, embedder.revision, chunk_id))
