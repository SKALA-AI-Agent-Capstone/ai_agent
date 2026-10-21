"""Write public source documents into knowledge.* as rag_ingest.

An existing (document_key, version_label, content_sha256) row is never rewritten.
Only its effective range may move when a newer statute version arrives.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
from pathlib import Path
import re

import psycopg

from .embedding import Embedder
from .lawapi import LawApiDocument
from .legalize import LawVersion, blob_url

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


def from_law(version: LawVersion) -> Document:
    meta = version.meta
    kind = meta.get("법령구분", "")
    issuer = meta.get("소관부처")
    return Document(
        document_key=f"legalize-kr:{version.path.removeprefix('kr/').removesuffix('.md')}",
        version_label=version.version_label,
        title=version.title,
        issuer=", ".join(issuer) if isinstance(issuer, list) else (issuer or "unknown"),
        collection="rules",
        source_type="law" if kind == "법률" else "regulation",
        source_url=blob_url(version),
        content_sha256=version.sha256,
        storage_key=version.storage_key,
        published_on=_date(meta.get("공포일자")),
        effective_from=version.effective_from,
        effective_until=version.effective_until,
        chunks=[{"section": c.section, "context": c.context, "body": c.body, "page_from": None}
                for c in version.chunks],
    )


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
        content_sha256=doc.sha256,
        storage_key=f"lawapi:{doc.target}:{doc.law_id}",
        published_on=doc.published_on,
        effective_from=doc.effective_from,
        chunks=[{"section": c.section, "context": c.context, "body": c.body, "page_from": None} for c in doc.chunks],
    )


def from_lawapi_statute(document_key: str, doc: LawApiDocument, source_url: str) -> Document:
    """화장품법 법률, 시행령, 시행규칙 loaded from the Law API when legalize-kr is unreachable."""
    return Document(
        document_key=document_key, version_label=doc.version_label, title=doc.title, issuer=doc.issuer,
        collection="rules", source_type="law" if doc.kind == "법률" else "regulation", source_url=source_url,
        content_sha256=doc.sha256, storage_key=f"lawapi:{doc.target}:{doc.law_id}", published_on=doc.published_on,
        effective_from=doc.effective_from,
        chunks=[{"section": c.section, "context": c.context, "body": c.body, "page_from": None} for c in doc.chunks],
    )


HEADING = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\d{1,2}\.\s|[가-하]\.\s|\[별표\s*\d*\]|제\d+[장절조])")
TOP_HEADING = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.|\[별표\s*\d*\]|제\d+장)")
FRONT_MATTER = ("점검표", "개정 이력", "개정이력", "이력서", "목 차", "목차", "CONTENTS")
MAX_PDF_CHUNK = 1200


def chunk_pdf_sections(pages: list[str], first_page: int = 1) -> list[dict]:
    """Split guideline PDFs at their own headings (Ⅰ., 1., 가., [별표]) and keep the heading as section.

    Slide decks (short pages) are kept one slide per chunk with the slide title as section.
    Front matter such as revision checklists and tables of contents is skipped.
    """
    texts = [(first_page + i, p or "") for i, p in enumerate(pages)]
    texts = [(n, p) for n, p in texts if p.strip() and not any(k in p[:80] for k in FRONT_MATTER)]
    if not texts:
        return []
    if sum(len(p) for _, p in texts) / len(texts) < 500:
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
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) > 500:
            raise ValueError("PUBLIC_DOCUMENT_PAGE_LIMIT")
        return [page.extract_text() or "" for page in pdf.pages]


def from_pdf(source: dict, path: Path, *, pages: list[str] | None = None,
             page_range: tuple[int, int] | None = None, collection: str | None = None,
             key_suffix: str = "") -> Document:
    acquisition = source["acquisition"]
    raw = path.read_bytes()
    pages = pages if pages is not None else read_pdf_pages(path)
    first, last = page_range or (1, len(pages))
    chunks = [{"section": c["section"], "context": source["title"], "body": c["body"], "page_from": c["page_from"],
               "page_until": c["page_until"]} for c in chunk_pdf_sections(pages[first - 1:last], first)]
    return Document(
        document_key=f"pdf:{source['id']}{key_suffix}",
        version_label=str(source.get("reported_version_date") or "unverified"),
        title=source["title"],
        issuer=ISSUERS.get(source.get("provider", ""), source.get("provider", "unknown")),
        collection=collection or acquisition["collection"],
        source_type="case" if (collection or acquisition["collection"]) == "cases" else acquisition["source_type"],
        source_url=acquisition.get("annex_pdf_url") or source["official_page"],
        content_sha256=hashlib.sha256(raw).hexdigest(),
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
        for index, chunk in enumerate(doc.chunks):
            cur.execute(
                """INSERT INTO knowledge.chunks (document_version_id, chunk_index, section, page_from,
                   page_until, body, context) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (version_id, index, chunk["section"], chunk["page_from"], chunk.get("page_until", chunk["page_from"]),
                 chunk["body"], chunk["context"]))
        if embedder:
            _embed_missing(cur, version_id, embedder)
        recompute_ranges(cur, doc.document_key)
    conn.commit()
    return "inserted"


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
