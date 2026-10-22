"""Search over knowledge.chunks.

Keyword ranking always runs, so the corpus is searchable before an embedding
model is chosen. When chunks carry embeddings from the configured model, the
two rankings are merged with reciprocal rank fusion.
"""
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import date
import math
import re

import psycopg
from psycopg.rows import dict_row

from .embedding import Embedder
from .parents import collapse

MIN_COVERAGE = 0.12
RRF_K = 60
# Hybrid order. Three ways were measured on the same claims (docs/12-hybrid-retrieval.md, compare.py):
# rrf      rank fusion of keyword and vector rankings (default). The agent attached relevant search evidence to
#          13 of 14 claims, against 10 with keywords alone. Statutes it ranks lower are attached by the rules anyway.
# confirm  keyword order, each hit carrying its cosine; vector-only hits after. Same top 3 as keywords alone.
# cascade  strong keyword hits, then vector-only hits, then weak keyword hits. Lowest relevance; kept for the table.
STRONG_KEYWORD = 0.45
STRONG_VECTOR = 0.45
VECTOR_WEIGHT = 1.0

CANDIDATES_SQL = """
WITH docs AS (
    SELECT DISTINCT ON (document_key) *
    FROM knowledge.document_versions
    WHERE collection_key = %(collection)s
      AND (effective_from IS NULL OR effective_from <= %(review_date)s)
      AND (effective_until IS NULL OR effective_until >= %(review_date)s)
    ORDER BY document_key, effective_from DESC NULLS LAST, retrieved_at DESC
)
SELECT c.chunk_id::text, c.section, c.context, c.body, c.page_from,
       c.parent_chunk_id::text AS parent_chunk_id, p.section AS parent_section, p.body AS parent_body,
       d.document_key, d.title, d.version_label, d.source_url, d.storage_key, d.source_type,
       d.effective_from, d.effective_until, d.corpus_version
FROM knowledge.chunks c JOIN docs d USING (document_version_id)
LEFT JOIN knowledge.chunks p ON p.chunk_id = c.parent_chunk_id
-- Leaves only: a split article is searched through its 항/호 children, never as a whole.
WHERE NOT EXISTS (SELECT 1 FROM knowledge.chunks k WHERE k.parent_chunk_id = c.chunk_id)
"""

VECTOR_SQL = """
SELECT chunk_id::text, 1 - (embedding <=> %(vector)s::vector) AS cosine
FROM knowledge.chunks
WHERE chunk_id = ANY(%(ids)s::uuid[]) AND embedding_model = %(model)s
ORDER BY embedding <=> %(vector)s::vector
LIMIT %(limit)s
"""


@dataclass
class Ranked:
    row: dict
    similarity: float
    keyword: float | None = None  # idf-weighted share of query bigrams found
    vector: float | None = None   # cosine similarity with the query embedding


def bigrams(text: str) -> list[str]:
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()
    return [compact[i:i + 2] for i in range(len(compact) - 1)]


def keyword_rank(query: str, rows: list[dict], *, k1: float = 1.2, b: float = 0.75) -> list[Ranked]:
    """BM25 over character bigrams for ordering; similarity is the idf-weighted share of query bigrams found."""
    query_terms = Counter(bigrams(query))
    if not query_terms or not rows:
        return []
    docs = [Counter(bigrams(f"{row['section'] or ''} {row['body']}")) for row in rows]
    average = sum(sum(d.values()) for d in docs) / len(docs) or 1
    idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5))
           for t in query_terms for n in [sum(1 for d in docs if t in d)]}
    total_weight = sum(idf.values()) or 1
    ranked = []
    for row, terms in zip(rows, docs):
        length = sum(terms.values()) or 1
        score = sum(idf[t] * terms[t] * (k1 + 1) / (terms[t] + k1 * (1 - b + b * length / average))
                    for t in query_terms if t in terms)
        coverage = sum(idf[t] for t in query_terms if t in terms) / total_weight
        if coverage >= MIN_COVERAGE:
            ranked.append((score, Ranked(row, round(coverage, 4), keyword=round(coverage, 4))))
    return [item for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)]


def cascade(keyword: list[Ranked], vector: list[Ranked]) -> list[Ranked]:
    """Keyword ranking leads; vectors confirm it (each keyword hit carries its cosine) and add what it missed."""
    cosine = {v.row["chunk_id"]: v.vector for v in vector}
    scored = [Ranked(k.row, max(k.keyword or 0, cosine.get(k.row["chunk_id"]) or 0), keyword=k.keyword,
                     vector=cosine.get(k.row["chunk_id"])) for k in keyword]
    found = {k.row["chunk_id"] for k in keyword}
    strong = [r for r in scored if (r.keyword or 0) >= STRONG_KEYWORD]
    added = [v for v in vector if v.row["chunk_id"] not in found and (v.vector or 0) >= STRONG_VECTOR]
    weak = [r for r in scored if (r.keyword or 0) < STRONG_KEYWORD]
    return strong + added + weak


def confirm(keyword: list[Ranked], vector: list[Ranked]) -> list[Ranked]:
    """Keyword order stays; every keyword hit carries its cosine so a semantically close match with weak word
    overlap can still clear the agent's vector threshold. Strong vector-only hits come after the keyword hits."""
    cosine = {v.row["chunk_id"]: v.vector for v in vector}
    found = {k.row["chunk_id"] for k in keyword}
    scored = [Ranked(k.row, max(k.keyword or 0, cosine.get(k.row["chunk_id"]) or 0), keyword=k.keyword,
                     vector=cosine.get(k.row["chunk_id"])) for k in keyword]
    return scored + [v for v in vector if v.row["chunk_id"] not in found and (v.vector or 0) >= STRONG_VECTOR]


def fuse(keyword: list[Ranked], vector: list[Ranked], vector_weight: float = VECTOR_WEIGHT) -> list[Ranked]:
    """Reciprocal rank fusion. Each result keeps both signals so callers can apply a threshold per scale."""
    scores: dict[str, float] = {}
    merged: dict[str, Ranked] = {}
    for ranking, weight in ((keyword, 1.0), (vector, vector_weight)):
        for position, item in enumerate(ranking):
            key = item.row["chunk_id"]
            scores[key] = scores.get(key, 0) + weight / (RRF_K + position + 1)
            seen = merged.get(key)
            k = item.keyword if item.keyword is not None else (seen.keyword if seen else None)
            v = item.vector if item.vector is not None else (seen.vector if seen else None)
            merged[key] = Ranked(item.row, max(x for x in (k, v) if x is not None), keyword=k, vector=v)
    return [merged[key] for key in sorted(scores, key=scores.get, reverse=True)]


async def search(dsn: str, *, collection: str, query: str, top_k: int, review_date: date,
                 embedder: Embedder | None = None, order: str = "rrf") -> tuple[str, list[Ranked], list[str]]:
    """Return (mode, ranked, corpus_versions). Raises psycopg.Error on database failure.

    An empty candidate list means the collection has nothing indexed for the date.
    """
    async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=3, row_factory=dict_row) as conn:
        await conn.execute("SET statement_timeout = '5000ms'")
        rows = await (await conn.execute(CANDIDATES_SQL, {"collection": collection, "review_date": review_date})).fetchall()
        if not rows:
            return "keyword", [], []
        corpus = sorted({row["corpus_version"] for row in rows})
        # Long articles are stored as 항/호 children: rank the children, answer with the whole article.
        keyword = keyword_rank(query, rows)
        if embedder is None:
            return "keyword", collapse(keyword, Ranked)[:top_k], corpus
        # The query embedding is an API call (OpenAI) or model inference; keep it off the event loop.
        query_vector = (await asyncio.to_thread(embedder.embed, [query]))[0]
        vector_rows = await (await conn.execute(VECTOR_SQL, {
            "vector": str(query_vector), "ids": [row["chunk_id"] for row in rows],
            "model": embedder.model_name, "limit": max(top_k * 8, 40),  # several children per article
        })).fetchall()
        if not vector_rows:
            return "keyword", collapse(keyword, Ranked)[:top_k], corpus
        by_id = {row["chunk_id"]: row for row in rows}
        vector = [Ranked(by_id[v["chunk_id"]], c, vector=c)
                  for v in vector_rows for c in [round(max(0.0, min(1.0, v["cosine"])), 4)]]
        merged = {"confirm": confirm, "cascade": cascade}.get(order, fuse)(keyword, vector)
        return "hybrid", collapse(merged, Ranked)[:top_k], corpus


INVENTORY_SQL = """
SELECT d.document_version_id::text, d.document_key, d.title, d.collection_key, d.source_type, d.version_label,
       d.effective_from, d.effective_until, d.corpus_version, count(c.chunk_id) AS chunks,
       count(c.embedding) AS embedded
FROM knowledge.document_versions d
LEFT JOIN knowledge.chunks c ON c.document_version_id = d.document_version_id AND c.parent_chunk_id IS NULL
GROUP BY d.document_version_id
ORDER BY d.document_key, d.effective_from NULLS LAST
"""


async def corpus_inventory(dsn: str) -> list[dict]:
    async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=3, row_factory=dict_row) as conn:
        rows = await (await conn.execute(INVENTORY_SQL)).fetchall()
    return [{**row, "effective_from": row["effective_from"] and row["effective_from"].isoformat(),
             "effective_until": row["effective_until"] and row["effective_until"].isoformat()} for row in rows]


DOCUMENT_TEXT_SQL = """
SELECT d.document_key, d.title, d.version_label, d.source_type, d.source_url,
       d.effective_from, d.effective_until, d.corpus_version,
       c.chunk_index, c.section, c.page_from, c.body
FROM knowledge.document_versions d JOIN knowledge.chunks c USING (document_version_id)
WHERE d.document_version_id = %(id)s::uuid AND c.parent_chunk_id IS NULL
ORDER BY c.chunk_index
"""


async def document_text(dsn: str, document_version_id: str) -> dict | None:
    """Verbatim text of one stored document version in chunk order. None when the version does not exist."""
    async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=3, row_factory=dict_row) as conn:
        rows = await (await conn.execute(DOCUMENT_TEXT_SQL, {"id": document_version_id})).fetchall()
    if not rows:
        return None
    head = rows[0]
    return {
        "document_version_id": document_version_id, "document_key": head["document_key"], "title": head["title"],
        "version": head["version_label"], "source_type": head["source_type"], "source_url": head["source_url"],
        "effective_from": head["effective_from"] and head["effective_from"].isoformat(),
        "effective_until": head["effective_until"] and head["effective_until"].isoformat(),
        "corpus_version": head["corpus_version"],
        "sections": [{"section": row["section"], "page": row["page_from"], "text": row["body"]} for row in rows],
    }


DOCUMENT_SQL = """
SELECT DISTINCT ON (document_key) document_version_id, document_key, title, version_label, source_url, storage_key,
       source_type, effective_from, effective_until, corpus_version
FROM knowledge.document_versions
WHERE document_key = ANY(%(keys)s)
  AND (effective_from IS NULL OR effective_from <= %(review_date)s)
  AND (effective_until IS NULL OR effective_until >= %(review_date)s)
ORDER BY document_key, effective_from DESC NULLS LAST, retrieved_at DESC
"""

CHUNKS_SQL = """
SELECT chunk_id::text, document_version_id, section, context, body, page_from
FROM knowledge.chunks WHERE document_version_id = ANY(%(ids)s) AND parent_chunk_id IS NULL ORDER BY chunk_index
"""

LINKS_SQL = """
SELECT source_law, source_section, target_law, target_section, relation FROM knowledge.article_links
WHERE (source_law = %(law)s AND source_section = %(section)s) OR (target_law = %(law)s AND target_section = %(section)s)
"""


async def load_documents(conn, keys: list[str], review_date: date) -> dict[str, dict]:
    """Current version of each document with its chunks: {document_key: {**version, "chunks": [...]}}."""
    docs = {row["document_key"]: {**row, "chunks": []} for row in
            await (await conn.execute(DOCUMENT_SQL, {"keys": keys, "review_date": review_date})).fetchall()}
    by_version = {doc["document_version_id"]: doc for doc in docs.values()}
    if by_version:
        for chunk in await (await conn.execute(CHUNKS_SQL, {"ids": list(by_version)})).fetchall():
            by_version[chunk["document_version_id"]]["chunks"].append(chunk)
    return docs


def provision_row(doc: dict, chunk: dict, **extra) -> dict:
    return {"chunk_id": chunk["chunk_id"], "section": chunk["section"], "context": chunk["context"], "body": chunk["body"],
            "page_from": chunk["page_from"], "document_key": doc["document_key"], "title": doc["title"],
            "version_label": doc["version_label"], "source_url": doc["source_url"], "storage_key": doc["storage_key"],
            "source_type": doc["source_type"], "effective_from": doc["effective_from"],
            "effective_until": doc["effective_until"], "corpus_version": doc["corpus_version"], **extra}


async def related_rows(conn, docs_by_law: dict[str, list[dict]], law: str, label: str, review_date: date,
                       query: str, limit: int = 3) -> list[dict]:
    """Follow article_links one hop in both directions from (law, label)."""
    from .links import article_label  # local import keeps links free of retrieval imports
    rows = await (await conn.execute(LINKS_SQL, {"law": law, "section": label})).fetchall()
    found, seen = [], set()
    for link in rows:
        forward = link["source_law"] == law and link["source_section"] == label
        target_law, target_label = (link["target_law"], link["target_section"]) if forward else (link["source_law"], link["source_section"])
        relation = link["relation"] if forward else {"cites": "cited_by", "implements": "implemented_by", "annex": "annex_of"}[link["relation"]]
        for doc in docs_by_law.get(target_law, []):
            chunks = [c for c in doc["chunks"] if article_label(c["section"]) == target_label]
            if target_label.startswith("별표") and len(chunks) > 1:
                chunks = [r.row for r in keyword_rank(query, chunks)[:1]] or chunks[:1]
            for chunk in chunks[:1]:
                key = (doc["document_key"], chunk["section"])
                if key not in seen:
                    seen.add(key)
                    found.append(provision_row(doc, chunk, relation=relation))
    order = {"implemented_by": 0, "annex": 1, "cites": 2, "annex_of": 3, "implements": 4, "cited_by": 5}
    return sorted(found, key=lambda row: order.get(row["relation"], 9))[:limit]
