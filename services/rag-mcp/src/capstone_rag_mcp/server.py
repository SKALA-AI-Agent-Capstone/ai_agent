from datetime import date
import os
from typing import Annotated
import uuid

from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
import psycopg
from pydantic import Field
from starlette.responses import JSONResponse

from capstone_shared.probes import database_probe
from capstone_shared.rag_contracts import (
    ClaimRulesResult,
    PrecedentHit,
    PrecedentSearchResult,
    RegulationHit,
    RegulationSearchResult,
    RetrievalStatus,
)
from capstone_shared.settings import internal_token, required_env
from .claims import find_ingredient, related_for_hits, resolve_claim_rules, to_hit
from .embedding import load_embedder
from .retrieval import Ranked, corpus_inventory, document_text, search


def regulation_hit(item: Ranked, related=None) -> RegulationHit:
    return to_hit(item.row, similarity=item.similarity, related=related, keyword=item.keyword, vector=item.vector)


def create_server() -> FastMCP:
    verifier = StaticTokenVerifier(
        tokens={internal_token(): {"client_id": "capstone-agent", "scopes": ["tools:read"]}},
        required_scopes=["tools:read"],
    )
    mcp = FastMCP("rag-mcp", auth=verifier)

    @mcp.custom_route("/health/live", methods=["GET"])
    async def live(request) -> JSONResponse:
        return JSONResponse({"service": "rag-mcp", "status": "ok"})

    @mcp.custom_route("/health/ready", methods=["GET"])
    async def ready(request) -> JSONResponse:
        db = await database_probe(required_env("RAG_DATABASE_URL"), require_vector=True)
        return JSONResponse(
            {"service": "rag-mcp", "status": db["status"], "database": db},
            status_code=200 if db["status"] == "ok" else 503,
        )

    @mcp.custom_route("/health/corpus", methods=["GET"])
    async def corpus(request) -> JSONResponse:
        """Public corpus inventory for the standards page. Lists document versions, never chunk text."""
        try:
            documents = await corpus_inventory(required_env("RAG_DATABASE_URL"))
        except psycopg.Error:
            return JSONResponse({"service": "rag-mcp", "status": "unavailable", "documents": []}, status_code=503)
        return JSONResponse({"service": "rag-mcp", "status": "indexed" if documents else "not_indexed",
                             "embeddingModel": embedder.model_name if embedder else None, "documents": documents})

    embedder = load_embedder()

    # Resources are read-only originals chosen by the app (the basis panel), not by the model.
    # Search tools return excerpts; these return the stored source as ingested.
    @mcp.resource("regulation://corpus", name="regulation_corpus", mime_type="application/json")
    async def regulation_corpus() -> dict:
        """Every stored source document version (law, regulation, guideline, association standard, cases)
        with its document_version_id, version label, effective dates and chunk count.
        Pass a document_version_id to regulation://documents/{document_version_id} to read that source in full."""
        try:
            return {"documents": await corpus_inventory(required_env("RAG_DATABASE_URL"))}
        except psycopg.Error:
            raise ResourceError("Knowledge database unavailable. No document list was read.") from None

    @mcp.resource("regulation://documents/{document_version_id}", name="regulation_document",
                  mime_type="application/json")
    async def regulation_document(document_version_id: str) -> dict:
        """Verbatim text of one stored source document version in its original order (조, 항, 별표, page),
        with source URL and effective dates. Read-only. Text is what was ingested; check source_url for the
        official original before relying on it."""
        try:
            uuid.UUID(document_version_id)
        except ValueError:
            raise ResourceError("Unknown document_version_id. Read regulation://corpus for valid ids.") from None
        try:
            document = await document_text(required_env("RAG_DATABASE_URL"), document_version_id)
        except psycopg.Error:
            raise ResourceError("Knowledge database unavailable. No document was read.") from None
        if document is None:
            raise ResourceError("Unknown document_version_id. Read regulation://corpus for valid ids.")
        return document

    @mcp.tool(annotations={"readOnlyHint": True})
    async def search_regulations(
        query_text: Annotated[str, Field(min_length=1, max_length=3000)],
        top_k: Annotated[int, Field(ge=1, le=10)] = 5,
        review_date: date | None = None,
    ) -> RegulationSearchResult:
        """Search statutes and official standards (collection A: 화장품법, 시행규칙, 식약처 지침, 고시, 협회 기준).

        Pass one advertising claim with its context as query_text. Returns verbatim excerpts with
        source, version and effective dates, filtered to the versions in force on review_date
        (default today). similarity is a retrieval score, not a violation probability.
        Check status: CORPUS_NOT_INDEXED and SEARCH_ERROR mean nothing was checked.
        """
        day = review_date or date.today()
        try:
            mode, ranked, corpus = await search(required_env("RAG_DATABASE_URL"), collection="rules",
                                                query=query_text, top_k=top_k, review_date=day, embedder=embedder)
        except psycopg.Error:
            return RegulationSearchResult(status=RetrievalStatus.SEARCH_ERROR, review_date=day,
                                          warnings=["Knowledge database unavailable. No regulation was checked."])
        if not corpus:
            return RegulationSearchResult(status=RetrievalStatus.CORPUS_NOT_INDEXED, review_date=day,
                                          warnings=["No regulation is indexed for this date. Run scripts/ingest_rag.py."])
        try:
            related = await related_for_hits(required_env("RAG_DATABASE_URL"), [i.row for i in ranked], day, query_text)
        except psycopg.Error:
            related = [[] for _ in ranked]
        hits = [regulation_hit(item, links) for item, links in zip(ranked, related)]
        return RegulationSearchResult(
            status=RetrievalStatus.FOUND if hits else RetrievalStatus.NO_RELEVANT_EVIDENCE,
            review_date=day, retrieval_mode=mode, corpus_versions=corpus, hits=hits,
            warnings=[] if hits else ["No matching provision. This is not a compliance finding."],
        )

    @mcp.tool(annotations={"readOnlyHint": True})
    async def lookup_claim_rules(
        phrase: Annotated[str, Field(min_length=1, max_length=1000)],
        claim_type: Annotated[str | None, Field(max_length=40)] = None,
        review_date: date | None = None,
    ) -> ClaimRulesResult:
        """Deterministic legal basis for one advertising claim (call this before search_regulations).

        Matches the claim against the versioned claim rule table by claim_type (drug, wrinkle, whitening,
        clinical, absolute, exaggeration, ingredient, patent) and by patterns in the phrase. For each rule it
        returns the exact current provisions (e.g. 화장품법 제13조 제1항제1호, 시행규칙 별표 5 제2호 가목),
        provisions linked to them, required substantiation rows from 실증 규정 별표, and the cross-checks to run
        (functional_report, substantiation, ingredient, gmp, patent). A match is a basis to review, not a verdict.
        """
        day = review_date or date.today()
        try:
            return await resolve_claim_rules(required_env("RAG_DATABASE_URL"), phrase, claim_type, day)
        except psycopg.Error:
            from .rules import load_rules
            return ClaimRulesResult(status=RetrievalStatus.SEARCH_ERROR, review_date=day, rules_version=load_rules()[0],
                                    warnings=["Knowledge database unavailable. No rule was resolved."])

    @mcp.tool(annotations={"readOnlyHint": True})
    async def scan_claim_lines(lines: Annotated[list[str], Field(max_length=2000)]) -> dict:
        """Find ad text lines that match a high-signal claim rule (의약품, 식약처나 전문가 추천, GMP, 특허, 수치,
        절대 표현, 기능성 효능, 줄기세포). Pure pattern match on the versioned rule table, no database.
        Use it to add claims the model did not extract; each returned line is verbatim input text."""
        from .rules import load_rules, scan_lines
        return {"rules_version": load_rules()[0], "matches": scan_lines([l[:500] for l in lines])}

    @mcp.tool(annotations={"readOnlyHint": True})
    async def check_ingredient_standard(
        ingredient: Annotated[str, Field(min_length=2, max_length=100)],
        review_date: date | None = None,
    ) -> dict:
        """Check one ingredient name against 화장품 안전기준 등에 관한 규정 (Korean cosmetics safety standard):
        별표 1 사용할 수 없는 원료 -> prohibited, 별표 2 사용상의 제한이 필요한 원료 -> restricted, otherwise not_listed.
        Exact text match on the loaded official annex. not_listed is not a safety finding (synonyms are not resolved)."""
        try:
            return await find_ingredient(required_env("RAG_DATABASE_URL"), ingredient, review_date or date.today())
        except psycopg.Error:
            return {"status": "error", "ingredient": ingredient, "listings": [], "error": "KNOWLEDGE_DB_UNAVAILABLE"}

    @mcp.tool(annotations={"readOnlyHint": True})
    async def search_precedents(
        query_text: Annotated[str, Field(min_length=1, max_length=3000)],
        top_k: Annotated[int, Field(ge=1, le=10)] = 3,
    ) -> PrecedentSearchResult:
        """Search past violation cases (collection B). Supplementary only, never a legal basis.

        Every hit is trust_level=low and verified=False and must be shown as an unverified reference.
        """
        try:
            mode, ranked, corpus = await search(required_env("RAG_DATABASE_URL"), collection="cases",
                                                query=query_text, top_k=top_k, review_date=date.today(),
                                                embedder=embedder)
        except psycopg.Error:
            return PrecedentSearchResult(status=RetrievalStatus.SEARCH_ERROR,
                                         warnings=["Knowledge database unavailable. No case was checked."])
        if not corpus:
            return PrecedentSearchResult(status=RetrievalStatus.CORPUS_NOT_INDEXED,
                                         warnings=["No case collection is indexed yet."])
        hits = [PrecedentHit(source=item.row["title"], excerpt=item.row["body"], similarity=item.similarity,
                             keyword_score=item.keyword, vector_score=item.vector,
                             document_key=item.row["document_key"], page=item.row["page_from"],
                             source_url=item.row["source_url"]) for item in ranked]
        return PrecedentSearchResult(
            status=RetrievalStatus.FOUND if hits else RetrievalStatus.NO_RELEVANT_EVIDENCE,
            retrieval_mode=mode, corpus_versions=corpus, hits=hits,
        )

    return mcp


def main() -> None:
    create_server().run(
        transport="http",
        host=os.getenv("SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVICE_PORT", "8003")),
    )
