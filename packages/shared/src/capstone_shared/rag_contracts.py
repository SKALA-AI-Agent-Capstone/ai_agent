"""Notion target RAG tools: search_regulations and search_precedents.

The Notion draft returns a bare list. These envelopes add a status so an empty
corpus or a database failure is never read as "no violation found".
"""
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from capstone_shared import CONTRACT_VERSION


class RetrievalStatus(StrEnum):
    FOUND = "FOUND"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    CORPUS_NOT_INDEXED = "CORPUS_NOT_INDEXED"
    SEARCH_ERROR = "SEARCH_ERROR"


class RetrievalMode(StrEnum):
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RelatedProvision(BaseModel):
    """A provision reached by following a citation, delegation or annex link from a hit."""
    source: str
    section: str | None = None
    excerpt: str
    version: str
    relation: Literal["cites", "cited_by", "implements", "implemented_by", "annex", "annex_of"]
    document_key: str
    source_url: str


class RegulationHit(BaseModel):
    # Notion RegulationHit fields
    source: str
    version: str
    scope: str
    excerpt: str
    similarity: float = Field(ge=0, le=1, description="Retrieval score, not a violation probability.")
    # 하이브리드 검색에서 어느 신호로 찾았는지. 척도가 달라 기준값도 따로 쓴다(키워드 일치율, 벡터 코사인).
    keyword_score: float | None = Field(default=None, ge=0, le=1)
    vector_score: float | None = Field(default=None, ge=0, le=1)
    trust_level: Literal["high"] = "high"
    source_type: Literal["law", "regulation", "guideline", "association"]
    # Traceability for the report (NFR-11)
    document_key: str
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    source_url: str
    acquisition: str
    effective_from: date | None = None
    effective_until: date | None = None
    # How the provision was found: keyword/vector search, a claim rule, or a link from another provision
    retrieval: Literal["search", "rule", "link"] = "search"
    item: str | None = Field(default=None, description="Numbered item inside the article, e.g. 제1항제1호")
    related: list[RelatedProvision] = Field(default_factory=list)


class EvidenceRequirement(BaseModel):
    """A row of the 화장품 표시광고 실증 규정 annex: advertised expression -> required substantiation."""
    expression: str
    requirement: str
    source: str
    section: str
    version: str
    source_url: str


class RuleMatch(BaseModel):
    rule_id: str
    label: str
    matched_by: Literal["claim_type", "pattern"]
    matched_text: str
    checks: list[str] = Field(default_factory=list, description="Cross-checks the agent should run")
    provisions: list[RegulationHit] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list, description="Rule provisions not present in the loaded corpus")


class ClaimRulesResult(BaseModel):
    contract_version: str = CONTRACT_VERSION
    status: RetrievalStatus
    review_date: date
    rules_version: str
    matches: list[RuleMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PrecedentHit(BaseModel):
    source: str
    excerpt: str
    similarity: float = Field(ge=0, le=1)
    # 하이브리드 검색에서 어느 신호로 찾았는지. 척도가 달라 기준값도 따로 쓴다(키워드 일치율, 벡터 코사인).
    keyword_score: float | None = Field(default=None, ge=0, le=1)
    vector_score: float | None = Field(default=None, ge=0, le=1)
    trust_level: Literal["low"] = "low"
    verified: Literal[False] = False
    document_key: str
    page: int | None = Field(default=None, ge=1)
    source_url: str


class RegulationSearchResult(BaseModel):
    contract_version: str = CONTRACT_VERSION
    status: RetrievalStatus
    review_date: date
    retrieval_mode: RetrievalMode | None = None
    corpus_versions: list[str] = Field(default_factory=list)
    hits: list[RegulationHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PrecedentSearchResult(BaseModel):
    contract_version: str = CONTRACT_VERSION
    status: RetrievalStatus
    retrieval_mode: RetrievalMode | None = None
    corpus_versions: list[str] = Field(default_factory=list)
    hits: list[PrecedentHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
