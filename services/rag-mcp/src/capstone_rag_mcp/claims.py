"""Resolve claim rules to the exact current provisions, and expand hits along article links."""
from datetime import date
import re

import psycopg
from psycopg.rows import dict_row

from capstone_shared.rag_contracts import (
    ClaimRulesResult,
    EvidenceRequirement,
    RegulationHit,
    RelatedProvision,
    RetrievalStatus,
    RuleMatch,
)
from .links import DOCUMENT_LAWS, article_label, law_of
from .retrieval import keyword_rank, load_documents, provision_row, related_rows
from .rules import item_line, load_rules, match_rules, section_matches

# A statute loaded from the Law API instead of legalize-kr (GitHub unreachable) answers for the same rule key.
KEY_FALLBACK = {f"legalize-kr:화장품법/{kind}": f"lawapi:화장품법/{kind}" for kind in ("법률", "시행령", "시행규칙")}


def pick(docs: dict, key: str) -> dict | None:
    return docs.get(key) or docs.get(KEY_FALLBACK.get(key, ""))


GENERIC_EFFECT_WORDS = re.compile(r"완화|개선|증가|감소|활성화|또는|적합|사용에|일시적|에\s|\(.*?\)|에 한함")


def expression_matches(section: str, phrase: str) -> bool:
    """An annex row applies when its core term (피부노화, 다크서클, 콜라겐 ...) appears in the phrase.

    Generic effect words (완화, 개선, 증가 ...) are removed first; they appear in almost every claim.
    """
    expression = re.sub(r"^별표\s*「|」$", "", section or "")
    compact_phrase = re.sub(r"\s+", "", phrase)
    for part in re.split(r"[,，]", GENERIC_EFFECT_WORDS.sub(" ", expression)):
        core = re.sub(r"\s+", "", part)
        if len(core) >= 2 and core in compact_phrase:
            return True
    return False


def acquisition_of(row: dict) -> str:
    key = row.get("storage_key") or ""
    return "legalize-kr" if key.startswith("legalize-kr@") else "lawapi" if key.startswith("lawapi:") else "pdf"


def to_hit(row: dict, *, similarity: float, retrieval: str = "search", excerpt: str | None = None,
           item: str | None = None, related: list[RelatedProvision] | None = None,
           keyword: float | None = None, vector: float | None = None) -> RegulationHit:
    return RegulationHit(
        source=row["title"], version=row["version_label"], scope=row["context"] or row["title"],
        source_type=row["source_type"] if row["source_type"] != "case" else "guideline",
        excerpt=excerpt or row["body"], similarity=round(max(0.0, min(1.0, similarity)), 4),
        document_key=row["document_key"], section=row["section"], page=row["page_from"],
        source_url=row["source_url"], acquisition=acquisition_of(row), effective_from=row["effective_from"],
        effective_until=row["effective_until"], retrieval=retrieval, item=item, related=related or [],
        keyword_score=keyword, vector_score=vector,
    )


def to_related(row: dict) -> RelatedProvision:
    return RelatedProvision(source=row["title"], section=row["section"], excerpt=row["body"],
                            version=row["version_label"], relation=row["relation"],
                            document_key=row["document_key"], source_url=row["source_url"])


def group_by_law(docs: dict[str, dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for key, doc in docs.items():
        law = law_of(key)
        if law:
            grouped.setdefault(law, []).append(doc)
    return grouped


async def expand(conn, row: dict, docs_by_law, review_date: date, query: str) -> list[RelatedProvision]:
    law, label = law_of(row["document_key"]), article_label(row["section"])
    if not law or not label:
        return []
    return [to_related(r) for r in await related_rows(conn, docs_by_law, law, label, review_date, query)]


async def connect(dsn: str):
    return await psycopg.AsyncConnection.connect(dsn, connect_timeout=3, row_factory=dict_row)


async def resolve_claim_rules(dsn: str, phrase: str, claim_type: str | None, review_date: date) -> ClaimRulesResult:
    version, _ = load_rules()
    matches = match_rules(phrase, claim_type)
    keys = sorted(set(DOCUMENT_LAWS) | set(KEY_FALLBACK.values()) | {p["document_key"] for m in matches for p in m.rule.provisions}
                  | {m.rule.evidence_rows["document_key"] for m in matches if m.rule.evidence_rows})
    async with await connect(dsn) as conn:
        await conn.execute("SET statement_timeout = '5000ms'")
        docs = await load_documents(conn, keys, review_date)
        if not docs:
            return ClaimRulesResult(status=RetrievalStatus.CORPUS_NOT_INDEXED, review_date=review_date,
                                    rules_version=version, warnings=["No provision is loaded for this date."])
        docs_by_law = group_by_law(docs)
        out = []
        for match in matches:
            rule, provisions, missing, seen = match.rule, [], [], set()
            for spec in rule.provisions:
                doc = pick(docs, spec["document_key"])
                chunks = [c for c in doc["chunks"] if section_matches(c["section"], spec["section"])] if doc else []
                if spec.get("query") and len(chunks) > 1:
                    chunks = [r.row for r in keyword_rank(f"{spec['query']} {phrase}", chunks)[:1]] or chunks[:1]
                if not chunks:
                    missing.append(f"{spec['document_key']} {spec['section']}")
                    continue
                row = provision_row(doc, chunks[0])
                item, excerpt = item_line(row["body"], spec["item_hint"]) if spec.get("item_hint") else (None, None)
                key = (row["document_key"], row["section"], item)
                if key in seen:
                    continue
                seen.add(key)
                provisions.append(to_hit(row, similarity=1.0, retrieval="rule", excerpt=excerpt, item=item,
                                         related=await expand(conn, row, docs_by_law, review_date, phrase)))
            evidence = []
            if rule.evidence_rows and pick(docs, rule.evidence_rows["document_key"]):
                doc = pick(docs, rule.evidence_rows["document_key"])
                rows = [c for c in doc["chunks"] if (c["section"] or "").startswith(rule.evidence_rows["section_prefix"])]
                for chunk in [c for c in rows if expression_matches(c["section"], phrase)][:2]:
                    requirement = next((l.strip(" ▸") for l in chunk["body"].split("\n") if "▸" in l), chunk["body"])
                    expression = re.sub(r"^별표\s*「|」$", "", chunk["section"])
                    evidence.append(EvidenceRequirement(expression=expression, requirement=requirement,
                                                        source=doc["title"], section=chunk["section"],
                                                        version=doc["version_label"], source_url=doc["source_url"]))
            out.append(RuleMatch(rule_id=rule.id, label=rule.label, matched_by=match.matched_by,
                                 matched_text=match.matched_text, checks=list(rule.checks), provisions=provisions,
                                 evidence_requirements=evidence, missing=missing))
    status = RetrievalStatus.FOUND if any(m.provisions for m in out) else RetrievalStatus.NO_RELEVANT_EVIDENCE
    return ClaimRulesResult(status=status, review_date=review_date, rules_version=version, matches=out,
                            warnings=[] if out else ["No claim rule matched. Use search_regulations for this phrase."])


async def related_for_hits(dsn: str, rows: list[dict], review_date: date, query: str) -> list[list[RelatedProvision]]:
    async with await connect(dsn) as conn:
        docs = await load_documents(conn, sorted(DOCUMENT_LAWS), review_date)
        docs_by_law = group_by_law(docs)
        return [await expand(conn, row, docs_by_law, review_date, query) for row in rows]


INGREDIENT_ANNEXES = {"별표 1": "prohibited", "별표 2": "restricted"}
SALT_SUFFIX = re.compile(r"(및\s*그\s*염류|및\s*그\s*에스텔류|및\s*그\s*염류\s*및\s*에스텔류)$")


def parse_box_table(text: str) -> list[list[str]]:
    """Rebuild rows of a box-drawing table: cells wrap over several lines between ├ separators."""
    rows, current = [], []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in "├└┌":
            if current:
                rows.append([" ".join(part for part in cell if part).strip() for cell in current])
            current = []
            continue
        if "│" not in stripped:
            continue
        cells = [c.strip() for c in stripped.strip("│").split("│")]
        while len(current) < len(cells):
            current.append([])
        for index, cell in enumerate(cells):
            current[index].append(cell)
    if current:
        rows.append([" ".join(part for part in cell if part).strip() for cell in current])
    return rows


def squeeze(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


# 고시와 광고가 같은 원료를 다르게 적는 흔한 표기 (하이드로퀴논 / 히드로퀴논)
SPELLINGS = [("하이드로", "히드로"), ("메틸", "메칠"), ("에틸", "에칠"), ("애시드", "애씨드"), ("에시드", "애씨드")]


def spelling_variants(name: str) -> set[str]:
    variants = {name}
    for a, b in SPELLINGS:
        for value in list(variants):
            variants |= {value.replace(a, b), value.replace(b, a)}
    return variants


async def find_ingredient(dsn: str, name: str, review_date: date) -> dict:
    """Whether an ingredient is listed in 화장품 안전기준 등에 관한 규정 별표 1 (사용할 수 없는 원료) or
    별표 2 (사용상의 제한이 필요한 원료). Exact name or alias match on rebuilt table rows; a partial match is
    reported as a mention only. Synonyms that are not written in the annex are not resolved."""
    target = squeeze(name)
    targets = spelling_variants(target)
    async with await connect(dsn) as conn:
        docs = await load_documents(conn, ["lawapi:S10"], review_date)
    doc = docs.get("lawapi:S10")
    if not doc:
        return {"status": "not_indexed", "ingredient": name, "listings": [], "mentions": []}
    listings, mentions = [], []
    for annex, listing in INGREDIENT_ANNEXES.items():
        text = "\n".join(c["body"] for c in doc["chunks"] if article_label(c["section"]) == annex)
        for row in parse_box_table(text):
            if not row or not row[0] or row[0].startswith("원료명"):
                continue
            names = {squeeze(SALT_SUFFIX.sub("", row[0])), squeeze(row[0])} | ({squeeze(row[-1])} if len(row) > 2 else set())
            entry = {"listing": listing, "annex": annex, "name": row[0], "detail": " | ".join(row[1:])[:200],
                     "source": doc["title"], "version": doc["version_label"], "source_url": doc["source_url"]}
            if targets & names:
                listings.append(entry)
            elif len(target) >= 3 and any(t in n for t in targets for n in names if n):
                mentions.append(entry)
    status = "prohibited" if any(l["listing"] == "prohibited" for l in listings) else \
        "restricted" if listings else "not_listed"
    return {"status": status, "ingredient": name, "listings": listings[:3], "mentions": mentions[:5]}
