"""Legal basis for each claim: rule-mapped provisions first, search as a supplement, cases as reference.

1. rag.lookup_claim_rules  the versioned claim rule table gives the exact provisions for the claim type
                           and phrase (제13조 제1항제1호, 별표 5 제2호 가목 ...), linked provisions,
                           substantiation rows and which cross-checks to run.
2. rag.search_regulations  catches guideline and association text the rules do not name.
3. rag.search_precedents   past violation cases, shown as unverified reference only.

Hits are citation candidates for the reviewer. The agent does not decide legality here.
"""
import asyncio
from datetime import date
from difflib import SequenceMatcher
import re

from .tools import McpTools
from .claim_screening import is_subjective_only

# Statute wording per claim type, used for the supplementary search query.
CLAIM_HINTS = {
    "drug": "의약품으로 잘못 인식할 우려가 있는 표시 또는 광고",
    "wrinkle": "기능성화장품 주름개선 심사결과와 다른 표시 또는 광고",
    "whitening": "기능성화장품 미백 심사결과와 다른 표시 또는 광고",
    "clinical": "표시ㆍ광고 내용의 실증 시험 자료",
    "absolute": "사실과 다르게 소비자를 속이거나 잘못 인식하도록 할 우려가 있는 표시 또는 광고",
    "exaggeration": "사실과 다르게 소비자를 속이거나 잘못 인식하도록 할 우려가 있는 표시 또는 광고",
    "ingredient": "화장품 원료 성분 표시 기재",
    "patent": "표시ㆍ광고 내용의 실증 특허",
}
# 키워드 일치율과 벡터 코사인은 척도가 달라 기준값을 따로 둔다. text-embedding-3-small(1024차원)에서
# 광고가 아닌 문장은 코사인 0.38 이하, 광고 주장은 0.46 이상이었다(docs/12-hybrid-retrieval.md).
SEARCH_MIN_SIMILARITY = 0.45
SEARCH_MIN_VECTOR = 0.45
MAX_SEARCH_BASIS = 1
PRECEDENT_MIN_SIMILARITY = 0.3
PRECEDENT_MIN_VECTOR = 0.45
SOURCE_TYPES = {"law": "law", "regulation": "law", "guideline": "guideline", "association": "association"}


async def source_document(read, document_key: str, version: str) -> dict | None:
    """Full original of one basis source, read through the rag-mcp resources for the basis panel.

    The basis keeps document_key and version label, so regulation://corpus resolves the document_version_id
    and regulation://documents/{id} returns its sections. Another version of the same law is never substituted.
    """
    corpus = await read("rag", "regulation://corpus")
    match = next((d for d in corpus["documents"]
                  if d["document_key"] == document_key and d["version_label"] == version), None)
    if match is None:
        return None
    return await read("rag", f"regulation://documents/{match['document_version_id']}")


def build_queries(finding: dict) -> list[str]:
    """One search per legal angle. Mixing two hints in one query dilutes both, so they run separately."""
    claim_type = finding.get("claimType", "")
    queries = [f"{finding['phrase']}\n{CLAIM_HINTS.get(claim_type, CLAIM_HINTS['exaggeration'])}"]
    if claim_type != "clinical" and re.search(r"\d", finding["phrase"]):
        queries.append(f"{finding['phrase']}\n{CLAIM_HINTS['clinical']}")
    return queries


def to_basis(hit: dict, rule: dict | None = None) -> dict:
    """Map a RegulationHit to the frontend basis shape (BasisItem.vue)."""
    location = hit.get("section") or (f"p.{hit['page']}" if hit.get("page") else "")
    article = " ".join(p for p in (location, hit.get("item")) if p)
    if rule:
        how = "주장 유형" if rule["matched_by"] == "claim_type" else "문구"
        plain = (f"'{rule['label']}' 규칙에 따라 붙은 조항입니다 ({how}: {rule['matched_text']}). "
                 "적용 여부와 위반 판단은 담당자가 원문으로 확인합니다.")
    else:
        plain = "검색으로 찾은 관련 기준 후보입니다. 적용 여부와 위반 판단은 담당자가 원문으로 확인합니다."
    expression = re.search(r"금지 표현:\s*[·•]?\s*([^\n]+)", hit.get("excerpt", ""))
    terms = re.findall(r"[가-힣]{2,}", expression.group(1)) if expression and "[별표2]" in hit.get("excerpt", "") else []
    def score(row):
        return sum(term in row.get("excerpt", "") for term in terms)
    selected = {}
    for row in hit.get("related", []):
        key = (row.get("source"), row.get("section"), row.get("relation"))
        if key not in selected or score(row) > score(selected[key]):
            selected[key] = row
    related = []
    for row in selected.values():
        related.append({"sourceName": row["source"], "article": row.get("section"), "relation": row["relation"],
                        "original": row["excerpt"], "version": row["version"], "sourceUrl": row.get("source_url")})
    return {
        "sourceId": hit["document_key"],
        "sourceType": SOURCE_TYPES.get(hit.get("source_type", ""), "law"),
        "sourceName": hit["source"],
        "article": article,
        "plain": plain,
        "original": hit["excerpt"],
        "version": hit["version"],
        "revisedAt": hit.get("effective_from"),
        "scope": hit.get("scope"),
        "sourceUrl": hit.get("source_url"),
        "similarity": hit.get("similarity"),
        "keywordScore": hit.get("keyword_score"),
        "vectorScore": hit.get("vector_score"),
        "retrieval": hit.get("retrieval", "search"),
        "ruleId": rule["rule_id"] if rule else None,
        "related": related,
    }


def relevant(hit: dict, keyword_min: float, vector_min: float) -> bool:
    """A hit counts when either signal clears its own threshold. Older servers send only similarity."""
    keyword, vector = hit.get("keyword_score"), hit.get("vector_score")
    if keyword is None and vector is None:
        return hit.get("similarity", 0) >= keyword_min
    return (keyword or 0) >= keyword_min or (vector or 0) >= vector_min


def select_hits(hits: list[dict]) -> list[dict]:
    return [h for h in hits if relevant(h, SEARCH_MIN_SIMILARITY, SEARCH_MIN_VECTOR)][:MAX_SEARCH_BASIS]


async def basis_for(call, finding: dict, day: str) -> dict:
    phrase, claim_type = finding["phrase"], finding.get("claimType")
    rules, search, cases = await asyncio.gather(
        call("rag", "lookup_claim_rules", {"phrase": phrase, "claim_type": claim_type, "review_date": day}),
        call("rag", "search_regulations", {"query_text": build_queries(finding)[0], "top_k": 3, "review_date": day}),
        call("rag", "search_precedents", {"query_text": phrase, "top_k": 2}),
        return_exceptions=True)
    statuses = [r.get("status") if isinstance(r, dict) else "SEARCH_ERROR" for r in (rules, search)]
    basis, seen = [], set()
    rule_view, evidence, checks, missing = [], [], set(), []
    if isinstance(rules, dict):
        for match in rules.get("matches", []):
            rule_view.append({"id": match["rule_id"], "label": match["label"], "matchedBy": match["matched_by"],
                              "matchedText": match["matched_text"], "checks": match["checks"], "missing": match["missing"]})
            checks |= set(match["checks"])
            missing += match["missing"]
            evidence += [{**e, "ruleId": match["rule_id"]} for e in match["evidence_requirements"]]
            for hit in match["provisions"]:
                key = (hit["document_key"], hit.get("section"), hit.get("item"))
                if key not in seen:
                    seen.add(key)
                    basis.append(to_basis(hit, match))
    if isinstance(search, dict):
        rule_articles = {(doc, section) for doc, section, _ in seen}
        for hit in select_hits(search.get("hits", [])):
            # The rule already cites this article (with its item); a search copy would only repeat it.
            if (hit["document_key"], hit.get("section")) not in rule_articles:
                seen.add((hit["document_key"], hit.get("section"), hit.get("item")))
                basis.append(to_basis(hit))
    references = []
    if isinstance(cases, dict):
        references = [{"id": f"{c['document_key']}:{c.get('page')}", "verified": False, "summary": c["excerpt"][:300],
                       "source": c["source"], "page": c.get("page"), "sourceUrl": c.get("source_url"),
                       "similarity": c.get("similarity")}
                      for c in cases.get("hits", []) if relevant(c, PRECEDENT_MIN_SIMILARITY, PRECEDENT_MIN_VECTOR)]
    if all(s == "SEARCH_ERROR" for s in statuses):
        state = "failed"
    elif all(s == "CORPUS_NOT_INDEXED" for s in statuses):
        state = "not_indexed"
    else:
        state = "basis_found" if basis else "no_match"
    return {"basis": basis, "rules": rule_view, "evidence": evidence, "checks": sorted(checks), "missing": missing,
            "references": references, "state": state}


SWEEP_CLAIM_TYPES = {"drug": "drug", "patent": "patent", "substantiation": "clinical", "absolute_comparison": "absolute"}


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


async def sweep_findings(call, texts: list[dict], findings: list[dict]) -> list[dict]:
    """Add OCR lines that match a high-signal rule but that the model did not extract (NFR-2: recall first).

    The phrase is the whole OCR line, so it is grounded in the ad text like model findings are.
    """
    texts = [text for text in texts if text.get("documentContext") != "patent_certificate"]
    if not texts:
        return []
    result = await call("rag", "scan_claim_lines", {"lines": [t["text"] for t in texts]})
    known = [squeeze(f["phrase"]) for f in findings]
    added = []
    for match in result.get("matches", []):
        line, source = match["line"].strip(), texts[match["index"]]
        compact = squeeze(line)
        if not compact or any(k and (k in compact or compact in k) for k in known):
            continue
        # The same figure or expression already sits in a claim that reads like this line (often an OCR-noisy duplicate).
        # Another sentence using the same word ("브라이트닝 듀얼 세럼" beside "…브라이트닝 세럼") is a claim of its own.
        if all(any(squeeze(token) in k and SequenceMatcher(None, compact, k).ratio() >= 0.6 for k in known)
               for token in match["matched"]):
            continue
        claim_type = next((SWEEP_CLAIM_TYPES[r] for r in match["rules"] if r in SWEEP_CLAIM_TYPES), None) or \
            ("wrinkle" if "주름" in line else "whitening" if re.search("미백|기미|브라이트닝|화이트닝|잡티", line) else "exaggeration")
        known.append(compact)
        added.append({"id": f"rule-{match['index']}", "phrase": line, "start": source["text"].find(line),
                      "end": source["text"].find(line) + len(line), "location": source["location"], "textId": source["id"],
                      "severity": "low", "baseSeverity": "low", "claimType": claim_type,
                      "reason": f"'{', '.join(match['matched'])}' 표현은 {', '.join(match.get('labels') or match['rules'])}에 해당해 근거를 확인해야 하는 문구입니다.",
                      "title": "규칙으로 찾은 확인할 문구", "basis": [], "baseBasis": [], "confidence": None, "rewrites": [],
                      "references": [], "category": "확인할 광고 주장", "legalStatus": "not_evaluated", "escalation": None,
                      "source": "rule_sweep"})
    return added


async def run_all(call, findings: list[dict], day: str) -> list[dict]:
    limit = asyncio.Semaphore(6)

    async def one(finding):
        async with limit:
            return await basis_for(call, finding, day)
    return await asyncio.gather(*(one(f) for f in findings))


async def legal_basis(findings: list[dict], *, review_date: date | None = None, call=None, texts: list[dict] | None = None) -> dict:
    """Return {"findings": [...], "status": connected|not_indexed|error|skipped, "snapshot": [...], "warnings": [...]}.

    Every finding is searched; there is no count limit (a cap once skipped the riskiest late claims).
    """
    day = (review_date or date.today()).isoformat()
    original_findings = findings
    subjective = [f for f in findings if is_subjective_only(f.get("phrase", ""))]
    findings = [f for f in findings if not is_subjective_only(f.get("phrase", ""))]
    subjective_dismissed = [
        {"id": f.get("id"), "phrase": f["phrase"], "textId": f.get("textId"),
         "reason": "주관적인 체감이나 바람만 표현함", "source": "subjective_screen"}
        for f in subjective
    ]

    async def run(session_call):
        swept = []
        if texts:
            try:
                swept = await sweep_findings(session_call, texts, original_findings)
            except Exception:
                swept = []
        combined, dismissed = list(findings), list(subjective_dismissed)
        for candidate in swept:
            if is_subjective_only(candidate.get("phrase", "")):
                dismissed.append({"id": candidate.get("id"), "phrase": candidate["phrase"],
                                  "textId": candidate.get("textId"), "reason": "주관적인 체감이나 바람만 표현함",
                                  "source": "subjective_screen"})
            else:
                combined.append(candidate)
        return combined, dismissed, (await run_all(session_call, combined, day) if combined else [])

    try:
        if call is None:
            async with McpTools(servers=("rag",)) as session_call:
                findings, dismissed, results = await run(session_call)
        else:
            findings, dismissed, results = await run(call)
    except Exception:
        return {"findings": findings, "dismissed": subjective_dismissed, "status": "error", "snapshot": [],
                "warnings": ["LEGAL_SEARCH_UNAVAILABLE"]}
    if not findings:
        return {"findings": findings, "dismissed": dismissed, "status": "skipped", "snapshot": [], "warnings": []}
    states = [r["state"] for r in results]
    if all(s == "not_indexed" for s in states):
        return {"findings": findings, "dismissed": dismissed, "status": "not_indexed", "snapshot": [],
                "warnings": ["LEGAL_CORPUS_NOT_INDEXED"]}
    if all(s == "failed" for s in states):
        return {"findings": findings, "dismissed": dismissed, "status": "error", "snapshot": [],
                "warnings": ["LEGAL_SEARCH_FAILED"]}
    enriched, snapshot, warnings = [], {}, []
    for finding, result in zip(findings, results):
        for item in result["basis"]:
            snapshot[(item["sourceName"], item["version"])] = {"name": item["sourceName"], "version": item["version"],
                                                               "revisedAt": item.get("revisedAt")}
        enriched.append({**finding, "basis": result["basis"], "baseBasis": result["basis"], "rules": result["rules"],
                         "evidenceRequirements": result["evidence"], "requiredChecks": result["checks"],
                         "references": result["references"] or finding.get("references", []),
                         "legalSearch": result["state"] if result["state"] in ("basis_found", "no_match") else "failed",
                         "title": "법령 근거 후보와 대조할 문구" if result["basis"] else finding.get("title")})
        if result["missing"]:
            warnings.append("LEGAL_RULE_PROVISION_NOT_LOADED")
    if any(r["state"] == "failed" for r in results):
        warnings.append("LEGAL_SEARCH_PARTIAL")
    return {"findings": enriched, "dismissed": dismissed, "status": "connected", "snapshot": list(snapshot.values()),
            "warnings": list(dict.fromkeys(warnings))}


# Older name used by rewrite_phrase and tests.
search_basis = legal_basis
