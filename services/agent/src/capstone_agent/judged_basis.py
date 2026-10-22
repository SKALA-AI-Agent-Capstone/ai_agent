"""Judge first, fetch basis on purpose, verify it apart. Only verified basis reaches a card.

1. judge    One model call reads every candidate claim (model extraction and rule sweep) with its line and decides
            whether it is a regulatory concern at all, which words cause it (span) and what the basis must show
            (purpose). Candidates judged not a concern leave the findings and are kept as dismissed.
2. gather   Per kept claim, a tool loop: the model picks rag-mcp tools (claim rules, statute search, case search)
            and writes the search query for its purpose. Guardrails: MAX_BASIS_CALLS calls, MAX_SAME_BASIS_TOOL
            per tool, BASIS_LOOP_SECONDS. The phrase and review date are filled in by code, never by the model.
3. verify   A separate call that sees only the ad phrase, its line and the candidate texts (not the judgment or the
            loop) decides for each candidate whether it governs this expression and quotes the sentence that does.
            A quote that is not in the candidate text is rejected by code. When nothing passes, the loop runs once
            more with the rejected candidates named, and the new candidates are verified the same way.

Without a model key the regulations node keeps the code-only path (regulations.legal_basis).
"""
import asyncio
from datetime import date
from difflib import SequenceMatcher
import json
import re
import time
from typing import Literal

from pydantic import BaseModel

from .instructions import instruction
from .regulations import sweep_findings, to_basis
from .tools import McpTools

ClaimType = Literal["wrinkle", "whitening", "clinical", "absolute", "ingredient", "patent", "exaggeration", "drug"]

MAX_BASIS_CALLS = 4
MAX_SAME_BASIS_TOOL = 2
BASIS_LOOP_SECONDS = 45
MAX_VERIFIED_BASIS = 2
CANDIDATE_TEXT = 1200


class Judgment(BaseModel):
    id: str
    problem: bool
    claimType: ClaimType
    span: str
    purpose: str
    reason: str


class ClaimJudgments(BaseModel):
    judgments: list[Judgment]


class BasisVerdict(BaseModel):
    id: str
    applies: bool
    quote: str
    why: str


class BasisCheck(BaseModel):
    verdicts: list[BasisVerdict]


BASIS_TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "lookup_claim_rules",
        "description": "주장 유형별 규칙 표에서 이 문구에 지정된 법령 조항을 찾는다. 의약품 오인, 기능성 효능, 실증이 필요한 "
                       "수치나 시험 표현처럼 규칙으로 정해진 조항이 있을 만한 주장에 쓴다.",
        "parameters": {"type": "object", "properties": {
            "claim_type": {"type": "string", "enum": list(ClaimType.__args__)}}, "required": ["claim_type"]}}},
    {"type": "function", "function": {
        "name": "search_regulations",
        "description": "법령, 고시, 가이드라인, 협회 기준 본문을 검색한다. query에는 판단 목적에 맞는 법적 쟁점을 "
                       "짧은 문장으로 쓴다(예: '인체적용시험 결과 표시 광고의 실증 요건').",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_precedents",
        "description": "비슷한 표현이 문제된 과거 행정처분 사례를 찾는다. 참고 자료가 판단에 도움이 될 때만 쓴다.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def line_of(finding: dict, texts_by_id: dict) -> str:
    return (texts_by_id.get(finding.get("textId")) or {}).get("text") or finding["phrase"]


def section_of(finding: dict, texts_by_id: dict, sections: dict) -> list[dict]:
    """The other copy blocks of the claim's section: its headline, numbers and test-condition footnotes."""
    own = texts_by_id.get(finding.get("textId")) or {}
    tile = (own.get("location") or finding.get("location") or {}).get("tileId")
    return [t for t in sections.get(tile, []) if t["id"] != own.get("id")]


ROLE_NAMES = {"claim": "주장", "result": "수치", "condition": "조건 각주"}


async def judge(provider, findings: list[dict], texts_by_id: dict, sections: dict) -> dict[str, Judgment]:
    """One call for all candidates. Each comes with its section's other copy as context, so a number is judged
    with the headline it supports and the footnote that states its test, and only the problem words go on."""
    def context(f):
        near = section_of(f, texts_by_id, sections)[:8]
        return "; ".join(f"({ROLE_NAMES.get(t.get('role'), '문구')}) {t['text'][:60]}" for t in near) or "(없음)"
    listing = "\n".join(f"[{f['id']}] 문구: {f['phrase']}\n줄 전체: {line_of(f, texts_by_id)}\n같은 구간: {context(f)}"
                        f"\n1차 메모: {f.get('reason', '')}" for f in findings)
    output = await provider.parse(instruction("claim_judgment"), ClaimJudgments, text=listing)
    return {j.id: j for j in output.judgments}


def span_box(span: str, block: dict | None):
    """Page-ratio box of the OCR words that spell the problem span, or None when no word matches."""
    target = squeeze(re.sub(r"[^0-9A-Za-z가-힣%.]", "", span)).lower()
    boxes = []
    for word in (block or {}).get("words", []):
        seen = re.sub(r"[^0-9A-Za-z가-힣%.]", "", word["text"]).lower()
        if not seen or not target:
            continue
        # A word counts when most of it is part of the span ('입증된' for the span '임상으로 입증').
        size = SequenceMatcher(None, seen, target).find_longest_match(0, len(seen), 0, len(target)).size
        if size >= max(1, round(min(len(seen), len(target)) * 0.6)) and (size >= 2 or len(seen) == 1):
            boxes.append(word["bbox"])
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def apply_judgment(finding: dict, judgment: Judgment | None, block: dict | None = None) -> dict:
    """The span marks the exact problem words for the red highlight; one the model did not copy is ignored."""
    if judgment is None:
        return finding
    updated = {**finding, "claimType": judgment.claimType, "reason": judgment.reason or finding.get("reason"),
               "purpose": judgment.purpose}
    at = finding["phrase"].find(judgment.span) if judgment.span else -1
    if at >= 0:
        start = finding.get("start", 0) + at
        updated["problemSpan"] = {"text": judgment.span, "start": start, "end": start + len(judgment.span),
                                  "bbox": span_box(judgment.span, block)}
    return updated


class Candidates:
    """Every hit a tool returned, keyed c1, c2... so the model and the verifier refer to the same text."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.keys: set = set()
        self.rule_meta = {"rules": [], "evidence": [], "checks": set(), "missing": []}
        self.rule_called = False

    def add(self, kind: str, hit: dict, rule: dict | None = None) -> str | None:
        key = (kind, hit["document_key"], hit.get("section"), hit.get("item"), hit.get("page"))
        if key in self.keys:
            return None
        self.keys.add(key)
        cid = f"c{len(self.items) + 1}"
        self.items[cid] = {"kind": kind, "hit": hit, "rule": rule}
        return cid

    def add_rules(self, result: dict) -> list[str]:
        self.rule_called = True
        ids = []
        for match in result.get("matches", []):
            self.rule_meta["rules"].append({"id": match["rule_id"], "label": match["label"], "matchedBy": match["matched_by"],
                                            "matchedText": match["matched_text"], "checks": match["checks"],
                                            "missing": match["missing"]})
            self.rule_meta["checks"] |= set(match["checks"])
            self.rule_meta["missing"] += match["missing"]
            self.rule_meta["evidence"] += [{**e, "ruleId": match["rule_id"]} for e in match["evidence_requirements"]]
            ids += [cid for hit in match["provisions"] if (cid := self.add("law", hit, match))]
        return ids

    def view(self, ids) -> list[dict]:
        return [{"id": cid, "source": self.items[cid]["hit"]["source"],
                 "article": " ".join(p for p in (self.items[cid]["hit"].get("section"), self.items[cid]["hit"].get("item")) if p),
                 "excerpt": self.items[cid]["hit"]["excerpt"][:400]} for cid in ids]


async def gather(provider, call, finding: dict, line: str, day: str, found: Candidates, rejected=()) -> list[dict]:
    """The tool loop for one claim. Returns its trace; candidates collect in `found`."""
    note = ""
    if rejected:
        note = "\n\n앞서 찾은 아래 후보는 이 문구에 적용되지 않는다고 판정됐다. 다른 쟁점이나 다른 표현으로 다시 찾아라:\n" + \
               "\n".join(f"- {c['source']} {c['article']}" for c in rejected)
    messages = [
        {"role": "system", "content": instruction("basis_tool_selection")},
        {"role": "user", "content": f"광고 문구: {finding['phrase']}\n문구가 있는 줄: {line}\n주장 유형: {finding.get('claimType')}\n"
                                    f"판단 이유: {finding.get('reason', '')}\n근거로 확인할 것: {finding.get('purpose', '')}{note}"},
    ]
    trace, calls, started = [], [], time.monotonic()

    async def run(name: str, args: dict) -> list[str]:
        if name == "lookup_claim_rules":
            result = await call("rag", "lookup_claim_rules", {"phrase": finding["phrase"],
                                "claim_type": args.get("claim_type") or finding.get("claimType"), "review_date": day})
            return found.add_rules(result)
        query = str(args.get("query") or finding["phrase"])[:300]
        if name == "search_regulations":
            result = await call("rag", "search_regulations", {"query_text": query, "top_k": 4, "review_date": day})
            return [cid for hit in result.get("hits", []) if (cid := found.add("law", hit))]
        if name == "search_precedents":
            result = await call("rag", "search_precedents", {"query_text": query, "top_k": 2})
            return [cid for hit in result.get("hits", []) if (cid := found.add("case", hit))]
        raise ValueError("UNKNOWN_TOOL")

    for _ in range(MAX_BASIS_CALLS + 1):
        if time.monotonic() - started > BASIS_LOOP_SECONDS:
            trace.append({"stop": "TIMEOUT"})
            break
        message = await provider.call_with_tools(messages, BASIS_TOOL_SPECS)
        tool_calls = message.tool_calls or []
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": message.content or "",
                         "tool_calls": [tc.model_dump() for tc in tool_calls]})
        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if len(calls) >= MAX_BASIS_CALLS or calls.count(name) >= MAX_SAME_BASIS_TOOL:
                output = {"skipped": "GUARDRAIL_LIMIT_REACHED"}
            else:
                calls.append(name)
                try:
                    ids = await run(name, args)
                    output = {"candidates": found.view(ids)}
                    trace.append({"tool": name, "arguments": args, "found": len(ids)})
                except Exception:
                    output = {"error": "TOOL_FAILED"}
                    trace.append({"tool": name, "arguments": args, "error": True})
            messages.append({"role": "tool", "tool_call_id": tool_call.id,
                             "content": json.dumps(output, ensure_ascii=False)[:4000]})
    return trace


async def verify(provider, finding: dict, line: str, found: Candidates, ids: list[str]) -> dict[str, BasisVerdict]:
    """Independent check: the verifier never sees the judgment, the purpose or the loop."""
    if not ids:
        return {}
    listing = "\n\n".join(
        f"[{cid}] {found.items[cid]['hit']['source']} "
        f"{' '.join(p for p in (found.items[cid]['hit'].get('section'), found.items[cid]['hit'].get('item')) if p)}\n"
        f"원문: {found.items[cid]['hit']['excerpt'][:CANDIDATE_TEXT]}" for cid in ids)
    output = await provider.parse(instruction("basis_verification"), BasisCheck,
                                  text=f"광고 문구: {finding['phrase']}\n문구가 있는 줄: {line}\n\n후보 조항:\n{listing}")
    verdicts = {}
    for verdict in output.verdicts:
        item = found.items.get(verdict.id)
        if verdict.id in ids and item and verdict.applies and squeeze(verdict.quote) \
                and squeeze(verdict.quote) in squeeze(item["hit"]["excerpt"]):
            verdicts[verdict.id] = verdict
    return verdicts


def verified_basis(found: Candidates, verdicts: dict[str, BasisVerdict]) -> tuple[list[dict], list[dict]]:
    basis, references, cited = [], [], set()
    for cid, verdict in verdicts.items():
        item = found.items[cid]
        # Two chunks of one article (or one guideline heading) are one citation on the card.
        cite = (item["hit"]["source"], item["hit"].get("section"), item["hit"].get("item"))
        if cite in cited:
            continue
        cited.add(cite)
        if item["kind"] == "case":
            hit = item["hit"]
            references.append({"id": f"{hit['document_key']}:{hit.get('page')}", "verified": True, "summary": verdict.why,
                               "quote": verdict.quote, "source": hit["source"], "page": hit.get("page"),
                               "sourceUrl": hit.get("source_url"), "similarity": hit.get("similarity")})
        elif len(basis) < MAX_VERIFIED_BASIS:
            basis.append({**to_basis(item["hit"], item["rule"]), "plain": verdict.why,
                          "verification": {"status": "verified", "quote": verdict.quote}})
    return basis, references


async def basis_for(provider, call, finding: dict, line: str, day: str) -> dict:
    found = Candidates()
    trace = await gather(provider, call, finding, line, day, found)
    first = list(found.items)
    verdicts = await verify(provider, finding, line, found, first)
    retried = False
    if not verdicts and first:
        retried = True
        trace += await gather(provider, call, finding, line, day, found, rejected=found.view(first))
        verdicts = await verify(provider, finding, line, found, [c for c in found.items if c not in first])
    if not found.rule_called:
        # Cross-checks need the rule table's required checks and evidence even when the model did not cite a rule.
        try:
            found.rule_called = True
            result = await call("rag", "lookup_claim_rules", {"phrase": finding["phrase"],
                                "claim_type": finding.get("claimType"), "review_date": day})
            for match in result.get("matches", []):
                found.rule_meta["checks"] |= set(match["checks"])
                found.rule_meta["evidence"] += [{**e, "ruleId": match["rule_id"]} for e in match["evidence_requirements"]]
        except Exception:
            pass
    basis, references = verified_basis(found, verdicts)
    return {"basis": basis, "references": references, "rules": found.rule_meta["rules"],
            "evidence": found.rule_meta["evidence"], "checks": sorted(found.rule_meta["checks"]),
            "missing": found.rule_meta["missing"],
            "trace": {"tools": trace, "candidates": len(found.items), "verified": len(verdicts), "retried": retried},
            "toolErrors": sum(1 for t in trace if t.get("error"))}


async def judged_basis(findings: list[dict], provider, *, texts: list[dict] | None = None,
                       section_texts: list[dict] | None = None, review_date: date | None = None, call=None) -> dict:
    """Same shape as regulations.legal_basis, plus "dismissed": candidates judged not a regulatory concern.

    texts are the claim copy the rule sweep reads; section_texts are all copy blocks, footnotes included,
    used as each claim's context."""
    day = (review_date or date.today()).isoformat()
    texts_by_id = {t["id"]: t for t in (section_texts or []) + (texts or [])}
    sections = {}
    for t in section_texts or texts or []:
        sections.setdefault((t.get("location") or {}).get("tileId"), []).append(t)

    async def run(session_call):
        swept = []
        if texts:
            try:
                swept = await sweep_findings(session_call, texts, findings)
            except Exception:
                swept = []
        candidates = findings + swept
        if not candidates:
            return [], [], [], []
        warnings = []
        try:
            judgments = await judge(provider, candidates, texts_by_id, sections)
        except Exception:
            judgments, warnings = {}, ["CLAIM_JUDGMENT_FAILED"]
        kept, dismissed, seen = [], [], {}
        for finding in candidates:
            judgment = judgments.get(finding["id"])
            if judgment is not None and not judgment.problem:
                dismissed.append({"id": finding["id"], "phrase": finding["phrase"], "textId": finding.get("textId"),
                                  "reason": judgment.reason, "source": finding.get("source", "model")})
            elif (key := squeeze(finding["phrase"])) in seen:
                # The same claim repeated in another section is one card; its other places are listed on it.
                seen[key].setdefault("alsoAt", []).append({"textId": finding.get("textId"), "location": finding.get("location")})
            else:
                card = apply_judgment(finding, judgment, texts_by_id.get(finding.get("textId")))
                # The section's test-condition footnotes travel with the claim for the evidence comparison.
                conditions = [t["text"] for t in section_of(finding, texts_by_id, sections) if t.get("role") == "condition"]
                if conditions:
                    card["conditions"] = conditions
                read = (texts_by_id.get(finding.get("textId")) or {}).get("readCheck") or {}
                if read.get("status") == "needs_review":
                    card["readCheck"] = read
                seen[key] = card
                kept.append(card)
        limit = asyncio.Semaphore(4)

        async def one(finding):
            async with limit:
                try:
                    return await basis_for(provider, session_call, finding, line_of(finding, texts_by_id), day)
                except Exception:
                    return None
        return kept, dismissed, await asyncio.gather(*(one(f) for f in kept)), warnings

    try:
        if call is None:
            async with McpTools(servers=("rag",)) as session_call:
                kept, dismissed, results, warnings = await run(session_call)
        else:
            kept, dismissed, results, warnings = await run(call)
    except Exception:
        return {"findings": findings, "dismissed": [], "status": "error", "snapshot": [],
                "warnings": ["LEGAL_SEARCH_UNAVAILABLE"]}
    if not kept:
        return {"findings": [], "dismissed": dismissed, "status": "skipped", "snapshot": [], "warnings": warnings}
    enriched, snapshot = [], {}
    for finding, result in zip(kept, results):
        if result is None:
            warnings.append("LEGAL_BASIS_PARTIAL")
            enriched.append({**finding, "basis": [], "baseBasis": [], "legalSearch": "failed"})
            continue
        for item in result["basis"]:
            snapshot[(item["sourceName"], item["version"])] = {"name": item["sourceName"], "version": item["version"],
                                                               "revisedAt": item.get("revisedAt")}
        if result["toolErrors"]:
            warnings.append("LEGAL_SEARCH_PARTIAL")
        enriched.append({**finding, "basis": result["basis"], "baseBasis": result["basis"], "rules": result["rules"],
                         "evidenceRequirements": result["evidence"], "requiredChecks": result["checks"],
                         "references": result["references"], "basisTrace": result["trace"],
                         "legalSearch": "verified" if result["basis"] else "not_verified",
                         "title": "검증된 근거와 대조할 문구" if result["basis"] else "근거를 확인하지 못한 문구"})
    if all(r is None for r in results):
        return {"findings": enriched, "dismissed": dismissed, "status": "error", "snapshot": [],
                "warnings": list(dict.fromkeys(warnings + ["LEGAL_SEARCH_FAILED"]))}
    return {"findings": enriched, "dismissed": dismissed, "status": "connected", "snapshot": list(snapshot.values()),
            "warnings": list(dict.fromkeys(warnings))}
