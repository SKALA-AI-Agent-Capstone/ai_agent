"""Submitted evidence against the ad's claims.

1. summarize  one structured summary per readable document: what kind of test, what it measured, when, on whom,
              by which institution, with what result. Read from the document text only.
2. match      every claim that needs substantiation is compared with those summaries in one call:
              supports            a document measured the same effect and its conditions fit the claim
              different_test      the documents measured something else (an irritation test for a pore claim)
              condition_mismatch  same kind of test, but period, subjects or numbers differ from the ad
              not_covered         no document speaks to the claim

The verdict tells the reviewer which document to ask for. It is not a legal finding, and a missing match never
means the claim is false; it means the submitted material does not show it.
"""
import asyncio
from difflib import SequenceMatcher
import re
from typing import Literal

from pydantic import BaseModel

from .instructions import instruction

VERDICTS = ("supports", "different_test", "condition_mismatch", "not_covered")
# Lines in the ad that state test conditions (footnotes such as "*연구기간 ... | 20명 대상 | 2주 후 비교").
CONDITION_LINE = re.compile(r"시험|연구|기간|대상|기관|기준|응답|명\s*대상|\d+\s*(주|일)\s*후|n\s*=")


class EvidenceSummary(BaseModel):
    kind: Literal["efficacy_test", "safety_test", "certificate", "other"]
    title: str
    purpose: str
    measures: list[str]
    period: str
    subjects: str
    institution: str
    result: str


class ClaimMatch(BaseModel):
    findingId: str
    documentId: str | None
    verdict: Literal["supports", "different_test", "condition_mismatch", "not_covered"]
    reason: str
    needed: str


class ClaimMatches(BaseModel):
    items: list[ClaimMatch]


async def summarize(provider, documents: list[dict]) -> list[dict]:
    async def one(doc):
        try:
            summary = await provider.parse(
                instruction("evidence_summary"),
                EvidenceSummary, text=f"문서 이름: {doc.get('name')}\n{doc.get('text', '')[:15000]}")
            return {"documentId": doc["id"], "name": doc.get("name"), "docType": doc.get("docType", ""),
                    **summary.model_dump()}
        except Exception:
            return None
    found = await asyncio.gather(*(one(d) for d in documents if d.get("text", "").strip()))
    return [s for s in found if s]


INSTITUTION_CLAIM = re.compile(r"기관|연구소|센터|연구원")
NAME_NOISE = re.compile(r"[\s*:：()㈜]|주식회사|연구기관|시험기관|\(주\)")


def same_institution(phrase: str, institution: str) -> bool:
    """Ad text comes from OCR, so '(주)프로봄' for '㈜프로브엠' is the same institution, not a mismatch."""
    a, b = NAME_NOISE.sub("", phrase), NAME_NOISE.sub("", institution)
    return bool(a and b) and SequenceMatcher(None, a, b).ratio() >= 0.75


def condition_lines(ad_lines: list[str]) -> list[str]:
    return [line for line in ad_lines if CONDITION_LINE.search(line)][:40]


async def match(provider, findings: list[dict], summaries: list[dict], ad_lines: list[str]) -> dict[str, dict]:
    """findingId -> {verdict, documentId, documentName, reason, needed}. Empty when there is nothing to compare."""
    targets = [f for f in findings if "substantiation" in f.get("requiredChecks", [])]
    if not targets or not summaries:
        return {}
    docs = "\n".join(
        f"- {s['documentId']} ({s['name']}): 종류 {s['kind']}, 제목 {s['title']}, 목적 {s['purpose']}, "
        f"측정 {', '.join(s['measures'])}, 기간 {s['period']}, 대상 {s['subjects']}, 기관 {s['institution']}, 결과 {s['result']}"
        for s in summaries)
    claims = "\n".join(f"- {f['id']}: {f['phrase']}" for f in targets)
    notes = "\n".join(f"- {line}" for line in condition_lines(ad_lines)) or "(없음)"
    output = await provider.parse(
        instruction("evidence_match"),
        ClaimMatches, text=f"제출 증빙 요약:\n{docs}\n\n광고에 적힌 시험 조건:\n{notes}\n\n실증이 필요한 주장:\n{claims}")
    names = {s["documentId"]: s["name"] for s in summaries}
    phrases = {f["id"]: f["phrase"] for f in targets}
    result = {}
    for m in output.items:
        if m.findingId not in phrases:
            continue
        entry = {"verdict": m.verdict, "documentId": m.documentId, "documentName": names.get(m.documentId),
                 "reason": m.reason, "needed": m.needed}
        phrase = phrases[m.findingId]
        if m.verdict == "condition_mismatch" and INSTITUTION_CLAIM.search(phrase):
            doc = next((s for s in summaries if s["institution"] and same_institution(phrase, s["institution"])), None)
            if doc:
                entry = {"verdict": "supports", "documentId": doc["documentId"], "documentName": doc["name"], "needed": "",
                         "reason": f"광고의 기관 표기는 글자 인식 차이로 보이며 문서의 {doc['institution']}와 같은 기관입니다."}
        result[m.findingId] = entry
    return result
