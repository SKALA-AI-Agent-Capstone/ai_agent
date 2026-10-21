"""Guardrails: personal data in tool results is masked before it reaches a model or the stored result.

mcp_call applies mask_tool_result to every business-mcp result, so the business tool loop never shows the
model unmasked evidence.

Only uploaded evidence (test reports, certificates) is expected to carry it, as contact names and numbers.
Advertising text is not masked: a claim must stay verbatim to be grounded in the source. Company identifiers
(business registration numbers, call-center numbers) and document numbers (patents, report numbers, dates)
are left as they are because the cross-checks match on them.
"""
import re

PATTERNS = (
    (re.compile(r"(?<!\d)\d{6}\s?-\s?[1-8]\d{6}(?!\d)"), "[주민등록번호]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[이메일]"),
    (re.compile(r"(?<!\d)01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"), "[휴대전화]"),
    (re.compile(r"(?<!\d)0(?:2|[3-6][1-5]|70)[-.)\s]\s?\d{3,4}[-.\s]\d{4}(?!\d)"), "[전화번호]"),
)


def mask_personal_data(text: str) -> tuple[str, int]:
    """Return the masked text and how many values were replaced."""
    count = 0
    for pattern, label in PATTERNS:
        text, replaced = pattern.subn(label, text)
        count += replaced
    return text, count


def mask_tool_result(tool: str, data):
    """Tool result boundary for business-mcp calls. structure_evidence carries the text of uploaded
    evidence twice: the combined text and each document's own text (documents[].text, read by the evidence
    summary). Both are masked here, before the business tool loop or the summary shows them to a model and
    before they are stored. Other business tools return public registry records and pass through unchanged."""
    if tool != "structure_evidence" or not isinstance(data, dict):
        return data
    masked, total = dict(data), 0
    if isinstance(data.get("text"), str):
        masked["text"], count = mask_personal_data(data["text"])
        total += count
    if isinstance(data.get("documents"), list):
        documents = []
        for document in data["documents"]:
            if isinstance(document, dict) and isinstance(document.get("text"), str):
                text, count = mask_personal_data(document["text"])
                document = {**document, "text": text}
                total += count
            documents.append(document)
        masked["documents"] = documents
    return {**masked, "maskedPersonalData": total} if total else data
