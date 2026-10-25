"""Two independent analysis paths. Raw assets live in closures, not checkpoints."""
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from typing import Literal, TypedDict

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from capstone_shared.documents import prepare_document, Tile
from capstone_shared.settings import internal_token, required_env
from .observability import traced_client
from .checks import cross_checks, patent_record_cards, resolved_by_evidence
from .ad_reading import read_tile
from .evidence import summarize, match as match_evidence
from .instructions import instruction
from .regulations import legal_basis, search_basis
from .judged_basis import judged_basis
from .overreach import review_overreach
from .ingredient_presence import rescue_ingredient_claims, settle_ingredient_claims
from .vague_effect import dismiss_vague_effects, set_aside_low_risk
from .severity_rules import raise_for_lapsed_patents
from .tools import McpTools
from .guardrails import mask_tool_result
from .patent_certificate import group_patent_certificate, patent_numbers
from .context_budget import prepare_parse, prepare_tool_messages, split_parse_text, TOOL_OUTPUT_BUDGET
from .review_models import (
    RunRequest, TextAssessment, ResultTile, RewriteSuggestion, RewriteVerification,
)

# 사람 재시도를 포함한 전체 리라이트 시도 한도. app.violations.rewrite_attempt_count가
# 이 값에 도달하면 더 이상 자동 리라이트를 제안하지 않고 rewritten_manual을 요구한다.
# 한도 초과 재시도 요청의 403 처리는 결정 API(Backend) 쪽에 있고 여기서는 다루지 않는다.
REWRITE_RETRY_LIMIT = 3

LEGAL_NOTICE = {
    "connected": "법령 근거는 검색으로 찾은 조문 중 검증을 통과한 것입니다. 적법 여부는 판정하지 않았으며 담당자가 원문으로 확인해야 합니다.",
    "not_indexed": "법령 자료가 적재되지 않아 근거를 검색하지 못했습니다. 적법 여부를 판단하지 않았습니다.",
    "error": "법령 검색 서비스에 연결하지 못해 근거를 확인하지 못했습니다. 적법 여부를 판단하지 않았습니다.",
    "skipped": "확인할 문구가 없어 법령 검색을 하지 않았습니다. 적합 판정을 의미하지 않습니다.",
    "not_connected": "법령 검색을 실행하지 않았습니다. 적법 여부를 판단하지 않았습니다.",
}


class RewriteDraft(BaseModel):
    text: str


class SeverityAssessment(BaseModel):
    severity: Literal["low", "medium", "high"]
    reason: str


class ReviewState(TypedDict, total=False):
    reviewId: str
    tenantId: str
    versionNo: int
    product: dict
    texts: list
    findings: list
    textStatus: str
    textWarnings: list
    sectionTitles: dict
    result: dict
    retrievalStatus: str
    standardsSnapshot: list
    legalWarnings: list
    dismissedFindings: list
    decision: str
    reportReady: bool


logger = logging.getLogger(__name__)

# Provider failures worth another model: rate limits, 5xx and connection or timeout errors.
# Bad requests and refusals would fail the same way on any model, so they are raised as is.
RETRYABLE_MODEL_ERRORS = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)


REASONING_HEADROOM = 12000


class Provider:
    """Model calls for every node. The SDK retries 429, 5xx and connection errors with backoff
    (MODEL_MAX_RETRIES, default 2). If the call still fails and FALLBACK_TEXT_MODEL or FALLBACK_VISION_MODEL
    is set, it is tried once; otherwise the error reaches the node, which records partial or failed.
    Both structured output (parse) and the business tool loop (call_with_tools) go through _with_fallback."""

    def __init__(self):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        retries = int(os.getenv("MODEL_MAX_RETRIES", "2"))
        self.client = AsyncOpenAI(api_key=key, timeout=50, max_retries=retries) if key else None
        self.limit = asyncio.Semaphore(int(os.getenv("MODEL_CONCURRENCY", "4")))
        if self.client:
            self.client = traced_client(self.client)

    async def parse(self, prompt, schema, *, image=None, text="", reasoning=None):
        """reasoning ("low", "medium") lets the model think before answering; its thinking counts against the
        output cap, so the cap grows by REASONING_HEADROOM. Used where a judgment, not a transcription, is asked."""
        if not self.client:
            raise RuntimeError("OPENAI_NOT_CONFIGURED")
        bounded, output_cap, compacted = prepare_parse(
            prompt, text or "제공한 광고 영역을 분석하세요.", schema, image=bool(image))
        images = image if isinstance(image, list) else ([image] if image else [])
        if compacted:
            logger.info("model input compacted for %s", schema.__name__)
        content = [{"type":"input_text", "text":bounded}]
        content += [{"type":"input_image", "image_url":url, "detail":"high"} for url in images]
        request = {"input":[{"role":"system", "content":prompt}, {"role":"user", "content":content}],
                   "text_format":schema, "store":False, "max_output_tokens":output_cap}
        if reasoning:
            request["reasoning"] = {"effort": reasoning}
            request["max_output_tokens"] = output_cap + REASONING_HEADROOM
        async with self.limit:
            response = await self._with_fallback("VISION" if image else "TEXT",
                                                 lambda model: self.client.responses.parse(model=model, **request))
        if response.output_parsed is None:
            raise RuntimeError("MODEL_REFUSED_OR_INCOMPLETE")
        return response.output_parsed

    async def call_with_tools(self, messages, tools):
        """business 노드의 도구선택 loop 전용. structured output(parse)과 달리 tool_calls가 필요해 Chat
        Completions를 쓴다."""
        if not self.client:
            raise RuntimeError("OPENAI_NOT_CONFIGURED")
        bounded, compacted = prepare_tool_messages(messages, tools)
        if compacted:
            logger.info("business tool conversation compacted")
        async with self.limit:
            response = await self._with_fallback("TEXT", lambda model: self.client.chat.completions.create(
                model=model, messages=bounded, tools=tools, tool_choice="auto", max_completion_tokens=TOOL_OUTPUT_BUDGET))
        return response.choices[0].message

    async def _with_fallback(self, kind, call):
        """Call with the {kind}_MODEL; after SDK retries, try FALLBACK_{kind}_MODEL once for retryable errors."""
        primary = os.getenv(f"{kind}_MODEL", "gpt-5.4-mini")
        fallback = os.getenv(f"FALLBACK_{kind}_MODEL", "").strip()
        try:
            return await call(primary)
        except RETRYABLE_MODEL_ERRORS as error:
            if not fallback or fallback == primary:
                raise
            logger.warning("model %s failed after retries (%s); trying fallback %s",
                           primary, type(error).__name__, fallback)
            return await call(fallback)


def basis_context(basis):
    """Statute text handed to the rewrite prompt. Excerpts are retrieved text, not instructions."""
    if not basis:
        return "(검색된 조문 없음)"
    lines = []
    for item in basis:
        lines.append(f"- {item['sourceName']} {item['article']}: {item['original'][:600]}")
        annexes = [related for related in item.get("related", [])
                   if related.get("relation") == "annex" and
                   (related.get("article") or "").startswith("[별표 2] >")]
        if annexes:
            lines.append("  연결된 실증 대상과 조건 (비고가 인용한 항목이며, 표현별 적용 여부 확인 필요):")
            lines.extend(f"  - {related['article']}: {related['original'][:450]}" for related in annexes[:8])
    return "\n".join(lines)


class RewriteRejected(Exception):
    """No draft passed the checks. The message is the last reason, shown to the reviewer."""


async def rewrite_phrase(provider, request, attempts=2, call=None):
    """사람이 요청한 문구 하나의 대체 문안. 자동 제안과 같은 규칙 검사와 모델 검증을 통과해야 내보낸다."""
    if call is None:
        async with McpTools(servers=("rag",)) as session_call:
            return await rewrite_phrase(provider, request, attempts, session_call)
    lookup = await search_basis([{"phrase":request.phrase,"claimType":request.claimType}], call=call)
    basis = lookup["findings"][0].get("basis", []) if lookup["status"] == "connected" and lookup["findings"] else []
    feedback, previous = [], list(request.previousRewrites)
    for _ in range(attempts):
        text, passed, reasoning = await draft_and_verify_rewrite(
            provider, request.phrase, request.reason, retry_note=request.retryNote, basis=basis,
            feedback=feedback, call=call, previous=previous)
        if passed:
            return text
        feedback.append(f"{text}: {reasoning}")
        previous.append(text)
    raise RewriteRejected(reasoning)


async def emit(request, kind, **data):
    """Send one progress event to Spring, which stores step marks and relays every event over SSE.
    Kinds: step_done(step), tool_started(tool), tool_finished(tool, ok). Failures are ignored because
    the persisted result stays the source of truth and the browser falls back to polling."""
    if request is None:
        return
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            await client.post(required_env("BACKEND_BASE_URL")+f"/internal/reviews/{request.reviewId}/events",
                headers={"Authorization":f"Bearer {internal_token()}","X-Run-Token":request.runToken},
                json={"versionNo":request.versionNo,"type":kind,**data})
    except Exception:
        pass


async def progress(request, step):
    await emit(request, "step_done", step=step)


async def prepare_assets(request: RunRequest):
    native, tiles = [], []
    async with httpx.AsyncClient(timeout=20) as client:
        for asset in request.assets:
            if asset.role != "advertisement":
                continue
            response = await client.get(required_env("BACKEND_BASE_URL")+f"/internal/reviews/{request.reviewId}/assets/{asset.id}",
                params={"versionNo":request.versionNo},
                headers={"Authorization":f"Bearer {internal_token()}","X-Run-Token":request.runToken})
            response.raise_for_status()
            blocks, images = await asyncio.to_thread(prepare_document,response.content,asset.model_dump())
            native.extend(blocks); tiles.extend(images)
            if len(tiles)>40:
                raise ValueError("TOTAL_TILE_LIMIT_40")
    await progress(request,0)
    return native, tiles


def deduplicate(blocks):
    result = []
    for block in blocks:
        text = re.sub(r"\s+", "", block["text"])
        if not text:
            continue
        loc = block["location"]
        duplicate = False
        for existing in result:
            old = existing["location"]
            if old["fileId"]!=loc["fileId"] or old["page"]!=loc["page"]:
                continue
            a, b = old.get("bbox") or old.get("region"), loc.get("bbox") or loc.get("region")
            overlap = a and b and min(a[3],b[3])>max(a[1],b[1])
            if overlap and re.sub(r"\s+","",existing["text"])==text:
                duplicate=True; break
        if not duplicate:
            result.append(block)
    return result


def tile_manifest(tiles):
    """result.tiles. Coordinates only; the UI crops the preserved original, so tile pixels are never stored."""
    return [ResultTile(id=t.id, index=i, fileId=t.asset_id, fileName=t.name, page=t.page,
                       region=t.location()["region"]).model_dump() for i, t in enumerate(tiles)]


def assign_tile(location, manifest):
    """One representative tile per finding, so a claim in the 240px overlap is not listed twice.
    The tileId from OCR is not enough: deduplicate keeps whichever overlapping tile read the line first,
    and PDF native text has no tile at all. Prefer tiles that fully contain the box, then the tile
    whose center is nearest. Region-only locations (VLM OCR, visual findings) keep their own tile."""
    if not location:
        return None
    box = location.get("bbox")
    if box is None:
        tile_id = location.get("tileId")
        return tile_id if any(t["id"] == tile_id for t in manifest) else None
    center = (box[1]+box[3])/2
    ranked = []
    for t in manifest:
        if t["fileId"] != location.get("fileId") or t["page"] != location.get("page"):
            continue
        top, bottom = t["region"][1], t["region"][3]
        overlap = min(bottom, box[3])-max(top, box[1])
        if overlap <= 0:
            continue
        contained = top <= box[1]+1e-9 and box[3] <= bottom+1e-9
        ranked.append((not contained, -overlap if not contained else 0, abs((top+bottom)/2-center), t["index"], t["id"]))
    return min(ranked)[-1] if ranked else None


def compact(text):
    return re.sub(r"\s+","",text)


async def extract_text(native, tiles, provider):
    """Ad copy of the document: native PDF text as is, and every image section read by read_tile (Tesseract boxes
    for position, one vision call for the wording, product-surface text left out). Returns (blocks, warnings)."""
    blocks, warnings = list(native), []
    for tile, (found, notes) in zip(tiles, await asyncio.gather(*(read_tile(provider, tile) for tile in tiles))):
        blocks.extend(found)
        warnings += notes
        if not found and provider.client:
            warnings.append(f"{tile.id}:NO_AD_COPY")
    return group_patent_certificate(deduplicate(blocks)), warnings


async def mcp_call(name, arguments, request=None):
    headers={}
    if request:
        headers={"X-Run-Token":request.runToken,"X-Review-Id":request.reviewId,
                 "X-Tenant-Id":request.tenantId,"X-Version-No":str(request.versionNo)}
    transport=StreamableHttpTransport(required_env("BUSINESS_MCP_URL"),headers=headers,auth=internal_token())
    async with asyncio.timeout(40):
        async with Client(transport) as client:
            response = await client.call_tool(name,arguments)
            if response.is_error:
                raise RuntimeError("BUSINESS_TOOL_FAILED")
            return mask_tool_result(name, response.data)


def compact_len(text):
    return len(compact(text or ""))


async def rewrite_guard(phrase, text, call=None):
    """Checks that do not need a model. Returns the failure reason, or None when the draft may go on to
    the model check. The rule check reuses the high-signal patterns of the claim rule table (the same
    ones the sweep uses): a rewrite must not keep the original's risky expression or add a new one."""
    if not compact(text):
        return "수정안이 비어 있습니다."
    if compact(text) == compact(phrase):
        return "원문과 같은 문구입니다."
    if compact_len(phrase) >= 4 and compact_len(text) < compact_len(phrase)*0.5:
        return "문구가 잘려 광고 문장으로 읽히지 않습니다."
    # 원문이 "~다."로 끝나는 문장이면 수정안도 문장으로 끝나야 한다. 뒤를 잘라 낸 수정안을 막는다.
    if re.search(r"다[.!]?$", compact(phrase)) and not re.search(r"다[.!]?$", compact(text)):
        return "원문은 완결된 문장인데 수정안은 문장이 중간에 끊겼습니다."
    if call is None:
        return None
    try:
        scan = await call("rag", "scan_claim_lines", {"lines": [phrase, text]})
    except Exception:
        return None  # The model check still runs; a tool outage must not block every rewrite.
    by_line = {m["index"]: m for m in scan.get("matches", [])}
    before, after = by_line.get(0, {}), by_line.get(1, {})
    labels = dict(zip(after.get("rules", []), after.get("labels", after.get("rules", []))))
    matched = dict(zip(after.get("rules", []), after.get("matched", [])))
    kept = [r for r in after.get("rules", []) if r in before.get("rules", [])]
    added = [r for r in after.get("rules", []) if r not in before.get("rules", [])]
    if added:
        return f"원문에 없던 {labels[added[0]]}('{matched[added[0]]}')을 새로 넣었습니다."
    if kept:
        return f"{labels[kept[0]]}('{matched[kept[0]]}')이 그대로 남아 있습니다."
    return None


async def verify_rewrite(provider, text, reason, basis=None, phrase="", call=None):
    """수정 문구가 원래 지적된 문제를 해소했는지 판단한다. 문구를 새로 쓰지 않는다.
    Agent 제안과 사람이 직접 쓴 rewritten_manual 문구를 같은 기준으로 본다 — app.violations.
    final_text_validated를 true로 만들기 전 필수 관문이고, 자가검증 loop도 이 함수를 그대로 쓴다.
    규칙 검사(rewrite_guard)를 먼저 하고, 통과한 문구만 조문 후보와 함께 모델이 본다."""
    if phrase:
        failure = await rewrite_guard(phrase, text, call)
        if failure:
            return RewriteVerification(passed=False, reasoning=failure)
    return await provider.parse(
        instruction("rewrite_verification"),
        RewriteVerification,
        text=f"원문: {phrase or '(제공되지 않음)'}\n수정 문구: {text}\n지적된 문제: {reason}\n"
             f"관련 조문 후보:\n{basis_context(basis)}")


async def draft_rewrite(provider, phrase, reason, basis=None, feedback=(), retry_note=None, previous=()):
    feedback_text = "\n".join(f"- {f}" for f in feedback) or "(없음)"
    previous_text = "\n".join(f"- {p}" for p in previous) or "(없음)"
    draft = await provider.parse(
        instruction("rewrite_draft"),
        RewriteDraft,
        text=f"원문: {phrase}\n지적된 문제: {reason}\n관련 조문 후보:\n{basis_context(basis)}\n"
             f"이전 수정안:\n{previous_text}\n이전 수정안이 실패한 이유:\n{feedback_text}"
             + (f"\n검토자 요청: {retry_note}" if retry_note else ""))
    return draft.text.strip()


async def draft_and_verify_rewrite(provider, phrase, reason, retry_note=None, basis=None, feedback=(),
                                   call=None, candidate=None, previous=()):
    """자가검증 loop의 한 사이클: 생성(또는 1차 추출 때 받은 후보)과 재검증은 분리된 호출이다.
    Returns (text, passed, reasoning)."""
    text = candidate or await draft_rewrite(provider, phrase, reason, basis, feedback, retry_note, previous)
    verification = await verify_rewrite(provider, text, reason, basis=basis, phrase=phrase, call=call)
    return text, verification.passed, verification.reasoning


async def assess_severity(provider, finding, evidence_text):
    """조문 근거와 제출 증빙을 같은 호출에서 함께 보고 검토 우선순위 등급만 매긴다. 적법/위반을
    확정하지 않는다 — 별도 규칙 엔진으로 분리하면 광고 문구를 읽은 모델과 다른 결론을 낼 수 있어
    같은 모델 호출 안에서 처리한다."""
    if not provider.client:
        return finding["severity"], None
    try:
        output = await provider.parse(
            instruction("severity_assessment"),
            SeverityAssessment,
            text=f"광고 문구: {finding['phrase']}\n확인 사유: {finding['reason']}\n주장 유형: {finding['claimType']}\n"
                 f"관련 조문 후보:\n{basis_context(finding.get('basis'))}\n"
                 f"제출된 증빙 서류 원문(없으면 미제출):\n{(evidence_text or '(없음)')[:4000]}")
        return output.severity, output.reason
    except Exception:
        return finding["severity"], None


async def build_violations(findings, provider, call=None):
    """② 위반 탐지는 text_path가 이미 했다(findings). 여기서는 ⑤ 자가검증 리라이트 loop만
    violation 단위로 돌린다. 1차 추출 때 받은 후보부터 검증하고, 떨어지면 실패 이유를 알려 주고
    다시 쓴다. 한도 안에 통과한 문구가 없으면 제안을 비워 둔다 — 검증 안 된 문구를 수정안으로
    내보내지 않는다. ⑥ 사람의 최종 결정은 Backend 결정 API가 저장한다."""
    async def one(finding):
        violation = {
            "id": finding["id"],
            "text_excerpt": finding["phrase"],
            "severity": finding["severity"],
            "rewrite_suggestion": None,
            "rewrite_attempt_count": 0,
            "verified": False,
            "rewrite_note": None,
            "rewrites": [],
            "final_resolution": None,
            "final_text": None,
            "final_text_validated": False,
            "decided_by": None,
            "decided_at": None,
        }
        seeds = [r for r in finding.get("rewrites") or [] if r and r.strip()][:1]
        feedback, tried = [], []
        while provider.client and violation["rewrite_attempt_count"] < REWRITE_RETRY_LIMIT:
            violation["rewrite_attempt_count"] += 1
            try:
                text, passed, reasoning = await draft_and_verify_rewrite(
                    provider, finding["phrase"], finding["reason"], basis=finding.get("basis"), feedback=feedback,
                    call=call, candidate=seeds.pop(0) if seeds else None, previous=tried)
            except Exception:
                continue  # Budget consumed either way; a provider failure isn't a recorded attempt.
            tried.append(text)
            violation["rewrites"].append({"attempt_no": violation["rewrite_attempt_count"], "text": text,
                                          "requested_by_human": False, "retry_note": None,
                                          "passed": passed, "reasoning": reasoning})
            if passed:
                violation.update(rewrite_suggestion=text, verified=True, rewrite_note=None)
                break
            feedback.append(f"{text}: {reasoning}")
            violation["rewrite_note"] = reasoning
        return violation

    # Findings are independent, so their loops run concurrently. Provider.limit still caps model calls.
    return list(await asyncio.gather(*(one(finding) for finding in findings)))


# business 노드의 ②③④ 도구선택 loop guardrail. Agent Graph 상세 설계 문서의 예시값을 그대로 따른다.
MAX_TOOL_CALLS_TOTAL = 8
MAX_SAME_TOOL_CALLS = 2
LOOP_TIMEOUT_SECONDS = 60

BUSINESS_TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "structure_evidence",
        "description": "제품에 등록된 증빙서류(인증서·전성분표·시험자료 등)의 읽기 결과를 가져온다. 광고 주장이 수치나 "
                        "시험 결과, 성분을 말할 때 호출한다. 증빙은 제품 등록 때 이미 읽혀 있어 여기서 다시 읽지 않는다.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "verify_functional_approval",
        "description": "식약처 기능성화장품 보고품목정보로 제품의 기능성 심사·보고 여부를 조회한다. 광고가 "
                        "미백, 주름개선, 자외선차단 등 기능성 효능을 주장할 때 호출한다.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "verify_patent",
        "description": "KIPRIS Plus로 특허번호가 실제 등록된 특허인지 확인한다. 광고나 증빙에 특허번호 또는 "
                        "'특허' 주장이 있을 때만 호출한다.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def tool_succeeded(output):
    """For the progress event only: an adapter error inside a result (api_error, not configured) is a failure.
    Adapter results always carry an "error" key, so only a non-empty value counts. A lookup that found
    nothing (not_found) or needs confirmation is a successful call, not a failure."""
    items = output if isinstance(output, list) else [output]
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).lower()
        if item.get("error") or status in ("api_error", "api_not_configured", "error"):
            return False
    return True


def evidence_total(request) -> int:
    """Evidence this review can rest on: files sent with the review plus the product's own evidence that Spring already
    holds a read result for. Counting only the files sent made a product whose evidence was registered once look as if
    nothing had been submitted (the model skipped the tool, and every figure claim came back 'no evidence')."""
    if request is None:
        return 0
    sent = len([a for a in request.assets if a.role == "evidence"])
    return sent + int(getattr(request, "productEvidenceCount", 0) or 0)


async def select_business_tools(provider, state, request):
    """business 노드의 도구선택 loop. LLM은 findings를 보고 structure_evidence/verify_functional_approval/
    verify_patent 중 관련 있는 것만 고른다 — 매번 조건문으로 다 지나가는 게 아니라 실제로 선택이다.
    각 tool의 실제 호출 인자(증빙 파일, 심사번호, 특허번호)는 LLM이 지어내지 않고 코드가 추출한 값만
    쓴다. LLM의 역할은 '이 tool이 지금 필요한가'라는 판단뿐이고, 값 추출은 하지 않는다.
    Guardrail: 총 호출 MAX_TOOL_CALLS_TOTAL회, 동일 tool 연속 MAX_SAME_TOOL_CALLS회, loop 전체
    LOOP_TIMEOUT_SECONDS초. 결과가 불충분해도(예: ambiguous) 같은 tool을 한도 내에서 다시 부를 수
    있다(근거 재검토) — tool 결과를 본 다음 턴에서 LLM이 직접 판단한다.
    """
    evidence = {"claimTypes": [], "requirements": [], "flags": [], "status": "not_submitted"}
    public = {"status": "skipped", "source": "식약처 공공데이터", "checkedAt": None}
    patents, warnings, trace = [], [], []
    documents = [a.model_dump() for a in request.assets if a.role == "evidence"] if request else []
    product = state["product"]
    report = product.get("functionalReportNo", "")
    text = "\n".join(t["text"] for t in state["texts"])
    candidate_numbers = patent_numbers(text)[:10]
    available = {
        # 증빙은 제품에 등록되어 있고, 접수 때 Spring이 읽기가 끝난 증빙의 수를 알려 준다.
        "structure_evidence": bool(getattr(request, "productEvidenceCount", 0)),
        "verify_functional_approval": bool(product.get("name")),
        "verify_patent": bool(candidate_numbers) or any(f.get("claimType") == "patent" for f in state["findings"]),
    }
    # Why a tool did not run, so the review can say "not needed" apart from "nothing to check with".
    unavailable_reason = {"structure_evidence": "NO_EVIDENCE", "verify_functional_approval": "NO_PRODUCT_NAME",
                          "verify_patent": "NO_PATENT_CLAIM"}

    def not_available():
        return [{"tool": name, "called": False, "reason": unavailable_reason[name]}
                for name, usable in available.items() if not usable]

    if not any(available.values()):
        return evidence, public, patents, warnings, not_available()

    async def invoke(name):
        await emit(request, "tool_started", tool=name)
        output = await run_tool(name)
        await emit(request, "tool_finished", tool=name, ok=tool_succeeded(output))
        return output

    async def run_tool(name):
        nonlocal evidence, public, patents
        try:
            if name == "structure_evidence":
                evidence = await mcp_call("structure_evidence",
                    {"tenant_id": state["tenantId"], "review_id": state["reviewId"], "documents": documents}, request)
                return evidence
            if name == "verify_functional_approval":
                public = await mcp_call("verify_functional_approval",
                    {"functional_approval_no": report, "product_name": product.get("name", "")})
                return public
            if name == "verify_patent":
                if not candidate_numbers:
                    patents = []
                    return {"status": "no_patent_number_detected"}
                patents = await mcp_call("verify_patent", {"patent_numbers": candidate_numbers})
                return patents
            return {"error": "UNKNOWN_TOOL"}
        except Exception:
            warnings.append(f"{name.upper()}_TOOL_UNAVAILABLE")
            return {"error": f"{name.upper()}_TOOL_UNAVAILABLE"}

    if not provider.client:
        # 모델 키가 없으면 LLM에게 선택을 물을 수 없다. business-mcp 호출은 OpenAI와 무관한 별개
        # 경로라, LLM 유무와 상관없이 구조적으로 관련 있는 도구는 그대로 실행한다(NFR-16).
        for name, usable in available.items():
            if usable:
                await invoke(name)
                trace.append({"tool": name, "called": True})
        return evidence, public, patents, warnings, trace + not_available()

    findings_summary = "\n".join(f"- \"{f['phrase']}\" (claimType={f.get('claimType')}): {f['reason']}"
                                 for f in state["findings"]) or "(없음)"
    messages = [
        {"role": "system", "content": instruction("business_tool_selection")},
        {"role": "user", "content":
            f"광고에서 추출한 확인 대상 문구:\n{findings_summary}\n\n"
            f"제품명: {product.get('name') or '(없음)'}\n"
            f"기능성 심사·보고번호 입력값: {report or '(없음)'}\n"
            f"제출된 증빙서류: {evidence_total(request)}건 (제품에 등록된 증빙 {int(getattr(request, 'productEvidenceCount', 0) or 0)}건 포함)\n"
            f"광고 문구에서 찾은 특허번호 형식 문자열: {len(candidate_numbers)}건"},
    ]
    called, limited = [], set()
    ended, model_note = "MODEL_NOT_NEEDED", ""
    started = time.monotonic()
    for _ in range(MAX_TOOL_CALLS_TOTAL + 1):
        if time.monotonic() - started > LOOP_TIMEOUT_SECONDS:
            warnings.append("TOOL_LOOP_TIMEOUT")
            ended = "LOOP_TIMEOUT"
            break
        try:
            message = await provider.call_with_tools(messages, BUSINESS_TOOL_SPECS)
        except Exception:
            warnings.append("TOOL_SELECTION_FAILED")
            # The model could not choose (provider error). Run the structurally relevant tools that have not
            # run yet, as the no-key path does, so a model outage does not silently skip the checks (NFR-16).
            for name, usable in available.items():
                if usable and name not in called:
                    called.append(name)
                    await invoke(name)
                    trace.append({"tool": name, "called": True, "fallback": True})
            break
        tool_calls = message.tool_calls or []
        if not tool_calls:
            # The selection prompt asks for a short reason when no more tools are needed; keep it for the review.
            model_note = (message.content or "").strip()[:300]
            break
        messages.append({"role": "assistant", "content": message.content or "",
                         "tool_calls": [tc.model_dump() for tc in tool_calls]})
        for tool_call in tool_calls:
            name = tool_call.function.name
            same = sum(1 for c in called if c == name)
            if len(called) >= MAX_TOOL_CALLS_TOTAL or same >= MAX_SAME_TOOL_CALLS or not available.get(name):
                output = {"skipped": "GUARDRAIL_LIMIT_REACHED" if available.get(name) else "TOOL_NOT_APPLICABLE"}
                if available.get(name):
                    limited.add(name)
            else:
                called.append(name)
                output = await invoke(name)
                trace.append({"tool": name, "called": True})
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(output, default=str)[:4000]})
    # Registered evidence is what the reviewer submitted to answer substantiation claims, and it is also where an
    # ingredient claim is confirmed (ingredient_presence): when a card needs it,
    # the comparison is not left to the model's choice (it skipped the tool in 2 of 4 runs with evidence present).
    if available["structure_evidence"] and "structure_evidence" not in called and \
            any("substantiation" in f.get("requiredChecks", []) or f.get("claimType") == "ingredient" for f in state["findings"]):
        called.append("structure_evidence")
        await invoke("structure_evidence")
        trace.append({"tool": "structure_evidence", "called": True, "required": True})
    for name, usable in available.items():
        if usable and name not in called:
            # Available but not run: the model judged it unnecessary for these cards (no card asked for it, or the
            # evidence rule above would have run it), or a guardrail or the time limit stopped the loop first.
            reason = "GUARDRAIL_LIMIT" if name in limited else ended
            entry = {"tool": name, "called": False, "reason": reason}
            if reason == "MODEL_NOT_NEEDED" and model_note:
                entry["note"] = model_note
            trace.append(entry)
    return evidence, public, patents, warnings, trace + not_available()


def merge_nested(findings):
    """A fragment the model also extracted as part of a longer phrase on the same line is one claim to
    review, not two cards. Keep the longer phrase."""
    kept = []
    for finding in findings:
        inner = compact(finding["phrase"])
        if any(other is not finding and other.get("textId") == finding.get("textId")
               and other["start"] <= finding["start"] and finding["end"] <= other["end"]
               and inner != compact(other["phrase"]) and inner in compact(other["phrase"]) for other in findings):
            continue
        if any(k.get("textId") == finding.get("textId") and k["start"] == finding["start"]
               and k["end"] == finding["end"] and compact(k["phrase"]) == inner for k in kept):
            continue
        kept.append(finding)
    return kept


async def review_text(provider, texts, manifest=()):
    """Text review: claim candidates the model extracts from the ad copy, chunk by chunk within the token budget,
    kept only when grounded in one text block. Returns (findings, failure warnings).
    The harness evaluation (scripts/evaluate_harness.py) runs this same step, so a prompt, limit or model
    change is measured on the code the review uses."""
    findings, failures = [], []
    ad = "\n".join(t["text"] for t in texts)
    spans, offset = [], 0
    for source in texts:
        spans.append((offset, offset + len(source["text"]), source))
        offset += len(source["text"]) + 1
    try:
        chunks = split_parse_text(instruction("text_review"), ad, TextAssessment)
    except Exception:
        chunks = []
        failures.append("TEXT_REVIEW_BUDGET_FAILED")
    for chunk_index, (chunk_start, chunk) in enumerate(chunks):
        try:
            output = await provider.parse(instruction("text_review"), TextAssessment, text=chunk)
        except Exception:
            failures.append(f"TEXT_REVIEW_CHUNK_FAILED:{chunk_index + 1}")
            continue
        phrase_counts = {}
        for issue in output.findings:
            if not issue.phrase:
                continue
            positions = [chunk_start + match.start() for match in re.finditer(re.escape(issue.phrase), chunk)]
            valid = [(source, position - left) for position in positions
                     for left, right, source in spans if left <= position and position + len(issue.phrase) <= right]
            occurrence = phrase_counts.get(issue.phrase, 0)
            phrase_counts[issue.phrase] = occurrence + 1
            match = valid[occurrence] if occurrence < len(valid) else None
            if match is None:
                continue  # Reject ungrounded or cross-block model spans.
            source, start = match
            if source.get("documentContext") == "patent_certificate" and not patent_numbers(issue.phrase):
                continue  # Filing/registration dates describe the certificate; they are not efficacy claims.
            findings.append({"id":f"text-{len(findings)}","phrase":issue.phrase,"start":start,"end":start+len(issue.phrase),
                "location":source["location"],"textId":source["id"],"tileId":assign_tile(source["location"],manifest),
                "severity":"low","baseSeverity":"low",
                "claimType":issue.claimType,"reason":issue.reason,"title":"근거 대조 전 확인할 문구","parseRisk":issue.risk,
                "basis":[],"baseBasis":[],"confidence":None,"rewrites":issue.rewrites,"references":[],"category":"확인할 광고 주장",
                "legalStatus":"not_evaluated","escalation":None})
    return merge_nested(findings), failures


def build_graph(native, tiles, provider, checkpointer, request=None):
    # tileId is set where each finding is created. Later nodes update these same finding dicts
    # (severity, escalation), so the result must not be built from copies.
    manifest = tile_manifest(tiles)

    async def text_path(state):
        texts, warnings = await extract_text(native,tiles,provider)
        findings = []
        status = "complete" if not warnings else "partial"
        # Test-condition footnotes are context for the evidence comparison, not claims to review.
        claim_texts = [t for t in texts if t.get("role") != "condition"]
        titles = {}
        for t in claim_texts:
            tile_id = (t.get("location") or {}).get("tileId")
            if tile_id and tile_id not in titles and t.get("role") == "claim":
                titles[tile_id] = t["text"][:24]
        if not provider.client:
            status="not_configured"; warnings.append("TEXT_MODEL_NOT_CONFIGURED")
        elif claim_texts:
            findings, failures = await review_text(provider, claim_texts, manifest)
            if failures:
                status="partial"; warnings += failures
        else:
            status="needs_input";warnings.append("NO_READABLE_TEXT")
        await progress(request,1)
        return {"texts":texts,"findings":findings,"textStatus":status,"textWarnings":warnings,"sectionTitles":titles}

    async def regulations(state):
        await emit(request, "tool_started", tool="legal_basis")
        claim_texts = [t for t in state["texts"] if t.get("role") != "condition"]
        # With a model: judge, pick basis tools per claim, verify apart (judged_basis.py). Without one: rules and search.
        # Cards the first pass rated low go no further unless they name something checkable (vague_effect.py).
        candidates, low_at_parse = set_aside_low_risk(state["findings"], state["texts"])
        if provider.client:
            # The overreach review releases cards nothing in the ad supports before any basis is fetched for them.
            lookup = await judged_basis(candidates, provider, texts=claim_texts, section_texts=state["texts"],
                                        review=lambda cards: review_overreach(provider, cards, state["texts"]))
        else:
            lookup = await legal_basis(candidates, texts=claim_texts)
        await emit(request, "tool_finished", tool="legal_basis", ok=lookup["status"] in ("connected", "skipped"))
        # A plain ingredient claim is not settled by judgment but by the attached material (checks node).
        lookup["findings"], lookup["dismissed"] = rescue_ingredient_claims(lookup["findings"], lookup.get("dismissed", []),
                                                                           state["texts"])
        # Generic care copy that names no specific effect has nothing for a reviewer to check (vague_effect.py).
        kept, vague = dismiss_vague_effects(lookup["findings"], state["texts"])
        lookup["findings"], lookup["dismissed"] = kept, lookup["dismissed"] + vague + low_at_parse
        # 규칙 sweep이 추가한 문구는 text_path를 거치지 않아 대표 구간이 비어 있다. 여기서 같은 방식으로 정한다.
        for finding in lookup["findings"]:
            if not finding.get("tileId"):
                finding["tileId"] = assign_tile(finding.get("location"), manifest)
        return {"findings": lookup["findings"], "retrievalStatus": lookup["status"],
                "standardsSnapshot": lookup["snapshot"], "legalWarnings": lookup["warnings"],
                "dismissedFindings": lookup.get("dismissed", [])}

    async def business(state):
        evidence,public,patents,warnings,tool_trace=await select_business_tools(provider,state,request)
        await progress(request,3)
        titles=state.get("sectionTitles",{})
        return {"result":{"texts":state["texts"],"tiles":[{**t,"title":titles.get(t["id"])} for t in manifest],
            "findings":state["findings"],"dismissedFindings":state.get("dismissedFindings",[]),"visualFindings":[],
            "evidence":evidence,"publicData":public,"patents":patents,"usage":{"credits":0,"billingEnabled":False},
            "standardsSnapshot":state.get("standardsSnapshot",[]),"retrievalStatus":state.get("retrievalStatus","not_connected"),
            "legalStatus":"not_evaluated","businessToolSelection":tool_trace,
            # Photos and graphs are not reviewed (this service reads ad copy only). The backend still expects a visual
            # status and an empty list of visual findings: complete and empty, or not_configured without a model key.
            "pipelineStatus":{"text":state["textStatus"],"visual":"complete" if provider.client else "not_configured"},
            "warnings":list(dict.fromkeys(state["textWarnings"]+warnings+state.get("legalWarnings",[])
                +[LEGAL_NOTICE.get(state.get("retrievalStatus"),LEGAL_NOTICE["not_connected"])])),
            "analyzedAt":datetime.now(timezone.utc).isoformat()}}

    async def checks(state):
        result = state["result"]
        # Evidence uploaded with this review and evidence registered on the product (read before submission).
        evidence_count = evidence_total(request)
        evidence, matches, warnings = dict(result["evidence"]), {}, []
        if evidence.get("documents") and provider.client:
            # 증빙마다 시험 요약을 만들고, 실증이 필요한 주장과 한 번에 대조한다(evidence.py).
            try:
                evidence["summaries"] = await summarize(provider, evidence["documents"])
                matches = await match_evidence(provider, result["findings"], evidence["summaries"],
                                               [t["text"] for t in state["texts"]])
                evidence["claimMatches"] = [{"findingId": k, **v} for k, v in matches.items()]
            except Exception:
                warnings.append("EVIDENCE_MATCH_FAILED")
        cards = result["findings"] + patent_record_cards(result["findings"], state["texts"], result.get("patents"))
        out = await cross_checks(cards, product=state["product"], evidence=evidence,
                                 evidence_count=evidence_count, ad_text="\n".join(t["text"] for t in state["texts"]),
                                 provider=provider, review_date=datetime.now(timezone.utc).date().isoformat(),
                                 matches=matches, patents=result.get("patents"))
        out["warnings"] = out["warnings"] + warnings
        result = {**result, "evidence": evidence}
        kept, resolved = resolved_by_evidence(out["findings"])
        # A card that only says what the product contains is settled when the attached material says it too.
        kept, settled = settle_ingredient_claims(kept, state["texts"], evidence.get("documents", []))
        resolved = resolved + settled
        return {"result": {**result, "findings": kept, "checksSummary": out["summary"],
                           "dismissedFindings": result.get("dismissedFindings", []) + resolved,
                           "functionalReports": out.get("functionalReports"),
                           "warnings": list(dict.fromkeys(result["warnings"] + out["warnings"]))}}

    async def violations(state):
        current_findings = [dict(finding) for finding in state["result"]["findings"]]
        evidence_text = state["result"].get("evidence", {}).get("text", "")
        assessments = await asyncio.gather(
            *(assess_severity(provider, finding, evidence_text) for finding in current_findings))
        for finding, (severity, reason) in zip(current_findings, assessments):
            if severity != finding["severity"]:
                finding["escalation"] = {"from": finding["severity"], "reason": reason or "조문 근거 및 증빙 대조 결과"}
            finding["severity"] = severity
        # A floor from public records (severity_rules.py): claims in an ad that pictures a lapsed patent are high.
        current_findings = raise_for_lapsed_patents(current_findings, state["result"].get("texts", []),
                                                    state["result"].get("patents"))
        try:
            async with McpTools(servers=("rag",)) as call:
                detected = await build_violations(current_findings, provider, call)
        except Exception:
            detected = await build_violations(current_findings, provider)
        # Backend stores finding.rewrites as the violation's suggestion. Only a rewrite that passed the
        # rule and model checks goes there; otherwise the card says why no suggestion was made.
        by_id = {v["id"]: v for v in detected}
        findings = []
        for f in current_findings:
            v = by_id.get(f["id"], {})
            if v.get("verified"):
                findings.append({**f, "rewrites": [v["rewrite_suggestion"]], "rewriteVerified": True, "rewriteNote": None})
            else:
                findings.append({**f, "rewrites": [], "rewriteVerified": False, "rewriteNote": v.get("rewrite_note")})
        return {"result": {**state["result"], "findings": findings, "violations": detected}}

    def approval(state):
        decision=interrupt({"reviewId":state["reviewId"],"versionNo":state["versionNo"],"gate":"human_review"})
        return {"decision":decision}

    def timed(name, node):
        async def run(state):
            started = time.monotonic()
            out = await node(state)
            logger.info("node %s took %.1fs", name, time.monotonic() - started)
            return out
        return run

    graph=StateGraph(ReviewState)
    graph.add_node("text_path",timed("text_path",text_path))
    graph.add_node("regulations",timed("regulations",regulations))
    graph.add_node("business",timed("business",business));graph.add_node("cross_checks",timed("cross_checks",checks));graph.add_node("violations",timed("violations",violations))
    graph.add_node("approval",approval)
    graph.add_node("report",lambda state:{"reportReady":True})
    graph.add_edge(START,"text_path");graph.add_edge("text_path","regulations");graph.add_edge("regulations","business")
    graph.add_edge("business","cross_checks");graph.add_edge("cross_checks","violations")
    graph.add_edge("violations","approval")
    graph.add_edge("approval","report");graph.add_edge("report",END)
    return graph.compile(checkpointer=checkpointer)
