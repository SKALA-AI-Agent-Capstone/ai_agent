"""Evidence reading for product registration: the Agent fetches the stored file from Spring and reads it.

Spring owns the file and the result; the Agent only reads. Reading runs in a worker thread because PaddleOCR is
blocking, and it is bounded so a long scan returns what it could read instead of timing out the request.
"""
import asyncio

import httpx
from pydantic import BaseModel, ConfigDict, Field

from capstone_shared.evidence_parse import parse_evidence
from capstone_shared.settings import internal_token, required_env

from .evidence_vlm import make_crosscheck

PARSE_TIMEOUT_SECONDS = 280
OCR_BUDGET_SECONDS = 170
MAX_OCR_PAGES = 40


class EvidenceParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidenceId: str = Field(min_length=1, max_length=64)
    productName: str | None = Field(default=None, max_length=200)


async def fetch_evidence(evidence_id: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(required_env("BACKEND_BASE_URL") + f"/internal/evidence/{evidence_id}/file",
                                    headers={"Authorization": f"Bearer {internal_token()}"})
        response.raise_for_status()
        return response.content


async def parse_stored_evidence(request: EvidenceParseRequest, provider=None) -> dict:
    raw = await fetch_evidence(request.evidenceId)
    crosscheck = make_crosscheck(provider, asyncio.get_running_loop())
    async with asyncio.timeout(PARSE_TIMEOUT_SECONDS):
        return await asyncio.to_thread(parse_evidence, raw, request.productName,
                                       ocr_budget_seconds=OCR_BUDGET_SECONDS, max_ocr_pages=MAX_OCR_PAGES,
                                       crosscheck=crosscheck)
