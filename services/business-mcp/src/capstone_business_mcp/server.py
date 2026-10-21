import os

import httpx
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.dependencies import get_http_headers
from starlette.responses import JSONResponse

from capstone_shared.settings import internal_token, required_env
from .review_tools import functional_report, functional_reports_by_name, mfds_list, patents, structure


def create_server() -> FastMCP:
    verifier = StaticTokenVerifier(
        tokens={internal_token(): {"client_id": "capstone-agent", "scopes": ["tools:read"]}},
        required_scopes=["tools:read"],
    )
    mcp = FastMCP("business-mcp", auth=verifier)

    @mcp.custom_route("/health/live", methods=["GET"])
    async def live(request) -> JSONResponse:
        return JSONResponse({"service": "business-mcp", "status": "ok"})

    @mcp.tool(annotations={"readOnlyHint": True})
    async def get_environment_info() -> dict:
        """Check the authenticated Spring Boot internal API. Contains no product data."""
        async with httpx.AsyncClient(timeout=4) as client:
            response = await client.get(
                required_env("BACKEND_BASE_URL") + "/internal/bootstrap",
                headers={"Authorization": f"Bearer {internal_token()}"},
            )
            response.raise_for_status()
            return response.json()

    @mcp.tool(annotations={"readOnlyHint":True})
    async def structure_evidence(tenant_id: str, review_id: str, documents: list[dict]) -> dict:
        """MVP extension: read run-scoped evidence via Backend. Do not expose execution credentials to a model."""
        headers={k.lower():v for k,v in get_http_headers().items()}
        if headers.get("x-review-id")!=review_id or headers.get("x-tenant-id")!=tenant_id:
            raise ValueError("EXECUTION_SCOPE_REQUIRED")
        return await structure(tenant_id,review_id,documents,int(headers["x-version-no"]),headers["x-run-token"])

    @mcp.tool(annotations={"readOnlyHint":True})
    async def verify_functional_approval(functional_approval_no: str, product_name: str) -> dict:
        """Lookup official product facts. Configuration failure is never a missing product."""
        return await functional_report(functional_approval_no,product_name)

    @mcp.tool(annotations={"readOnlyHint":True})
    async def verify_patent(patent_numbers: list[str]) -> list[dict]:
        """Query KIPRIS exact registration candidates; do not infer claim coverage."""
        return await patents(patent_numbers)

    @mcp.tool(annotations={"readOnlyHint":True})
    async def find_functional_reports(product_name: str, functional_approval_no: str | None = None) -> dict:
        """Find MFDS functional cosmetic reports by product name (report number optional).
        Returns exact name matches with reported functions (미백, 주름개선, 자외선차단) and efficacy text,
        plus partial-name candidates. Use when an ad claims a functional effect or 식약처 심사/보고.
        not_found means no report was located by this name, not that the product is unreported."""
        return await functional_reports_by_name(product_name,functional_approval_no)

    @mcp.tool(annotations={"readOnlyHint":True})
    async def verify_gmp_company(company_name: str) -> dict:
        """Check whether a manufacturer appears in the MFDS cosmetics GMP-compliant company list.
        Use when an ad or evidence claims GMP certification. Not a product quality or efficacy finding."""
        return await mfds_list("gmp",company_name)

    @mcp.tool(annotations={"readOnlyHint":True})
    async def lookup_ingredient_regulation(ingredient_name: str) -> dict:
        """Look up an ingredient's standard name and the countries that ban or restrict it (MFDS 화장품 규제정보).
        Use when an ad names an ingredient or claims it is safe or free of restricted ingredients."""
        return await mfds_list("regulation",ingredient_name)

    return mcp


def main() -> None:
    create_server().run(
        transport="http",
        host=os.getenv("SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVICE_PORT", "8002")),
    )
