import asyncio
import os
import httpx

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastmcp import Client

from capstone_shared import CONTRACT_VERSION
from capstone_shared.probes import database_probe
from capstone_shared.settings import (
    internal_token,
    required_env,
    valid_internal_authorization,
)
from .observability import run_config, trace_headers, trace_id
from .review_models import RunRequest, ResumeRequest, RewriteRequest, VerifyRewriteRequest
from .review_engine import Provider, prepare_assets, build_graph, progress, rewrite_phrase, verify_rewrite, RewriteRejected
from .regulations import source_document
from .tools import McpTools
from .resume import resume_thread, thread_lock
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


async def require_internal(authorization: str | None = Header(default=None)) -> None:
    if not valid_internal_authorization(authorization):
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")


async def mcp_probe(url: str) -> dict:
    try:
        async with asyncio.timeout(5):
            async with Client(url, auth=internal_token()) as client:
                tools = await client.list_tools()
        return {"status": "ok", "tools": [tool.name for tool in tools]}
    except Exception:
        return {"status": "down", "reason": "MCP_UNAVAILABLE"}


def create_app() -> FastAPI:
    internal_token()
    app = FastAPI(title="Cosmetics review Agent local MVP", version="0.2.0-mvp")

    @app.get("/health/live")
    async def live() -> dict:
        return {"service": "agent", "status": "ok", "mode": "development-scaffold"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        db = await database_probe(required_env("AGENT_DATABASE_URL"))
        return JSONResponse(
            {"service": "agent", "status": db["status"], "database": db},
            status_code=200 if db["status"] == "ok" else 503,
        )

    @app.get("/internal/environment", dependencies=[Depends(require_internal)])
    async def environment() -> dict:
        database, business, rag = await asyncio.gather(
            database_probe(required_env("AGENT_DATABASE_URL")),
            mcp_probe(required_env("BUSINESS_MCP_URL")),
            mcp_probe(required_env("RAG_MCP_URL")),
        )
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": "development-scaffold",
            "dependencies": {
                "agent_database": database,
                "business_mcp": business,
                "rag_mcp": rag,
            },
            "implemented": ["health", "authenticated-mcp-discovery", "ad-parsing", "independent-text-and-photo-paths", "langgraph", "postgres-hitl"],
            "planned": ["rag-retrieval", "evidence-claim-matching", "queue-worker", "billing"],
        }

    @app.post("/internal/runs", dependencies=[Depends(require_internal)])
    async def create_run(request: RunRequest) -> dict:
        config = run_config(f"{request.tenantId}:{request.reviewId}:v{request.versionNo}", run_name="review",
                            review_id=request.reviewId, tenant_id=request.tenantId, version_no=request.versionNo)
        # Only a started graph has a LangSmith run. A failure while reading the files has no trace to point at.
        started = False
        try:
            async with asyncio.timeout(270):
                native, tiles = await prepare_assets(request)
                async with AsyncPostgresSaver.from_conn_string(required_env("AGENT_DATABASE_URL")) as saver:
                    await saver.setup()
                    graph = build_graph(native, tiles, Provider(), saver, request)
                    started = True
                    output = await graph.ainvoke({"reviewId":request.reviewId,"tenantId":request.tenantId,
                        "versionNo":request.versionNo,"product":request.product}, config)
                    await progress(request,4)
                    result = output["result"]
                    if trace_id(config):
                        result = {**result, "traceId": trace_id(config)}
                    return {"result":result,"hitlStatus":"pending","contract_version":"0.2.0-mvp"}
        except (ValueError, httpx.HTTPStatusError) as error:
            raise HTTPException(status_code=422,detail="INPUT_UNREADABLE_OR_EXPIRED",
                                headers=trace_headers(config) if started else None) from None
        except Exception:
            # Spring retries a 5xx. The header lets the failed job point at the errored LangSmith run.
            raise HTTPException(status_code=500,detail="RUN_FAILED",headers=trace_headers(config) if started else None) from None

    @app.post("/internal/runs/resume", dependencies=[Depends(require_internal)])
    async def resume(request: ResumeRequest) -> dict:
        thread=f"{request.tenantId}:{request.reviewId}:v{request.versionNo}"
        async with thread_lock(required_env("AGENT_DATABASE_URL"), thread):
            async with AsyncPostgresSaver.from_conn_string(required_env("AGENT_DATABASE_URL")) as saver:
                graph=build_graph([],[],Provider(),saver)
                config=run_config(thread, run_name="review-resume",
                                  review_id=request.reviewId, tenant_id=request.tenantId, version_no=request.versionNo)
                return await resume_thread(graph, config, request.state)

    @app.post("/internal/rewrites", dependencies=[Depends(require_internal)])
    async def rewrite(request: RewriteRequest) -> dict:
        provider=Provider()
        if not provider.client:
            raise HTTPException(status_code=503,detail="TEXT_MODEL_NOT_CONFIGURED")
        try:
            async with asyncio.timeout(90):
                text=await rewrite_phrase(provider,request)
        except RewriteRejected as rejected:
            # 검증을 통과한 문안이 없다. 마지막 실패 이유를 그대로 돌려 검토자가 방향을 고쳐 다시 요청하게 한다.
            raise HTTPException(status_code=422,detail=str(rejected) or "REWRITE_NOT_VERIFIED") from None
        except Exception:
            raise HTTPException(status_code=502,detail="REWRITE_PROVIDER_FAILED") from None
        if not text:
            raise HTTPException(status_code=502,detail="REWRITE_PROVIDER_FAILED")
        return {"text":text}

    @app.post("/internal/rewrites/verify", dependencies=[Depends(require_internal)])
    async def verify(request: VerifyRewriteRequest) -> dict:
        provider=Provider()
        if not provider.client:
            raise HTTPException(status_code=503,detail="TEXT_MODEL_NOT_CONFIGURED")
        try:
            async with asyncio.timeout(60):
                async with McpTools(servers=("rag",)) as call:
                    result=await verify_rewrite(provider,request.text,request.reason,phrase=request.phrase,call=call)
        except Exception:
            raise HTTPException(status_code=502,detail="REWRITE_VERIFICATION_FAILED") from None
        return {"passed":result.passed,"reasoning":result.reasoning}

    @app.get("/internal/regulations/document", dependencies=[Depends(require_internal)])
    async def regulation_document(key: str, version: str) -> dict:
        tools = McpTools(servers=("rag",))
        try:
            async with tools:
                document = await source_document(tools.read, key, version)
        except Exception:
            raise HTTPException(status_code=502,detail="REGULATION_SOURCE_UNAVAILABLE") from None
        if document is None:
            raise HTTPException(status_code=404,detail="REGULATION_DOCUMENT_NOT_FOUND")
        return document

    @app.post("/internal/runs/delete", dependencies=[Depends(require_internal)])
    async def delete_runs(request: ResumeRequest) -> dict:
        async with AsyncPostgresSaver.from_conn_string(required_env("AGENT_DATABASE_URL")) as saver:
            await saver.setup()
            for no in range(1,request.versionNo+1):
                await saver.adelete_thread(f"{request.tenantId}:{request.reviewId}:v{no}")
        return {"deleted":True}

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVICE_PORT", "8001")),
    )
