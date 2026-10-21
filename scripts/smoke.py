#!/usr/bin/env python3
"""Exercise the running development stack without modifying corpus data."""
import argparse
import asyncio
import json
from uuid import uuid4

import httpx
import pika
import psycopg
from fastmcp import Client

from capstone_shared.settings import internal_token, required_env


def check_queue() -> None:
    """Verify AMQP delivery on a connection-scoped synthetic queue, not a business worker."""
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=required_env("RABBITMQ_HOST"),
        port=int(required_env("RABBITMQ_PORT")),
        virtual_host=required_env("RABBITMQ_VHOST"),
        credentials=pika.PlainCredentials(required_env("RABBITMQ_USERNAME"), required_env("RABBITMQ_PASSWORD")),
        socket_timeout=5, blocked_connection_timeout=10, heartbeat=30,
    ))
    try:
        channel = connection.channel()
        queue = channel.queue_declare(queue="", exclusive=True, auto_delete=True).method.queue
        channel.confirm_delivery()
        payload = json.dumps({"kind": "foundation-smoke", "message_id": str(uuid4())}).encode()
        channel.basic_publish(exchange="", routing_key=queue, body=payload, mandatory=True,
                              properties=pika.BasicProperties(content_type="application/json"))
        delivery, _, received = channel.basic_get(queue, auto_ack=False)
        assert delivery is not None and received == payload
        channel.basic_nack(delivery.delivery_tag, requeue=True)
        delivery, _, received = channel.basic_get(queue, auto_ack=False)
        assert delivery is not None and delivery.redelivered and received == payload
        channel.basic_ack(delivery.delivery_tag)
        assert channel.queue_declare(queue=queue, passive=True).method.message_count == 0
    finally:
        connection.close()
    print("PASS: RabbitMQ AMQP publish confirmation, receive, nack/requeue and manual ack on a synthetic queue.")


async def main(include_queue: bool = False) -> None:
    token = internal_token()
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=15) as http:
        response = await http.get(required_env("BACKEND_BASE_URL") + "/api/dev/status")
        response.raise_for_status()
        data = response.json()
        assert data["dependencies"]["application_database"]["status"] == "ok"
        assert data["dependencies"]["agent"]["status"] == "ok"
        dependencies = data["dependencies"]["agent_environment"]["dependencies"]
        assert all(item["status"] == "ok" for item in dependencies.values())
        denied = await http.get(required_env("BACKEND_BASE_URL") + "/internal/bootstrap")
        assert denied.status_code == 401, "Spring internal API must require a token"
        ready = await http.get(required_env("AGENT_BASE_URL") + "/health/ready")
        assert ready.status_code == 200
        invalid_run = await http.post(
            required_env("AGENT_BASE_URL") + "/internal/runs", headers=headers
        )
        assert invalid_run.status_code == 422, "Agent execution now requires a scoped MVP request"
        for name in ["BUSINESS_MCP_URL", "RAG_MCP_URL"]:
            denied = await http.post(required_env(name), json={})
            assert denied.status_code == 401, f"{name} must require a token"

    async with Client(required_env("BUSINESS_MCP_URL"), auth=token) as client:
        info = await client.call_tool("get_environment_info", {})
        assert info.structured_content["status"] == "ok"
    async with Client(required_env("RAG_MCP_URL"), auth=token) as client:
        names = {tool.name for tool in await client.list_tools()}
        assert {"search_regulations", "search_precedents"} <= names
        assert not names & {"search_rules", "search_cases", "get_evidence"}, "placeholder tools must stay removed"
        result = await client.call_tool("search_regulations", {"query_text": "합성 테스트 문구", "top_k": 1})
        assert result.structured_content["status"] in {"FOUND", "NO_RELEVANT_EVIDENCE", "CORPUS_NOT_INDEXED"}
        corpus = json.loads((await client.read_resource("regulation://corpus"))[0].text)
        assert isinstance(corpus["documents"], list)
    async with await psycopg.AsyncConnection.connect(
        required_env("RAG_DATABASE_URL"), connect_timeout=3
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            assert await cursor.fetchone(), "pgvector extension must be installed"
            await cursor.execute("SELECT count(*) FROM knowledge.document_versions")
            await cursor.fetchone()
            await cursor.execute("SELECT collection_key, source_verified FROM knowledge.document_versions LIMIT 0")
            await cursor.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'knowledge.chunks'::regclass AND attname = 'embedding'"
            )
            assert await cursor.fetchone() == ("vector(1024)",)
            zero_vector = "[" + ",".join(["0"] * 1024) + "]"
            unit_vector = "[1," + ",".join(["0"] * 1023) + "]"
            await cursor.execute("SELECT %s::vector <-> %s::vector", (zero_vector, unit_vector))
            assert await cursor.fetchone() == (1.0,)
            await cursor.execute(
                "SELECT has_table_privilege(current_user, 'knowledge.chunks', 'INSERT'), "
                "has_table_privilege(current_user, 'knowledge.chunks', 'SELECT'), "
                "has_schema_privilege(current_user, 'app', 'USAGE')"
            )
            insert, select, app_access = await cursor.fetchone()
            assert (insert, select, app_access) == (False, True, False)
    if include_queue:
        await asyncio.to_thread(check_queue)
    print("PASS: HTTP, MCP authentication, cross-service calls, pgvector and read-only DB access.")
    print("No document was inserted and no advertisement was evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-queue", action="store_true", help="Require the optional RabbitMQ profile")
    asyncio.run(main(parser.parse_args().include_queue))
