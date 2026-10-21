"""One MCP session per server for a graph node, so a node can call many tools without reconnecting."""
import asyncio
import json
from contextlib import AsyncExitStack

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from capstone_shared.settings import internal_token, required_env

SERVERS = {"rag": "RAG_MCP_URL", "business": "BUSINESS_MCP_URL"}


class ToolError(RuntimeError):
    pass


class McpTools:
    """async with McpTools() as call: await call("rag", "lookup_claim_rules", {...})

    Returns the tool's structured content. Raises ToolError when the tool reports an error.
    """

    def __init__(self, servers=("rag", "business"), timeout: float = 180):
        self.servers, self.timeout = servers, timeout
        self.clients: dict[str, Client] = {}
        self.stack = AsyncExitStack()

    async def __aenter__(self):
        for name in self.servers:
            transport = StreamableHttpTransport(required_env(SERVERS[name]), auth=internal_token())
            self.clients[name] = await self.stack.enter_async_context(Client(transport))
        return self.call

    async def __aexit__(self, *exc):
        await self.stack.aclose()

    async def call(self, server: str, tool: str, arguments: dict):
        async with asyncio.timeout(60):
            response = await self.clients[server].call_tool(tool, arguments, raise_on_error=False)
        if response.is_error:
            raise ToolError(f"{server}.{tool} failed")
        content = response.structured_content
        if isinstance(content, dict) and set(content) == {"result"}:
            return content["result"]
        return content if content is not None else response.data

    async def read(self, server: str, uri: str) -> dict:
        """Read a JSON resource. The server's ResourceError (unknown id, database down) propagates."""
        async with asyncio.timeout(30):
            contents = await self.clients[server].read_resource(uri)
        return json.loads(contents[0].text)
