"""Resume a review thread after the human decision, at most once per thread.

The Backend queue may deliver the same resume twice (redelivery after a lost ACK, stale-job recovery).
A PostgreSQL advisory lock on the thread id makes concurrent calls run one after the other, and the
second one finds the checkpoint already finished and returns alreadyResumed instead of running the
nodes after approval again.
"""
from contextlib import asynccontextmanager

import psycopg
from fastapi import HTTPException
from langgraph.types import Command


@asynccontextmanager
async def thread_lock(database_url: str, thread_id: str):
    """Session advisory lock on its own connection, released when the block ends or the connection drops."""
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as connection:
        await connection.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (thread_id,))
        try:
            yield
        finally:
            await connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (thread_id,))


async def resume_thread(graph, config: dict, state: str) -> dict:
    """Call while holding the thread lock. Missing checkpoint is a permanent 404 for the Backend queue."""
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="CHECKPOINT_NOT_FOUND")
    if not snapshot.next:
        return {"reportReady": True, "alreadyResumed": True}
    output = await graph.ainvoke(Command(resume=state), config)
    return {"reportReady": output.get("reportReady", False)}
