import psycopg


async def database_probe(dsn: str, *, require_vector: bool = False) -> dict:
    try:
        async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=3) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SET statement_timeout = '3000ms'")
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
                if require_vector:
                    await cursor.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        return {"status": "down", "reason": "VECTOR_EXTENSION_MISSING"}
                    return {"status": "ok", "vector_version": row[0]}
        return {"status": "ok"}
    except psycopg.Error:
        return {"status": "down", "reason": "DATABASE_UNAVAILABLE"}
