import asyncpg
import asyncio

async def test():
    try:
        conn = await asyncpg.connect('postgresql://postgres:45.86.86.86@localhost:5432/rfp_ai')
        version = await conn.fetchval('SELECT version()')
        print(f'Connected: {version[:80]}...')
        
        # Check if pgvector extension exists
        ext = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        print(f'pgvector extension: {ext}')
        
        await conn.close()
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(test())