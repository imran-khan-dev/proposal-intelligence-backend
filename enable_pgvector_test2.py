import asyncpg
import asyncio

async def test():
    try:
        conn = await asyncpg.connect('postgresql://postgres:45.86.86.86@localhost:5432/proposal_intelligence_test')
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print('Enabled pgvector extension')
        await conn.close()
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(test())