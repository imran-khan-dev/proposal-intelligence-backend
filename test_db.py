import asyncpg
import asyncio

async def test_connection():
    try:
        # Try connecting to the test database
        conn = await asyncpg.connect('postgresql://postgres:45.86.86.86@localhost:5432/proposal_intelligence_test')
        print('Connected to test database successfully')
        # Check if pgvector extension is available
        result = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        print(f'pgvector extension available: {result}')
        # Check tables
        tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        print(f'Tables: {[t["table_name"] for t in tables]}')
        await conn.close()
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(test_connection())