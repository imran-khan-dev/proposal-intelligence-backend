import asyncpg
import asyncio

async def test():
    try:
        conn = await asyncpg.connect('postgresql://postgres:45.86.86.86@localhost:5432/rfp_ai_test')
        
        tables = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        print("Tables:")
        for t in tables:
            print(f"  {t['table_name']}")
        
        # Check indexes
        indexes = await conn.fetch("SELECT indexname, tablename FROM pg_indexes WHERE schemaname = 'public'")
        print("\nIndexes:")
        for idx in indexes:
            print(f"  {idx['indexname']} on {idx['tablename']}")
        
        await conn.close()
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(test())