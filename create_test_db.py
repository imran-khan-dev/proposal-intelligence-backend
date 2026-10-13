import asyncpg
import asyncio

async def test():
    try:
        # Connect to default postgres database to create test database
        conn = await asyncpg.connect('postgresql://postgres:45.86.86.86@localhost:5432/postgres')
        
        # Check if test database exists
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = 'rfp_ai_test'")
        print(f"Test database exists: {exists}")
        
        if not exists:
            await conn.execute('CREATE DATABASE rfp_ai_test')
            print('Created test database rfp_ai_test')
        else:
            print('Test database already exists')
        
        await conn.close()
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(test())