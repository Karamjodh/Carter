from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

DATABASE_URL = "sqlite+aiosqlite:///./carter.db" # dialect+driver://connection_details

engine = create_async_engine(DATABASE_URL, echo = True)
AsyncSessionLocal = async_sessionmaker(engine, class_ = AsyncSession, expire_on_commit = False)
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise