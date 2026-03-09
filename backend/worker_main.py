import asyncio

from app.worker.ingestion_worker import run_loop

if __name__ == "__main__":
    asyncio.run(run_loop())
