import asyncio
import logging

from app.worker.prof_ingestion_worker import run_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [prof-worker] %(message)s",
)

if __name__ == "__main__":
    logging.info("Starting prof-worker...")
    asyncio.run(run_loop())
