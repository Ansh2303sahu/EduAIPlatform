import asyncio
from typing import Tuple

CLAMD_HOST = "clamav"   # docker service name
CLAMD_PORT = 3310

# Minimal INSTREAM protocol: send file to clamd
async def clamav_scan_bytes(data: bytes) -> Tuple[bool, str]:
    reader, writer = await asyncio.open_connection(CLAMD_HOST, CLAMD_PORT)
    try:
        writer.write(b"zINSTREAM\0")
        await writer.drain()

        # send in chunks: [len:4bytes big endian][data]
        chunk_size = 8192
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            writer.write(len(chunk).to_bytes(4, "big") + chunk)
            await writer.drain()

        # terminate stream
        writer.write((0).to_bytes(4, "big"))
        await writer.drain()

        result = await reader.readline()
        text = result.decode("utf-8", "replace").strip()

        # typical: "stream: OK" or "stream: Eicar-Test-Signature FOUND"
        if "FOUND" in text:
            return False, text
        if "OK" in text:
            return True, text
        return False, f"unknown_scan_response: {text}"
    finally:
        writer.close()
        await writer.wait_closed()
