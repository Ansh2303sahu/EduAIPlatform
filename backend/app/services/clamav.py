from __future__ import annotations

import asyncio
from typing import Tuple

from app.core.config import settings


class ClamAVError(RuntimeError):
    pass


def _clean_clamd_text(s: str) -> str:
    # Postgres TEXT cannot store NUL; also keep it printable-ish.
    return (s or "").replace("\x00", "").strip()


async def clamav_scan_bytes(data: bytes) -> Tuple[bool, str]:
    """
    Scan bytes using clamd INSTREAM.
    Returns (is_clean, result_text)

    Typical result_text:
      - "stream: OK"
      - "stream: Eicar-Test-Signature FOUND"
    """
    host = getattr(settings, "clamd_host", "clamav")
    port = int(getattr(settings, "clamd_port", 3310))

    # Defense-in-depth: avoid absurd payloads reaching clamd
    max_bytes = int(getattr(settings, "max_upload_bytes", 10 * 1024 * 1024))
    if len(data) > max_bytes:
        raise ClamAVError(f"Refusing to scan: payload too large ({len(data)} bytes)")

    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=3.0)
    except Exception as e:
        raise ClamAVError(f"Could not connect to clamd at {host}:{port}: {e}") from e

    try:
        # INSTREAM command (z = "null-terminated" in clamd protocol)
        writer.write(b"zINSTREAM\0")
        await asyncio.wait_for(writer.drain(), timeout=3.0)

        chunk_size = 8192
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            writer.write(len(chunk).to_bytes(4, "big") + chunk)
            await asyncio.wait_for(writer.drain(), timeout=10.0)

        # Terminate stream
        writer.write((0).to_bytes(4, "big"))
        await asyncio.wait_for(writer.drain(), timeout=3.0)

        # Read response line (cap time)
        line = await asyncio.wait_for(reader.readline(), timeout=10.0)

        # Decode safely; strip NULs defensively
        text = _clean_clamd_text(line.decode("utf-8", "replace"))

        # Normalize decision
        # clamd replies commonly:
        #   "stream: OK"
        #   "stream: <SigName> FOUND"
        #   "stream: ERROR <...>"
        if "FOUND" in text:
            return False, text
        if text.endswith("OK") or " OK" in text:
            return True, text
        if "ERROR" in text:
            # Treat clamd errors as NOT clean (quarantine)
            return False, f"clamd_error: {text}"

        # Unknown response -> quarantine
        return False, f"unknown_clamd_response: {text}"

    except asyncio.TimeoutError as e:
        raise ClamAVError("clamd scan timed out") from e
    except Exception as e:
        raise ClamAVError(f"clamd scan failed: {e}") from e
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def clamav_ping() -> bool:
    """
    Lightweight health check: PING -> PONG.
    Useful for startup checks/tests.
    """
    host = getattr(settings, "clamd_host", "clamav")
    port = int(getattr(settings, "clamd_port", 3310))

    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2.0)
        writer.write(b"zPING\0")
        await asyncio.wait_for(writer.drain(), timeout=2.0)
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        text = _clean_clamd_text(line.decode("utf-8", "replace"))
        return "PONG" in text
    except Exception:
        return False
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
