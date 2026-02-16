from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse


router = APIRouter(prefix="/streaming", tags=["streaming"])


def _sse(data: str, event: str | None = None) -> str:
    """Format a Server-Sent Event (SSE) message.

    Candidates: you may keep this helper or replace it.
    """

    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")

    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")

    return "\n".join(lines) + "\n\n"


@router.get("/notepad")
async def stream_notepad(
    path: str = Query(
        default=str(Path("data/notepad.txt")),
        description="Path to the .txt file to stream. Relative paths are resolved from the backend working directory.",
    ),
    chunk_size: int = Query(default=10, ge=1, le=8192),
    delay_ms: int = Query(default=30, ge=0, le=5000),
) -> StreamingResponse:
    """Stream a .txt file to the frontend as Server-Sent Events (SSE).
    
    - Streams incremental chunks so the UI can render as data arrives.
    - Handles missing files and invalid params cleanly.
    """

    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (Path.cwd() / file_path).resolve()

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    async def event_generator() -> AsyncIterator[bytes]:
        """Yield SSE events that stream `file_path` progressively."""
        async def _read_file() -> AsyncIterator[str]:
            with file_path.open("r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        # Send a one-time metadata event so the frontend can display the source.
        meta_payload = f"path={file_path}; chunk_size={chunk_size}; delay_ms={delay_ms}"
        yield _sse(meta_payload, event="meta").encode("utf-8")

        async for chunk in _read_file():
            yield _sse(chunk, event="chunk").encode("utf-8")
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

        # Signal completion so the client can update status and close the stream.
        yield _sse("done", event="done").encode("utf-8")

    return StreamingResponse(event_generator(), media_type="text/event-stream")
