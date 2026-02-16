from fastapi import APIRouter, HTTPException

from .data import DOCUMENTS  # noqa: F401
from .models import SearchEntry, SearchRequest, SearchResult
from .integrations import search_documents

router = APIRouter(prefix="/search", tags=["search"])
SEARCH_HISTORY: list[SearchEntry] = []


@router.post("", response_model=list[SearchResult])
async def search(request: SearchRequest) -> list[SearchResult]:
    """
    Search over the in-memory DOCUMENTS collection.
    - Implement simple ranking logic over DOCUMENTS based on `request.query`.
    - Return the top `request.top_k` results, sorted by score (highest first).

    Focus on clear, readable, and well-structured code.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    results: list[SearchResult] = await search_documents(
        query=query,
        documents=DOCUMENTS,
        top_k=request.top_k,
    )
    return results
