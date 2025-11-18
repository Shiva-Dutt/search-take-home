from datetime import datetime

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    document: Document
    score: float = Field(..., ge=0)
    reason: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class SearchEntry(BaseModel):
    query: str = Field(..., min_length=1)
    timestamp: datetime
