from datetime import datetime
from typing import Optional

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


class CypherQuery(BaseModel):
    """Fields that can be converted to a Cypher Query in natural language."""

    node_label: str = Field(
        ...,
        description="Primary node label to match, e.g. Disease, Symptom, Drug, Patient.",
    )
    where: Optional[str] = Field(
        default=None,
        description="Optional Cypher WHERE clause without the 'WHERE' keyword.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of rows to return from the knowledge graph.",
    )

    def __str__(self) -> str:
        """Render this model as a Cypher query string."""
        base = f"MATCH (n:{self.node_label})"
        where_clause = f" WHERE {self.where}" if self.where else ""
        return f"{base}{where_clause} RETURN n LIMIT {self.limit}"
