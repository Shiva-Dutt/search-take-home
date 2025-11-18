import asyncio

from langchain_core.documents import Document


def vectorize(text: str):
    pass


def similarity_score(query: str, document: Document) -> float:
    pass


async def llm_scoring(document: Document) -> float:
    pass


async def score_documents(documents: list[Document]) -> list[float]:
    scores = await asyncio.gather(*[llm_scoring(doc) for doc in documents])
    return scores
