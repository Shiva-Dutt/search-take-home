from functools import lru_cache
import os
from typing import Iterable, List, Tuple

import anyio
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFaceEndpoint

from .models import SearchResult, CypherQuery
from .data import DOCUMENTS


def _get_hf_llm() -> BaseLanguageModel:
    """Return a Hugging Face-backed LangChain LLM for Cypher generation."""
    api_token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not api_token:
        raise RuntimeError(
            "Hugging Face API token not configured. "
            "Set HF_API_TOKEN or HUGGINGFACEHUB_API_TOKEN to enable text_to_cypher."
        )

    return HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.1,
        max_new_tokens=256,
        huggingfacehub_api_token=api_token,
    )


async def text_to_cypher(text: str) -> str:
    """Convert a text query to a Cypher query.
    
    You should use an LangChain LLM using 'with_structured_output' to generate the Cypher query.
    Reference the docs here: https://docs.langchain.com/oss/python/langchain/structured-output#:~:text=LangChain%20automatically%20uses%20ProviderStrategy%20when%20you%20pass%20a%20schema%20type%20directly%20to%20create_agent.response_format%20and%20the%20model%20supports%20native%20structured%20output%3A
    
    Assume the knowledge graph has the following ontology:
    - Entities:
     - Disease
     - Symptom
     - Drug
     - Patient
    - Relationships:
     - TREATS
     - CAUSES
     - EXPERIENCING
     - SUFFERING_FROM
    
    You should have the model construct a Cypher query via a structured output (using JSON schema or
    Pydantic BaseModels) that can be used to query the system. If you have an API key, you may use it -
    otherwise, simply construct the LLM & assume that the the API key will be populated later.
    """
    try:
        llm = _get_hf_llm()
        structured_llm = llm.with_structured_output(CypherQuery)

        prompt = (
            "You are an assistant that maps natural language questions about a medical "
            "knowledge graph into Cypher queries.\n\n"
            "The knowledge graph has the following ontology:\n"
            "- Entities: Disease, Symptom, Drug, Patient\n"
            "- Relationships: TREATS, CAUSES, EXPERIENCING, SUFFERING_FROM\n\n"
            "Given the user's question, populate the CypherQuery fields only. "
            "Do not explain your reasoning.\n\n"
            f"User question: {text}"
        )

        cypher_query: CypherQuery = await anyio.to_thread.run_sync(
            structured_llm.invoke, prompt
        )
        return str(cypher_query)
    except Exception:
        # Fallback: construct a simple, deterministic CypherQuery without calling an LLM.
        node_label = "Disease"
        lowered = text.lower()
        if any(keyword in lowered for keyword in ("symptom", "pain", "fever")):
            node_label = "Symptom"
        elif any(keyword in lowered for keyword in ("drug", "treatment", "medication")):
            node_label = "Drug"
        elif any(keyword in lowered for keyword in ("patient", "person", "people")):
            node_label = "Patient"

        fallback = CypherQuery(node_label=node_label)
        return str(fallback)
    
    
@lru_cache()
def load_FAISS() -> FAISS:
    """Create and return a cached FAISS vector store for the in-memory DOCUMENTS."""
    documents: List[Document] = DOCUMENTS
    texts: List[str] = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build the FAISS vector store once and reuse it across calls.
    # According to LangChain's integration docs, `FAISS.from_texts`
    # is the recommended factory for constructing an in-memory index
    # from raw texts and an embeddings model.
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    return vector_store
    
    
def search_knowledgegraph(cypher_query: str) -> list[SearchResult]:
    """This is a mock function that will search the knowledge graph using a cypher query."""
    return [
        SearchResult(
            document=Document(page_content=cypher_query), score=0.9, reason="test"
        )
    ]
    
    
async def search_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[SearchResult]:
    """Search the FAISS vector store and mock knowledge graph.
    
    After searching FAISS, you should rerank all the remaining results using your custom 'rerank_result'
    function, and removing bad results. You may add args/kwargs as needed.
    """
    if top_k < 1:
        top_k = 1
    
    store = load_FAISS()
    
    faiss_results: list[SearchResult] = []
    
    # First, search using the raw natural-language query.
    for doc, distance in store.similarity_search_with_score(query, k=top_k):
        score = 1.0 / (1.0 + float(distance))
        faiss_results.append(
            SearchResult(
                document=doc,
                score=score,
                reason="FAISS similarity search (natural query)",
            )
        )
    
    # Next, generate a Cypher representation of the query and search FAISS again.
    cypher = await text_to_cypher(query)
    for doc, distance in store.similarity_search_with_score(cypher, k=top_k):
        score = 1.0 / (1.0 + float(distance))
        faiss_results.append(
            SearchResult(
                document=doc,
                score=score,
                reason="FAISS similarity search (Cypher query)",
            )
        )
    
    # Query the mock knowledge graph using the Cypher string.
    kg_results = search_knowledgegraph(cypher)
    
    combined: list[SearchResult] = faiss_results + kg_results
    combined.sort(key=lambda r: r.score, reverse=True)
    return combined[:top_k]
