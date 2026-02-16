Note: Config .env file with HUGGINGFACEHUB_API_TOKEN 


# Search Take-Home Assignment

This repo contains a small FastAPI backend and a TypeScript frontend. Your task is to implement:

1. A simple search endpoint + client utilities
2. A small streaming endpoint that streams text from a `.txt` file to the browser (Server-Sent Events)

## What You Need to Build

### 1. Backend: `POST /api/search`

In [`backend/features/search/router.py`](backend/features/search/router.py:1), you'll find a POST endpoint stub for `/api/search`.

Your job:

- Implement ranking logic over a small in-memory corpus `DOCUMENTS`
- Create and utilize FAISS vector database in [`backend/features/search/integrations.py`](backend/features/search/integrations.py:1)
- Implement a `CypherQuery` pydantic `BaseModel` class in [`backend/features/search/models.py`](backend/features/search/models.py:1), with its own `__str__` function that returns itself as a Cypher Query
- Implement a prototype `text_to_cypher` function, which utilizes langchain LLMs, the `CypherQuery`, and `with_structured_output` to create a query from natural language
- Search the FAISS index using the natural language query AND the `CypherQuery`
- Search the knowledgegraph using the mock function (0.0-1.0 range)
- Return the top `k` results as `SearchResult` objects.
- Keep the code clean, readable and maintainable.

### 2. Backend: `GET /api/streaming/notepad` (SSE)

In [`backend/features/streaming/router.py`](backend/features/streaming/router.py:1), you'll find a stub GET endpoint for `/api/streaming/notepad`.

Your job:

- Implement Server-Sent Events (SSE) streaming of a plain text file (provided at [`backend/data/notepad.txt`](backend/data/notepad.txt:1)).
- Stream the contents progressively (in chunks) so the browser UI can display text as it arrives.
- Accept optional query params:
  - `path`: path to a `.txt` file (relative paths resolve from the backend working directory)
  - `chunk_size`: number of characters to read per chunk
  - `delay_ms`: optional delay between chunks (helps demonstrate streaming)
- Return helpful HTTP errors (e.g., 404 if the file does not exist).

Notes:

- The backend endpoint currently returns a placeholder SSE event. Candidates should implement the actual streaming.

### 3. TypeScript: API client & search history

Most of the frontend is already wired up with a simple React UI. Your primary tasks are in the TypeScript modules under [`frontend/src/lib`](frontend/src/lib:1).

Your job:

- In [`frontend/src/lib/api.ts`](frontend/src/lib/api.ts:1), implement a type-safe `search(query: string, topK?: number)` function that:
  - Calls the `/api/search` endpoint via `fetch` with a JSON body.
  - Returns a `Promise<SearchResult[]>` parsed from the JSON response.
  - Throws a descriptive error when the response is not OK (e.g., non-2xx status).
- In [`frontend/src/lib/searchHistory.ts`](frontend/src/lib/searchHistory.ts:1), implement pure, immutable helpers to manage recent search queries:
  - Represent each entry as a typed object (e.g., `SearchQuery`) containing the query string and timestamp.
  - Implement an `addToHistory` function that adds a new query, trims history to a maximum size, and avoids duplicate adjacent queries.
  - Implement a `getRecentQueries` function that returns a list of recent query strings in most-recent-first order.
- The React file [`frontend/src/features/search/SearchPage.tsx`](frontend/src/features/search/SearchPage.tsx:1) is already set up to use these helpers to display results and recent searches.
  - You generally do **not** need to modify React components to complete the assignment.
  - Minor adjustments to the UI are fine if they help you structure the TypeScript code cleanly.

Styling is intentionally minimal; focus on clear, well-typed TypeScript logic rather than CSS or advanced React patterns.

## Repository Structure

```text
backend/
  main.py                              # FastAPI app setup + router registration
  requirements.txt
  data/
    documents.json                     # Small corpus of documents for search
    notepad.txt                        # Text file for SSE streaming exercise
  features/search/
    data.py                            # Loads DOCUMENTS into memory
    models.py                          # Pydantic models (Document, SearchRequest, SearchResult)
    router.py                          # Search endpoint stub
    integrations.py                    # Search + scoring integrations
  features/streaming/
    router.py                          # SSE endpoint stub
frontend/
  src/lib/api.ts                       # TypeScript client for /api/search (your task)
  src/lib/searchHistory.ts             # Pure TypeScript helpers for search history (your task)
  src/features/search/SearchPage.tsx   # React UI that consumes the TS helpers
```

You should only need to modify:

- [`backend/features/search/router.py`](backend/features/search/router.py:1)
- [`backend/features/search/integrations.py`](backend/features/search/integrations.py:1)
- [`backend/features/search/models.py`](backend/features/search/models.py:1)
- [`backend/features/streaming/router.py`](backend/features/streaming/router.py:1)
- [`frontend/src/lib/api.ts`](frontend/src/lib/api.ts:1)
- [`frontend/src/lib/searchHistory.ts`](frontend/src/lib/searchHistory.ts:1)
- (Optional) [`frontend/src/features/search/SearchPage.tsx`](frontend/src/features/search/SearchPage.tsx:1)

## Running the App

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on windows
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Backend runs on `http://localhost:8000`
Frontend runs on `http://localhost:5173`

## What We're Evaluating

- API correctness
- Python clarity and maintainability
- Appropriate use of models and response structure
- TypeScript clarity, typing, and maintainability
- Correct handling of async flows and errors in TypeScript
- Clean, simple usage of React (no advanced patterns required)
- Correct data flow (request, loading, error, results, recent searches)
- Practical streaming implementation (SSE), and attention to edge cases (file paths, errors, incremental output)

## Submission Instructions

To submit your work:

1. **Fork this repository on GitHub.**
2. **Clone your fork locally and complete the assignment there.**
3. **Commit and push** your changes to your fork's `master` branch.
4. **Open a Pull Request** from your fork's `master` branch into this repository's `master` branch.
5. Title the PR: `Take-Home submission  <Your Name>`
