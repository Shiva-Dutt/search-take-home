# Search Take-Home Assignment

This repo contains a small FastAPI backend and a React (Vite) frontend. Your task is to implement a simple search endpoint and wire up the existing search UI to use it.

## What You Need to Build

### 1. Backend: `/api/search`

In `backend/features/search/search_router.py`, you'll find a POST endpoint stub for `/api/search`.

Your job:

- Implement ranking logic over a small in-memory corpus `DOCUMENTS`
- A simple approach is fine (e.g. tokenize query --> match tokens in title/body --> compute a score).
- Return the top `k` results as `SearchResult` objects.
- Include a short "reason" string explaining why each document matched.
- Keep the code clean, readable and maintainable.

No external ML libraries are required.

### 2. Frontend: Search UI

The file `frontend/src/features/search/SearchPage.tsx` contains a skeleton UI with TODOs.

Your job:

- Call the `/api/search` endpoint using the helper in `src/lib/api.ts`.
- Implement loading and error states
- Render the returned results (title, score, reason).
- Handle empty states gracefully.

This does not require styling beyond basic TSX/inline CSS.

## Repository Structure

```text
backend/
  main.py                              # FastAPI app setup + router registration
  requirements.txt
  data/
    documents.json                     # Small corpus of documents for search
  features/search/
    data.py                            # Loads DOCUMENTS into memory
    models.py                          # Pydantic models (Document, SearchRequest, SearchResult)
    search_router.py                   # Your backend task lives here
frontend/
  src/lib/api.ts                       # Helper to call /api/search
  src/features/search/SearchPage.tsx   # Your frontend task lives here
```

You should only need to modify:

- `backend/features/search/search_router.py`
- `frontend/src/features/search/SearchPage.tsx`

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
- Clean, simple React
- Correct data flow (loading, error, results)
