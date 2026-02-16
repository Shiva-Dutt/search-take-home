export type SearchDocumentMetadata = {
  id: number;
  title: string;
  // Allow additional metadata fields without losing type safety.
  [key: string]: unknown;
};

export type SearchDocument = {
  page_content: string;
  metadata: SearchDocumentMetadata;
};

export type SearchResult = {
  document: SearchDocument;
  score: number;
  reason?: string;
};

export class SearchError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "SearchError";
    this.status = status;
  }
}

/**
 * Perform a search request against the backend.
 *
 * TODO (candidate):
 * - Call the `/api/search` endpoint via `fetch`.
 * - Send `{ query, top_k: topK }` as a JSON body.
 * - On non-OK responses, throw a `SearchError` with a helpful message and
 *   the HTTP status code.
 * - Parse and return the JSON response typed as `SearchResult[]`.
 */
export async function search(query: string, topK = 5): Promise<SearchResult[]> {
  const trimmed = query.trim();
  if (!trimmed) {
    return [];
  }

  const response = await fetch("/api/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: trimmed,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    let message = `Search request failed with status ${response.status}`;
    try {
      const contentType = response.headers.get("content-type") ?? "";
      if (contentType.includes("application/json")) {
        const data = (await response.json()) as { detail?: unknown };
        if (data && typeof data.detail === "string") {
          message = `Search failed (${response.status}): ${data.detail}`;
        }
      } else {
        const text = await response.text();
        if (text) {
          message = `Search failed (${response.status}): ${text}`;
        }
      }
    } catch {
      // Swallow JSON/text parsing errors and fall back to the generic message.
    }

    throw new SearchError(message, response.status);
  }

  const data = (await response.json()) as SearchResult[];
  return data;
}
