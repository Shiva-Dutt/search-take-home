export type SearchQuery = {
  /** Raw query string entered by the user. */
  query: string;
  /** Unix timestamp in milliseconds when the query was executed. */
  timestamp: number;
};

export type SearchHistory = SearchQuery[];

export type AddToHistoryOptions = {
  /**
   * Maximum number of entries to keep.
   * Defaults to 10 if not provided.
   */
  maxEntries?: number;
};

/**
 * Add a new query to the existing history and return a new history array.
 *
 * TODO (candidate):
 * - Ignore queries that are empty or only whitespace.
 * - Add the new query as the most recent entry with the current timestamp.
 * - Avoid duplicate *adjacent* queries (if the last query has the same text,
 *   update its timestamp instead of adding a new entry).
 * - Trim the history so it never exceeds `maxEntries` (default 10).
 * - Do not mutate the `history` array; always return a new one.
 */
export function addToHistory(
  history: SearchHistory,
  query: string,
  options: AddToHistoryOptions = {},
): SearchHistory {
  const trimmed = query.trim();
  if (!trimmed) {
    return history.slice();
  }

  const maxEntries = options.maxEntries ?? 10;
  if (maxEntries <= 0) {
    return [];
  }

  const now = Date.now();

  if (history.length > 0 && history[0].query === trimmed) {
    const [latest, ...rest] = history;
    const updated: SearchQuery = { ...latest, timestamp: now };
    return [updated, ...rest].slice(0, maxEntries);
  }

  const nextEntry: SearchQuery = {
    query: trimmed,
    timestamp: now,
  };

  return [nextEntry, ...history].slice(0, maxEntries);
}

/**
 * Return a list of recent query strings in most-recent-first order.
 *
 * TODO (candidate):
 * - Use the provided `history` array (do not mutate it).
 * - Return only the `query` strings, ordered from newest to oldest.
 * - Optionally deduplicate consecutive identical strings if your
 *   `addToHistory` implementation does not already enforce this.
 */
export function getRecentQueries(history: SearchHistory): string[] {
  return history.map((entry) => entry.query);
}

