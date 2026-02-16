import { useEffect, useMemo, useState } from "react";
import { streamNotepad } from "../../lib/streaming";

export function NotepadStream() {
  const [meta, setMeta] = useState<string>("");
  const [text, setText] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "streaming" | "done" | "error">(
    "idle",
  );

  const abort = useMemo(() => {
    return streamNotepad(
      {
        onMeta: (d) => setMeta(d),
        onChunk: (d) => {
          setStatus("streaming");
          setText((prev) => prev + d);
        },
        onDone: () => setStatus("done"),
        onError: () => setStatus("error"),
      },
      { path: "data/notepad.txt", chunkSize: 15, delayMs: 10 },
    );
  }, []);

  useEffect(() => abort, [abort]);

  return (
    <section style={{ marginTop: "2rem" }}>
      <h2>Streaming Notepad</h2>
      <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 8 }}>
        <div>source: {meta || "(connecting...)"}</div>
        <div>status: {status}</div>
      </div>
      <pre
        style={{
          background: "#0b1020",
          color: "#d7e1ff",
          padding: 12,
          borderRadius: 8,
          whiteSpace: "pre-wrap",
          overflow: "auto",
          maxHeight: 280,
        }}
      >
        {text || (status === "idle" ? "(waiting for stream...)" : "")}
      </pre>
    </section>
  );
}
