import { useEffect, useRef, useState } from "react";

const BACKEND_URL =
  (import.meta.env.VITE_BACKEND_URL as string) || "http://localhost:8000";

type LibrarySummary = { name: string; document_count: number };
type LibraryFile = { filename: string; chunk_count: number };
type Citation = {
  source?: string | null;
  title?: string | null;
  page?: number | null;
  chunk_id?: string | null;
  snippet: string;
};
type ChatStatus =
  | "answered"
  | "needs_web_search"
  | "answered_from_web"
  | "refused";
type ChatResponse = {
  query: string;
  collection: string;
  status: ChatStatus;
  answer: string;
  citations: Citation[];
  reason: string | null;
  rewritten_query: string | null;
};
type Message = {
  role: "user" | "assistant";
  content: string;
  status?: ChatStatus;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BACKEND_URL}${path}`, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = j.detail || detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return r.json();
}

function formatBotMessage(d: ChatResponse): string {
  if (d.status === "answered") return d.answer;
  if (d.status === "needs_web_search") {
    return [
      "**Not enough info in your library to answer confidently.**",
      `_Grader's verdict: ${d.reason ?? "—"}_`,
      "Click **Search the web** below to retry with web sources.",
    ].join("\n\n");
  }
  if (d.status === "answered_from_web") {
    const prefix = d.rewritten_query
      ? `_Searched the web for:_ \`${d.rewritten_query}\`\n\n`
      : "";
    return prefix + d.answer;
  }
  if (d.status === "refused") {
    const prefix = d.rewritten_query
      ? `_Tried searching the web for:_ \`${d.rewritten_query}\`\n\n`
      : "";
    return prefix + d.answer;
  }
  return d.answer || "(no response)";
}

// Minimal markdown: bold, italic, code, paragraphs. No XSS — only the LLM
// answer flows through here and it's already escaped server-side, but we
// still treat text as text.
function renderMarkdown(s: string): string {
  const escape = (t: string) =>
    t
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  return escape(s)
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-cream-200 text-brown-700">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*|_([^_]+)_/g, (_m, a, b) => `<em>${a ?? b}</em>`)
    .replace(/\n\n+/g, '</p><p class="mt-2">')
    .replace(/\n/g, "<br/>");
}

export default function App() {
  const [libraries, setLibraries] = useState<LibrarySummary[]>([]);
  const [activeLib, setActiveLib] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [newLibName, setNewLibName] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [tab, setTab] = useState<"chat" | "files">("chat");
  const [chats, setChats] = useState<Record<string, Message[]>>({});
  const [citations, setCitations] = useState<Citation[]>([]);
  const [pendingWebQuery, setPendingWebQuery] = useState<string | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [files, setFiles] = useState<LibraryFile[]>([]);
  const [busy, setBusy] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [query, setQuery] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);

  const currentChat = activeLib ? chats[activeLib] ?? [] : [];

  // Initial library list
  useEffect(() => {
    refreshLibraries();
  }, []);

  // Refresh files when library or tab changes
  useEffect(() => {
    if (!activeLib) {
      setFiles([]);
      return;
    }
    refreshFiles(activeLib);
  }, [activeLib]);

  // Scroll to bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentChat.length, busy]);

  async function refreshLibraries() {
    try {
      const data = await api<{ libraries: LibrarySummary[] }>("/api/libraries");
      setLibraries(data.libraries);
      if (!activeLib && data.libraries.length > 0) {
        setActiveLib(data.libraries[0].name);
      }
    } catch {
      // ignore — backend probably not up yet
    }
  }

  async function refreshFiles(lib: string) {
    try {
      const data = await api<{ files: LibraryFile[] }>(
        `/api/libraries/${lib}/files`
      );
      setFiles(data.files);
    } catch {
      setFiles([]);
    }
  }

  async function createLibrary() {
    const name = newLibName.trim();
    if (!name) return;
    setCreateError(null);
    try {
      await api("/api/libraries", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      setNewLibName("");
      setShowCreate(false);
      await refreshLibraries();
      setActiveLib(name);
    } catch (e) {
      setCreateError(String((e as Error).message));
    }
  }

  async function uploadPdf(file: File) {
    if (!activeLib) return;
    setUploadMsg(null);
    setBusy(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await api<{ filename: string; pages: number; chunks_added: number }>(
        `/api/libraries/${activeLib}/documents`,
        { method: "POST", body: form }
      );
      setUploadMsg({
        ok: true,
        text: `Ingested ${data.filename}: ${data.pages} pages → ${data.chunks_added} chunks.`,
      });
      await refreshFiles(activeLib);
      await refreshLibraries();
    } catch (e) {
      setUploadMsg({ ok: false, text: String((e as Error).message) });
    } finally {
      setBusy(false);
    }
  }

  async function ask(allowWebSearch: boolean, queryOverride?: string) {
    if (!activeLib) return;
    const q = (queryOverride ?? query).trim();
    if (!q) return;

    setBusy(true);
    if (!queryOverride) {
      setChats((prev) => {
        const next = { ...prev };
        next[activeLib] = [...(next[activeLib] ?? []), { role: "user", content: q }];
        return next;
      });
      setQuery("");
    }

    try {
      const data = await api<ChatResponse>("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          collection: activeLib,
          allow_web_search: allowWebSearch,
        }),
      });
      setChats((prev) => {
        const next = { ...prev };
        next[activeLib] = [
          ...(next[activeLib] ?? []),
          { role: "assistant", content: formatBotMessage(data), status: data.status },
        ];
        return next;
      });
      setCitations(data.citations || []);
      setPendingWebQuery(data.status === "needs_web_search" ? q : null);
    } catch (e) {
      setChats((prev) => {
        const next = { ...prev };
        next[activeLib] = [
          ...(next[activeLib] ?? []),
          { role: "assistant", content: `Backend error: ${(e as Error).message}` },
        ];
        return next;
      });
    } finally {
      setBusy(false);
    }
  }

  async function exportPdf() {
    if (!activeLib || currentChat.length === 0) return;
    const { jsPDF } = await import("jspdf");
    const pdf = new jsPDF();
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(18);
    pdf.text(`Chat — ${activeLib}`, 14, 20);
    pdf.setFontSize(10);
    pdf.setTextColor(120);
    pdf.text(new Date().toLocaleString(), 14, 27);
    pdf.setTextColor(0);

    let y = 38;
    const pageHeight = pdf.internal.pageSize.getHeight();
    const wrap = (text: string, maxWidth: number) => pdf.splitTextToSize(text, maxWidth);

    for (const msg of currentChat) {
      if (y > pageHeight - 20) {
        pdf.addPage();
        y = 20;
      }
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(11);
      const label = msg.role === "user" ? "You" : "Assistant";
      const status = msg.status && msg.status !== "answered" ? `  [${msg.status.replace(/_/g, " ")}]` : "";
      pdf.text(`${label}${status}`, 14, y);
      y += 6;
      pdf.setFont("helvetica", "normal");
      const lines = wrap(msg.content.replace(/[*_`]/g, ""), 180);
      for (const ln of lines) {
        if (y > pageHeight - 15) {
          pdf.addPage();
          y = 20;
        }
        pdf.text(ln, 14, y);
        y += 5;
      }
      y += 4;
    }

    if (citations.length) {
      if (y > pageHeight - 30) {
        pdf.addPage();
        y = 20;
      }
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(13);
      pdf.text("Sources", 14, y);
      y += 8;
      pdf.setFontSize(10);
      for (const [i, c] of citations.entries()) {
        if (y > pageHeight - 15) {
          pdf.addPage();
          y = 20;
        }
        const head =
          c.page != null
            ? `${i + 1}. ${c.source ?? "unknown"} (p.${c.page})`
            : c.title
            ? `${i + 1}. ${c.title} — ${c.source ?? ""}`
            : `${i + 1}. ${c.source ?? "unknown"}`;
        pdf.setFont("helvetica", "bold");
        pdf.text(head, 14, y);
        y += 5;
        pdf.setFont("helvetica", "normal");
        pdf.setTextColor(110);
        const snippet = (c.snippet || "").slice(0, 300);
        for (const ln of wrap(snippet, 180)) {
          if (y > pageHeight - 15) {
            pdf.addPage();
            y = 20;
          }
          pdf.text(ln, 14, y);
          y += 4.5;
        }
        pdf.setTextColor(0);
        y += 2;
      }
    }

    pdf.save(`${activeLib}-chat.pdf`);
  }

  return (
    <div className="h-screen flex flex-col">
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-72 bg-cream-200 border-r border-cream-300 flex flex-col">
          <div className="p-6 border-b border-cream-300">
            <h1 className="text-2xl font-semibold text-brown-700">Universal RAG</h1>
            <p className="text-sm text-brown-500 mt-1">
              Ask your PDFs, with citations.
            </p>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs uppercase tracking-wider text-brown-500 font-semibold">
                Libraries
              </h2>
              <button
                onClick={() => setShowCreate((s) => !s)}
                className="text-brown-500 hover:text-brown-700 text-lg leading-none w-6 h-6 rounded hover:bg-cream-300 transition"
                title="Create library"
              >
                +
              </button>
            </div>

            {showCreate && (
              <div className="mb-3 p-3 bg-cream-50 border border-cream-300 rounded-md">
                <input
                  type="text"
                  value={newLibName}
                  placeholder="ml-papers"
                  onChange={(e) => setNewLibName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && createLibrary()}
                  autoFocus
                  className="w-full px-2 py-1.5 text-sm bg-cream-100 border border-cream-300 rounded focus:outline-none focus:border-brown-400"
                />
                {createError && (
                  <p className="text-xs text-red-700 mt-1">{createError}</p>
                )}
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={createLibrary}
                    className="flex-1 px-3 py-1.5 text-sm bg-brown-700 text-cream-50 rounded hover:bg-brown-600 transition"
                  >
                    Create
                  </button>
                  <button
                    onClick={() => {
                      setShowCreate(false);
                      setNewLibName("");
                      setCreateError(null);
                    }}
                    className="px-3 py-1.5 text-sm text-brown-500 hover:text-brown-700 transition"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {libraries.length === 0 && !showCreate && (
              <p className="text-sm text-brown-500 italic">
                No libraries yet. Click + to create one.
              </p>
            )}

            <ul className="space-y-1">
              {libraries.map((lib) => (
                <li key={lib.name}>
                  <button
                    onClick={() => {
                      setActiveLib(lib.name);
                      setCitations([]);
                      setPendingWebQuery(null);
                    }}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm transition ${
                      lib.name === activeLib
                        ? "bg-brown-700 text-cream-50"
                        : "text-brown-700 hover:bg-cream-300"
                    }`}
                  >
                    <div className="font-medium">{lib.name}</div>
                    <div
                      className={`text-xs ${
                        lib.name === activeLib ? "text-cream-200" : "text-brown-500"
                      }`}
                    >
                      {lib.document_count} chunks
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {!activeLib ? (
            <div className="flex-1 flex items-center justify-center p-10">
              <div className="max-w-md text-center">
                <h2 className="text-3xl font-semibold text-brown-700 mb-3">
                  Welcome
                </h2>
                <p className="text-brown-500">
                  Create a library in the sidebar to get started. Each library
                  is its own collection of PDFs you can chat with, with cited
                  answers and an opt-in web fallback.
                </p>
              </div>
            </div>
          ) : (
            <>
              <header className="flex items-center justify-between px-8 py-5 border-b border-cream-300">
                <div>
                  <h2 className="text-2xl font-semibold text-brown-700">
                    {activeLib}
                  </h2>
                  <p className="text-xs text-brown-500 mt-0.5">
                    {libraries.find((l) => l.name === activeLib)?.document_count ?? 0} chunks
                  </p>
                </div>
                {currentChat.length > 0 && (
                  <button
                    onClick={exportPdf}
                    className="px-4 py-1.5 text-sm border border-cream-400 rounded-md hover:bg-cream-200 transition text-brown-700"
                  >
                    Export PDF
                  </button>
                )}
              </header>

              <nav className="flex border-b border-cream-300 px-8">
                {(["chat", "files"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`px-1 py-3 mr-8 text-sm font-medium capitalize transition border-b-2 ${
                      tab === t
                        ? "text-brown-700 border-brown-400"
                        : "text-brown-500 border-transparent hover:text-brown-700"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </nav>

              {tab === "chat" ? (
                <>
                  <div className="flex-1 overflow-y-auto px-8 py-6">
                    {currentChat.length === 0 ? (
                      <p className="text-brown-500 italic text-center mt-10">
                        Ask a question to start.
                      </p>
                    ) : (
                      <div className="space-y-4 max-w-3xl mx-auto">
                        {currentChat.map((m, i) => (
                          <div
                            key={i}
                            className={
                              m.role === "user" ? "flex justify-end" : "flex justify-start"
                            }
                          >
                            <div
                              className={`max-w-[80%] px-4 py-3 rounded-lg ${
                                m.role === "user"
                                  ? "bg-brown-700 text-cream-50"
                                  : "bg-cream-50 border border-cream-300 text-brown-700"
                              }`}
                              dangerouslySetInnerHTML={{
                                __html: `<p>${renderMarkdown(m.content)}</p>`,
                              }}
                            />
                          </div>
                        ))}
                        {busy && (
                          <div className="flex justify-start">
                            <div className="px-4 py-3 rounded-lg bg-cream-50 border border-cream-300 text-brown-500 italic">
                              Thinking…
                            </div>
                          </div>
                        )}
                        <div ref={chatEndRef} />
                      </div>
                    )}
                  </div>

                  {pendingWebQuery && !busy && (
                    <div className="px-8 pb-3 max-w-3xl mx-auto w-full">
                      <button
                        onClick={() => {
                          const q = pendingWebQuery;
                          setPendingWebQuery(null);
                          ask(true, q);
                        }}
                        className="w-full px-4 py-2 bg-brown-700 text-cream-50 rounded-md hover:bg-brown-600 transition"
                      >
                        Search the web
                      </button>
                    </div>
                  )}

                  {citations.length > 0 && (
                    <div className="px-8 pb-3 max-w-3xl mx-auto w-full">
                      <button
                        onClick={() => setSourcesOpen((o) => !o)}
                        className="text-xs text-brown-500 hover:text-brown-700 underline"
                      >
                        {sourcesOpen ? "Hide" : "Show"} sources ({citations.length})
                      </button>
                      {sourcesOpen && (
                        <div className="mt-2 space-y-2">
                          {citations.map((c, i) => (
                            <div
                              key={i}
                              className="p-3 bg-cream-50 border border-cream-300 rounded-md text-sm"
                            >
                              <div className="font-medium text-brown-700">
                                {i + 1}.{" "}
                                {c.page != null
                                  ? `${c.source} (p.${c.page})`
                                  : c.title
                                  ? `${c.title} — ${c.source ?? ""}`
                                  : c.source ?? "unknown"}
                              </div>
                              {c.snippet && (
                                <div className="text-xs text-brown-500 mt-1">
                                  {c.snippet.slice(0, 240)}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      ask(false);
                    }}
                    className="border-t border-cream-300 px-8 py-4"
                  >
                    <div className="max-w-3xl mx-auto flex gap-2">
                      <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder={`Ask ${activeLib}…`}
                        disabled={busy}
                        className="flex-1 px-4 py-2.5 bg-cream-50 border border-cream-300 rounded-md focus:outline-none focus:border-brown-400 text-brown-700"
                      />
                      <button
                        type="submit"
                        disabled={busy || !query.trim()}
                        className="px-5 py-2.5 bg-brown-700 text-cream-50 rounded-md hover:bg-brown-600 transition disabled:opacity-50"
                      >
                        Send
                      </button>
                    </div>
                  </form>
                </>
              ) : (
                <div className="flex-1 overflow-y-auto px-8 py-6 max-w-3xl mx-auto w-full">
                  <section className="mb-8">
                    <h3 className="text-lg font-semibold text-brown-700 mb-3">
                      Upload a PDF
                    </h3>
                    <label
                      className={`block border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition ${
                        busy
                          ? "border-cream-300 bg-cream-50 opacity-50"
                          : "border-cream-400 bg-cream-50 hover:border-brown-400 hover:bg-cream-100"
                      }`}
                    >
                      <input
                        type="file"
                        accept=".pdf"
                        disabled={busy}
                        className="hidden"
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) uploadPdf(f);
                          e.target.value = "";
                        }}
                      />
                      <p className="text-brown-500">
                        {busy ? "Ingesting…" : "Click or drop a PDF here"}
                      </p>
                      <p className="text-xs text-brown-400 mt-1">Max 10MB</p>
                    </label>
                    {uploadMsg && (
                      <p
                        className={`mt-2 text-sm ${
                          uploadMsg.ok ? "text-brown-700" : "text-red-700"
                        }`}
                      >
                        {uploadMsg.text}
                      </p>
                    )}
                  </section>

                  <section>
                    <h3 className="text-lg font-semibold text-brown-700 mb-3">
                      Files in this library ({files.length})
                    </h3>
                    {files.length === 0 ? (
                      <p className="text-sm text-brown-500 italic">
                        No files yet. Upload one above.
                      </p>
                    ) : (
                      <ul className="space-y-2">
                        {files.map((f) => (
                          <li
                            key={f.filename}
                            className="flex items-center justify-between p-3 bg-cream-50 border border-cream-300 rounded-md"
                          >
                            <span className="font-medium text-brown-700">
                              {f.filename}
                            </span>
                            <span className="text-xs text-brown-500">
                              {f.chunk_count} chunks
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </section>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
