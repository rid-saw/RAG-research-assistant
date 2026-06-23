import { Command } from "cmdk";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState, useMemo } from "react";

const BACKEND_URL =
  (import.meta.env.VITE_BACKEND_URL as string) || "http://localhost:8000";

// --------- types ---------

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

type Stage = "retrieving" | "grading" | "generating" | "web_search";
type StageEvent = { event: "stage"; stage: Stage };
type TokenEvent = { event: "token"; text: string };
type DoneEvent = {
  event: "done";
  status: ChatStatus;
  citations: Citation[];
  reason: string | null;
  rewritten_query: string | null;
};
type ErrEvent = { event: "error"; detail: string };
type StreamEvent = StageEvent | TokenEvent | DoneEvent | ErrEvent;

type Message = {
  role: "user" | "assistant";
  content: string;
  status?: ChatStatus;
  citations?: Citation[];
  reason?: string | null;
  rewritten?: string | null;
};

// --------- helpers ---------

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BACKEND_URL}${path}`, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail || detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return r.json();
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderInlineMd(s: string): string {
  return escapeHtml(s)
    .replace(
      /`([^`]+)`/g,
      '<code class="px-1 py-0.5 rounded bg-cream-200 text-brown-700 text-[0.92em]">$1</code>'
    )
    .replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold">$1</strong>')
    .replace(/_([^_]+)_/g, '<em class="italic">$1</em>');
}

const SUGGESTED_QUESTIONS = [
  "Summarize the main points of this document.",
  "What is the methodology used here?",
  "List the key findings or conclusions.",
  "Are there any limitations or open questions discussed?",
];

const STAGE_LABELS: Record<Stage, string> = {
  retrieving: "Retrieving",
  grading: "Grading",
  web_search: "Searching web",
  generating: "Generating",
};

const STAGE_ORDER: Stage[] = ["retrieving", "grading", "generating"];
const STAGE_ORDER_WEB: Stage[] = ["retrieving", "grading", "web_search", "generating"];

// --------- components ---------

function CitationMarker({
  index,
  citation,
}: {
  index: number;
  citation: Citation | undefined;
}) {
  const [open, setOpen] = useState(false);
  if (!citation) return <sup className="text-brown-400">[{index}]</sup>;
  return (
    <span className="relative inline-block">
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="align-super text-[0.7em] font-semibold text-oxblood-500 hover:text-oxblood-600 px-0.5"
      >
        {index}
      </button>
      <AnimatePresence>
        {open && (
          <motion.span
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 left-0 top-full mt-1 w-80 bg-cream-50 border border-cream-300 rounded-md shadow-lg p-3 text-left"
          >
            <span className="block text-xs font-semibold text-brown-700">
              {citation.page != null
                ? `${citation.source} (p.${citation.page})`
                : citation.title
                ? `${citation.title}`
                : citation.source ?? "unknown"}
            </span>
            {citation.title && citation.source && (
              <span className="block text-[10px] text-brown-400 mt-0.5">
                {citation.source}
              </span>
            )}
            <span className="block text-xs text-brown-500 mt-2 leading-relaxed">
              {citation.snippet.slice(0, 220)}
              {citation.snippet.length > 220 ? "…" : ""}
            </span>
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

function MarkdownWithCitations({
  text,
  citations,
}: {
  text: string;
  citations: Citation[];
}) {
  // Split text on [n] markers; render markers as CitationMarker components.
  const parts = useMemo(() => {
    const out: Array<{ kind: "text" | "cite"; value: string }> = [];
    const regex = /\[(\d+)\]/g;
    let lastIdx = 0;
    let m: RegExpExecArray | null;
    while ((m = regex.exec(text)) !== null) {
      if (m.index > lastIdx) out.push({ kind: "text", value: text.slice(lastIdx, m.index) });
      out.push({ kind: "cite", value: m[1] });
      lastIdx = m.index + m[0].length;
    }
    if (lastIdx < text.length) out.push({ kind: "text", value: text.slice(lastIdx) });
    return out;
  }, [text]);

  const paragraphs: Array<typeof parts> = [[]];
  for (const part of parts) {
    if (part.kind === "text") {
      const blocks = part.value.split(/\n\n+/);
      blocks.forEach((b, i) => {
        if (i > 0) paragraphs.push([]);
        if (b) paragraphs[paragraphs.length - 1].push({ kind: "text", value: b });
      });
    } else {
      paragraphs[paragraphs.length - 1].push(part);
    }
  }

  return (
    <>
      {paragraphs.map((para, pi) => (
        <p key={pi} className={pi > 0 ? "mt-3" : ""}>
          {para.map((p, i) =>
            p.kind === "text" ? (
              <span
                key={i}
                dangerouslySetInnerHTML={{ __html: renderInlineMd(p.value) }}
              />
            ) : (
              <CitationMarker
                key={i}
                index={Number(p.value)}
                citation={citations[Number(p.value) - 1]}
              />
            )
          )}
        </p>
      ))}
    </>
  );
}

function PipelineStrip({
  stage,
  done,
  allowWeb,
}: {
  stage: Stage | null;
  done: boolean;
  allowWeb: boolean;
}) {
  const stages = allowWeb ? STAGE_ORDER_WEB : STAGE_ORDER;
  const reached = stage ? stages.indexOf(stage) : -1;

  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4, transition: { duration: 0.4 } }}
      className="flex items-center gap-5 text-xs"
    >
      {stages.map((s, i) => {
        const isActive = stage === s && !done;
        const isDone = i < reached || done;
        return (
          <div key={s} className="flex items-center gap-1.5">
            {isDone ? (
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-3 h-3 rounded-full bg-oxblood-500 flex items-center justify-center text-[8px] text-cream-50 font-bold"
              >
                ✓
              </motion.span>
            ) : isActive ? (
              <motion.span
                animate={{ scale: [1, 1.3, 1], opacity: [1, 0.6, 1] }}
                transition={{ duration: 1.2, repeat: Infinity }}
                className="w-3 h-3 rounded-full bg-oxblood-500"
              />
            ) : (
              <span className="w-3 h-3 rounded-full border border-cream-400" />
            )}
            <span
              className={
                isActive
                  ? "font-semibold text-brown-700"
                  : isDone
                  ? "text-brown-500"
                  : "text-brown-400"
              }
            >
              {STAGE_LABELS[s]}
            </span>
          </div>
        );
      })}
    </motion.div>
  );
}

function CommandPalette({
  open,
  setOpen,
  libraries,
  onPick,
}: {
  open: boolean;
  setOpen: (v: boolean) => void;
  libraries: LibrarySummary[];
  onPick: (name: string) => void;
}) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-brown-800/30 backdrop-blur-sm flex items-start justify-center pt-32"
          onClick={() => setOpen(false)}
        >
          <motion.div
            initial={{ scale: 0.95, y: -10 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.95, y: -10 }}
            transition={{ duration: 0.15 }}
            onClick={(e) => e.stopPropagation()}
            className="w-[520px] max-w-[90vw] bg-cream-50 border border-cream-300 rounded-lg shadow-2xl overflow-hidden"
          >
            <Command className="w-full" loop>
              <Command.Input
                autoFocus
                placeholder="Switch library…"
                className="w-full px-4 py-3 bg-transparent border-b border-cream-300 text-brown-700 placeholder:text-brown-400 focus:outline-none font-medium"
              />
              <Command.List className="max-h-80 overflow-y-auto p-2">
                <Command.Empty className="px-3 py-6 text-center text-sm text-brown-400">
                  No libraries match.
                </Command.Empty>
                <Command.Group heading="Libraries" className="text-[10px] uppercase tracking-wider text-brown-400 px-3 pt-2 pb-1">
                  {libraries.map((lib) => (
                    <Command.Item
                      key={lib.name}
                      value={lib.name}
                      onSelect={() => {
                        onPick(lib.name);
                        setOpen(false);
                      }}
                      className="flex items-center justify-between px-3 py-2 rounded-md text-sm text-brown-700 cursor-pointer data-[selected=true]:bg-oxblood-500 data-[selected=true]:text-cream-50"
                    >
                      <span className="font-medium">{lib.name}</span>
                      <span className="text-xs opacity-70">{lib.document_count} chunks</span>
                    </Command.Item>
                  ))}
                </Command.Group>
              </Command.List>
            </Command>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// --------- main app ---------

export default function App() {
  const [libraries, setLibraries] = useState<LibrarySummary[]>([]);
  const [activeLib, setActiveLib] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [newLibName, setNewLibName] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [tab, setTab] = useState<"chat" | "files">("chat");
  const [chats, setChats] = useState<Record<string, Message[]>>({});
  const [files, setFiles] = useState<LibraryFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [query, setQuery] = useState("");
  const [pendingWebQuery, setPendingWebQuery] = useState<string | null>(null);

  const [stage, setStage] = useState<Stage | null>(null);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [allowWebInStream, setAllowWebInStream] = useState(false);

  const [paletteOpen, setPaletteOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const currentChat = activeLib ? chats[activeLib] ?? [] : [];

  // initial loads
  useEffect(() => {
    refreshLibraries();
  }, []);

  useEffect(() => {
    if (!activeLib) {
      setFiles([]);
      return;
    }
    refreshFiles(activeLib);
  }, [activeLib]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentChat.length, streamingText, isStreaming]);

  // ⌘K palette
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      } else if (e.key === "Escape") {
        setPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function refreshLibraries() {
    try {
      const data = await api<{ libraries: LibrarySummary[] }>("/api/libraries");
      setLibraries(data.libraries);
      if (!activeLib && data.libraries.length > 0) {
        setActiveLib(data.libraries[0].name);
      }
    } catch {
      // ignore
    }
  }

  async function refreshFiles(lib: string) {
    try {
      const d = await api<{ files: LibraryFile[] }>(`/api/libraries/${lib}/files`);
      setFiles(d.files);
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
    setUploading(true);
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
      setUploading(false);
    }
  }

  const sendQuery = useCallback(
    async (q: string, allowWebSearch: boolean) => {
      if (!activeLib) return;
      const lib = activeLib;
      setStreamingText("");
      setStage(null);
      setIsStreaming(true);
      setAllowWebInStream(allowWebSearch);
      setPendingWebQuery(null);

      if (!allowWebSearch) {
        setChats((prev) => {
          const next = { ...prev };
          next[lib] = [...(next[lib] ?? []), { role: "user", content: q }];
          return next;
        });
      }

      try {
        const r = await fetch(`${BACKEND_URL}/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: q,
            collection: lib,
            allow_web_search: allowWebSearch,
          }),
        });
        if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);

        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let accumulated = "";
        let final: DoneEvent | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let nl;
          while ((nl = buffer.indexOf("\n\n")) !== -1) {
            const block = buffer.slice(0, nl);
            buffer = buffer.slice(nl + 2);
            for (const line of block.split("\n")) {
              if (!line.startsWith("data:")) continue;
              const json = line.slice(5).trim();
              if (!json) continue;
              let evt: StreamEvent;
              try {
                evt = JSON.parse(json) as StreamEvent;
              } catch {
                continue;
              }
              if (evt.event === "stage") {
                setStage(evt.stage);
              } else if (evt.event === "token") {
                accumulated += evt.text;
                setStreamingText(accumulated);
              } else if (evt.event === "done") {
                final = evt;
              } else if (evt.event === "error") {
                accumulated = `Backend error: ${evt.detail}`;
                setStreamingText(accumulated);
                final = {
                  event: "done",
                  status: "refused",
                  citations: [],
                  reason: evt.detail,
                  rewritten_query: null,
                };
              }
            }
          }
        }

        // commit final message
        const finalAnswer = accumulated || (final?.status === "needs_web_search" ? "" : "(no response)");
        const refusalMsg =
          "I couldn't find a reliable answer in the available sources. Try rephrasing your question, or add more documents to this library.";
        let content = finalAnswer;
        if (final?.status === "needs_web_search") {
          content = `**Not enough info in your library to answer confidently.**\n\n_Grader's verdict: ${final.reason ?? "—"}_\n\nClick **Search the web** below to retry with web sources.`;
        } else if (final?.status === "refused") {
          content = refusalMsg;
        } else if (final?.status === "answered_from_web" && final.rewritten_query) {
          content = `_Searched the web for:_ \`${final.rewritten_query}\`\n\n${finalAnswer}`;
        }

        setChats((prev) => {
          const next = { ...prev };
          next[lib] = [
            ...(next[lib] ?? []),
            {
              role: "assistant",
              content,
              status: final?.status,
              citations: final?.citations ?? [],
              reason: final?.reason ?? null,
              rewritten: final?.rewritten_query ?? null,
            },
          ];
          return next;
        });
        if (final?.status === "needs_web_search") {
          setPendingWebQuery(q);
        }
      } catch (e) {
        setChats((prev) => {
          const next = { ...prev };
          next[lib] = [
            ...(next[lib] ?? []),
            { role: "assistant", content: `Backend error: ${(e as Error).message}` },
          ];
          return next;
        });
      } finally {
        setIsStreaming(false);
        setStreamingText("");
        setStage(null);
      }
    },
    [activeLib]
  );

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
    const wrap = (t: string, w: number) => pdf.splitTextToSize(t, w);

    for (const msg of currentChat) {
      if (y > pageHeight - 20) { pdf.addPage(); y = 20; }
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(11);
      const label = msg.role === "user" ? "You" : "Assistant";
      const status = msg.status && msg.status !== "answered" ? `  [${msg.status.replace(/_/g, " ")}]` : "";
      pdf.text(`${label}${status}`, 14, y);
      y += 6;
      pdf.setFont("helvetica", "normal");
      const plain = msg.content.replace(/[*_`]/g, "").replace(/\[\d+\]/g, "");
      for (const ln of wrap(plain, 180)) {
        if (y > pageHeight - 15) { pdf.addPage(); y = 20; }
        pdf.text(ln, 14, y); y += 5;
      }
      y += 4;
    }
    pdf.save(`${activeLib}-chat.pdf`);
  }

  // --------- render ---------

  return (
    <div className="h-screen flex flex-col bg-cream-100 text-brown-700">
      <CommandPalette
        open={paletteOpen}
        setOpen={setPaletteOpen}
        libraries={libraries}
        onPick={(name) => {
          setActiveLib(name);
        }}
      />

      <div className="flex-1 flex overflow-hidden">
        {/* sidebar */}
        <aside className="w-72 shrink-0 bg-cream-200 border-r border-cream-300 flex flex-col">
          <div className="px-6 pt-6 pb-4 border-b border-cream-300">
            <h1 className="text-xl font-serif font-semibold text-brown-700">
              Universal RAG
            </h1>
            <p className="text-xs text-brown-500 mt-1 leading-relaxed">
              Ask your PDFs, with citations.
            </p>
            <button
              onClick={() => setPaletteOpen(true)}
              className="mt-3 w-full text-left text-xs text-brown-400 hover:text-brown-700 flex items-center justify-between px-2 py-1.5 rounded-md border border-cream-300 bg-cream-50/60 hover:bg-cream-50 transition"
            >
              <span>Quick switch…</span>
              <kbd className="font-mono text-[10px] text-brown-400 border border-cream-300 px-1 py-0.5 rounded">
                ⌘K
              </kbd>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-4">
            <div className="flex items-center justify-between mb-2 px-1">
              <h2 className="text-[10px] uppercase tracking-widest text-brown-500 font-semibold">
                Libraries
              </h2>
              <motion.button
                whileTap={{ scale: 0.92 }}
                onClick={() => setShowCreate((s) => !s)}
                className="text-brown-500 hover:text-brown-700 w-6 h-6 rounded hover:bg-cream-300 transition flex items-center justify-center"
                title="Create library"
              >
                <span className="text-lg leading-none">{showCreate ? "−" : "+"}</span>
              </motion.button>
            </div>

            <AnimatePresence>
              {showCreate && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden mb-3"
                >
                  <div className="p-3 bg-cream-50 border border-cream-300 rounded-md">
                    <input
                      type="text"
                      value={newLibName}
                      placeholder="ml-papers"
                      onChange={(e) => setNewLibName(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && createLibrary()}
                      autoFocus
                      className="w-full px-2 py-1.5 text-sm bg-cream-100 border border-cream-300 rounded focus:outline-none focus:border-oxblood-500 text-brown-700"
                    />
                    {createError && (
                      <p className="text-xs text-oxblood-500 mt-1">{createError}</p>
                    )}
                    <div className="flex gap-2 mt-2">
                      <button
                        onClick={createLibrary}
                        className="flex-1 px-3 py-1.5 text-sm bg-oxblood-500 text-cream-50 rounded hover:bg-oxblood-600 transition"
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
                </motion.div>
              )}
            </AnimatePresence>

            {libraries.length === 0 && !showCreate && (
              <p className="text-xs text-brown-400 italic px-1">
                No libraries yet. Click + to create one.
              </p>
            )}

            <ul className="space-y-0.5">
              {libraries.map((lib) => (
                <li key={lib.name}>
                  <motion.button
                    whileHover={{ x: 2 }}
                    onClick={() => setActiveLib(lib.name)}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm transition relative ${
                      lib.name === activeLib
                        ? "bg-oxblood-500 text-cream-50"
                        : "text-brown-700 hover:bg-cream-300"
                    }`}
                  >
                    <div className="font-medium leading-snug">{lib.name}</div>
                    <div
                      className={`text-[11px] mt-0.5 ${
                        lib.name === activeLib ? "text-cream-200/80" : "text-brown-400"
                      }`}
                    >
                      {lib.document_count} chunks
                    </div>
                  </motion.button>
                </li>
              ))}
            </ul>
          </div>
        </aside>

        {/* main */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {!activeLib ? (
            <div className="flex-1 flex items-center justify-center p-10">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="max-w-md text-center"
              >
                <h2 className="text-4xl font-serif text-brown-700 mb-3">Welcome</h2>
                <p className="text-brown-500 leading-relaxed">
                  Create a library in the sidebar to get started. Each library
                  is its own collection of PDFs you can chat with, with cited
                  answers and an opt-in web fallback.
                </p>
              </motion.div>
            </div>
          ) : (
            <>
              <header className="px-10 pt-8 pb-4 border-b border-cream-300">
                <div className="flex items-end justify-between">
                  <motion.h2
                    key={activeLib}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="text-4xl font-serif text-brown-700 leading-none"
                  >
                    {activeLib}
                  </motion.h2>
                  {currentChat.length > 0 && (
                    <button
                      onClick={exportPdf}
                      className="text-xs text-brown-500 hover:text-brown-700 border border-cream-400 hover:border-brown-500 rounded-md px-3 py-1.5 transition"
                    >
                      Export PDF
                    </button>
                  )}
                </div>
              </header>

              <nav className="flex border-b border-cream-300 px-10">
                {(["chat", "files"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className="relative px-1 py-3 mr-8 text-sm font-medium capitalize transition"
                  >
                    <span className={tab === t ? "text-brown-700" : "text-brown-400 hover:text-brown-700"}>
                      {t}
                    </span>
                    {tab === t && (
                      <motion.span
                        layoutId="tab-underline"
                        className="absolute left-0 right-0 -bottom-px h-0.5 bg-oxblood-500"
                      />
                    )}
                  </button>
                ))}
              </nav>

              {tab === "chat" ? (
                <>
                  <div className="flex-1 overflow-y-auto px-10 py-8">
                    <div className="max-w-3xl mx-auto">
                      {currentChat.length === 0 && !isStreaming ? (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ duration: 0.4 }}
                          className="mt-8"
                        >
                          <p className="text-brown-500 italic mb-5">
                            Ask {activeLib} a question to start, or try one of these:
                          </p>
                          <div className="space-y-2">
                            {SUGGESTED_QUESTIONS.map((q) => (
                              <motion.button
                                key={q}
                                whileHover={{ x: 3 }}
                                onClick={() => sendQuery(q, false)}
                                disabled={isStreaming}
                                className="w-full text-left px-4 py-3 bg-cream-50 border border-cream-300 rounded-md text-sm text-brown-700 hover:border-oxblood-500 hover:bg-cream-50 transition disabled:opacity-50"
                              >
                                {q}
                              </motion.button>
                            ))}
                          </div>
                        </motion.div>
                      ) : (
                        <div className="space-y-6">
                          <AnimatePresence initial={false}>
                            {currentChat.map((m, i) => (
                              <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.25 }}
                              >
                                <div className="text-[10px] uppercase tracking-widest font-semibold text-brown-400 mb-1.5">
                                  {m.role === "user" ? "You" : "Assistant"}
                                </div>
                                <div className="text-brown-700 leading-relaxed text-[15px]">
                                  {m.role === "assistant" ? (
                                    <MarkdownWithCitations
                                      text={m.content}
                                      citations={m.citations ?? []}
                                    />
                                  ) : (
                                    <span dangerouslySetInnerHTML={{ __html: renderInlineMd(m.content) }} />
                                  )}
                                </div>
                                {m.role === "assistant" && (m.citations?.length ?? 0) > 0 && (
                                  <div className="mt-3 pl-4 border-l border-cream-300">
                                    <div className="text-[10px] uppercase tracking-widest font-semibold text-brown-400 mb-1.5">
                                      Sources
                                    </div>
                                    <ol className="text-xs text-brown-500 space-y-1">
                                      {m.citations!.map((c, ci) => (
                                        <li key={ci}>
                                          <span className="text-oxblood-500 font-semibold mr-1">
                                            {ci + 1}.
                                          </span>
                                          {c.page != null
                                            ? `${c.source} (p.${c.page})`
                                            : c.title
                                            ? `${c.title} — ${c.source ?? ""}`
                                            : c.source ?? "unknown"}
                                        </li>
                                      ))}
                                    </ol>
                                  </div>
                                )}
                              </motion.div>
                            ))}
                          </AnimatePresence>

                          {/* live streaming assistant turn */}
                          {isStreaming && (
                            <motion.div
                              initial={{ opacity: 0, y: 8 }}
                              animate={{ opacity: 1, y: 0 }}
                            >
                              <div className="text-[10px] uppercase tracking-widest font-semibold text-brown-400 mb-1.5">
                                Assistant
                              </div>
                              <div className="mb-3">
                                <PipelineStrip
                                  stage={stage}
                                  done={false}
                                  allowWeb={allowWebInStream}
                                />
                              </div>
                              {streamingText ? (
                                <div className="text-brown-700 leading-relaxed text-[15px]">
                                  <MarkdownWithCitations text={streamingText} citations={[]} />
                                  <span className="inline-block w-1.5 h-4 bg-oxblood-500 ml-0.5 align-middle animate-pulse" />
                                </div>
                              ) : null}
                            </motion.div>
                          )}
                          <div ref={chatEndRef} />
                        </div>
                      )}
                    </div>
                  </div>

                  {pendingWebQuery && !isStreaming && (
                    <div className="px-10 pb-3">
                      <div className="max-w-3xl mx-auto">
                        <motion.button
                          initial={{ opacity: 0, y: 4 }}
                          animate={{ opacity: 1, y: 0 }}
                          onClick={() => {
                            const q = pendingWebQuery;
                            setPendingWebQuery(null);
                            sendQuery(q, true);
                          }}
                          className="px-4 py-2 bg-oxblood-500 text-cream-50 rounded-md hover:bg-oxblood-600 transition text-sm font-medium"
                        >
                          Search the web instead
                        </motion.button>
                      </div>
                    </div>
                  )}

                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      if (!query.trim() || isStreaming) return;
                      const q = query.trim();
                      setQuery("");
                      sendQuery(q, false);
                    }}
                    className="border-t border-cream-300 px-10 py-5"
                  >
                    <div className="max-w-3xl mx-auto flex gap-3 items-center">
                      <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder={`Ask ${activeLib}…`}
                        disabled={isStreaming}
                        className="flex-1 px-4 py-3 bg-cream-50 border border-cream-300 rounded-md focus:outline-none focus:border-oxblood-500 text-brown-700 placeholder:text-brown-400 transition"
                      />
                      <motion.button
                        whileTap={{ scale: 0.96 }}
                        type="submit"
                        disabled={isStreaming || !query.trim()}
                        className="px-5 py-3 bg-oxblood-500 text-cream-50 rounded-md hover:bg-oxblood-600 transition disabled:opacity-40 font-medium text-sm"
                      >
                        Send
                      </motion.button>
                    </div>
                  </form>
                </>
              ) : (
                <div className="flex-1 overflow-y-auto px-10 py-8">
                  <div className="max-w-3xl mx-auto">
                    <section className="mb-10">
                      <h3 className="text-xs uppercase tracking-widest text-brown-400 font-semibold mb-3">
                        Upload
                      </h3>
                      <motion.label
                        whileHover={{ scale: uploading ? 1 : 1.005 }}
                        className={`block border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition ${
                          uploading
                            ? "border-cream-300 bg-cream-50 opacity-50"
                            : "border-cream-400 bg-cream-50 hover:border-oxblood-500"
                        }`}
                      >
                        <input
                          type="file"
                          accept=".pdf"
                          disabled={uploading}
                          className="hidden"
                          onChange={(e) => {
                            const f = e.target.files?.[0];
                            if (f) uploadPdf(f);
                            e.target.value = "";
                          }}
                        />
                        <p className="text-brown-500 text-sm">
                          {uploading ? "Ingesting…" : "Click or drop a PDF here"}
                        </p>
                        <p className="text-[11px] text-brown-400 mt-1">PDF · max 10MB</p>
                      </motion.label>
                      {uploadMsg && (
                        <motion.p
                          initial={{ opacity: 0, y: -4 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={`mt-2 text-sm ${uploadMsg.ok ? "text-brown-500" : "text-oxblood-500"}`}
                        >
                          {uploadMsg.text}
                        </motion.p>
                      )}
                    </section>

                    <section>
                      <h3 className="text-xs uppercase tracking-widest text-brown-400 font-semibold mb-3">
                        Files ({files.length})
                      </h3>
                      {files.length === 0 ? (
                        <p className="text-sm text-brown-400 italic">
                          No files yet. Upload one above.
                        </p>
                      ) : (
                        <ul className="divide-y divide-cream-300 border border-cream-300 rounded-md overflow-hidden">
                          {files.map((f) => (
                            <motion.li
                              key={f.filename}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              className="flex items-center justify-between p-4 bg-cream-50 hover:bg-cream-100 transition"
                            >
                              <span className="font-medium text-brown-700 text-sm">{f.filename}</span>
                              <span className="text-[11px] text-brown-400">{f.chunk_count} chunks</span>
                            </motion.li>
                          ))}
                        </ul>
                      )}
                    </section>
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
