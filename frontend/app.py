import os
from datetime import datetime

import requests
import streamlit as st
from fpdf import FPDF

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120


# ---------- backend client ----------

def list_libraries() -> list[str]:
    try:
        r = requests.get(f"{BACKEND_URL}/api/libraries", timeout=10)
        r.raise_for_status()
        return [lib["name"] for lib in r.json().get("libraries", [])]
    except Exception:
        return []


def create_library_api(name: str) -> tuple[bool, str]:
    try:
        r = requests.post(f"{BACKEND_URL}/api/libraries", json={"name": name}, timeout=30)
        r.raise_for_status()
        return True, f"Created `{name}`."
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", "Unknown error")
        except Exception:
            detail = e.response.text
        return False, f"Error: {detail}"
    except Exception as e:
        return False, f"Backend unreachable: {e}"


def upload_pdf_api(library: str, file_bytes: bytes, filename: str) -> tuple[bool, str]:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/libraries/{library}/documents",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return True, f"Ingested {data['filename']}: {data['pages']} pages → {data['chunks_added']} chunks."
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", "Unknown error")
        except Exception:
            detail = e.response.text
        return False, f"Error: {detail}"
    except Exception as e:
        return False, f"Backend unreachable: {e}"


def list_library_files(library: str) -> list[dict]:
    try:
        r = requests.get(f"{BACKEND_URL}/api/libraries/{library}/files", timeout=10)
        r.raise_for_status()
        return r.json().get("files", [])
    except Exception:
        return []


def ask_api(library: str, query: str, allow_web_search: bool) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "query": query,
                "collection": library,
                "allow_web_search": allow_web_search,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", "Unknown error")
        except Exception:
            detail = e.response.text
        return {"status": "refused", "answer": f"Backend error: {detail}", "citations": [], "reason": None, "rewritten_query": None}
    except Exception as e:
        return {"status": "refused", "answer": f"Backend unreachable: {e}", "citations": [], "reason": None, "rewritten_query": None}


# ---------- formatters ----------

def format_bot_message(data: dict) -> str:
    status = data.get("status", "")
    answer = data.get("answer", "")
    reason = data.get("reason")
    rewritten = data.get("rewritten_query")

    if status == "answered":
        return answer
    if status == "needs_web_search":
        return (
            f"**Not enough info in your library to answer confidently.**\n\n"
            f"_Grader's verdict: {reason}_\n\n"
            f"Click **Search the web** below to retry with web sources."
        )
    if status == "answered_from_web":
        prefix = f"_Searched the web for:_ `{rewritten}`\n\n" if rewritten else ""
        return f"{prefix}{answer}"
    if status == "refused":
        prefix = f"_Tried searching the web for:_ `{rewritten}`\n\n" if rewritten else ""
        return f"{prefix}{answer}"
    return answer or "(no response)"


# ---------- PDF export ----------

def _latin1_safe(s: str) -> str:
    repl = {
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "—": "--", "–": "-",
        "…": "...", " ": " ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def build_chat_pdf(library: str, chat: list[dict], citations: list[dict]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _latin1_safe(f"Chat - {library}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)

    for msg in chat:
        role = msg.get("role", "")
        content = _latin1_safe(msg.get("content", ""))
        status = msg.get("status")

        label = "You" if role == "user" else "Assistant"
        if status and status != "answered":
            label = f"{label}  [{status.replace('_', ' ')}]"

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 5, content, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    if citations:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Sources (most recent answer)", new_x="LMARGIN", new_y="NEXT")
        for i, c in enumerate(citations, 1):
            if c.get("page") is not None:
                head = f"{i}. {c.get('source', 'unknown')} (p.{c['page']})"
            elif c.get("title"):
                head = f"{i}. {c['title']} - {c.get('source', '')}"
            else:
                head = f"{i}. {c.get('source', 'unknown')}"
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 5, _latin1_safe(head), new_x="LMARGIN", new_y="NEXT")
            snippet = (c.get("snippet") or "").replace("\n", " ").strip()
            if snippet:
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(110, 110, 110)
                pdf.multi_cell(0, 5, _latin1_safe(snippet[:300]), new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

    return bytes(pdf.output())


# ---------- page config ----------

st.set_page_config(
    page_title="Universal RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      html, body, [class*="css"] {
        font-family: 'Lora', Georgia, 'Times New Roman', serif;
        color: #3a2f24;
      }
      .stApp { background-color: #faf7f2; }
      section[data-testid="stSidebar"] {
        background-color: #f0e9da;
        border-right: 1px solid #e8dfcc;
      }
      h1, h2, h3, h4 {
        font-family: 'Lora', Georgia, serif;
        color: #3a2f24;
        letter-spacing: -0.01em;
      }
      .stButton button,
      .stDownloadButton button {
        background-color: #faf7f2;
        color: #3a2f24;
        border: 1px solid #c9b896;
        border-radius: 6px;
        box-shadow: none;
        font-weight: 500;
      }
      .stButton button:hover,
      .stDownloadButton button:hover {
        background-color: #ede2cd;
        border-color: #9a8567;
        color: #3a2f24;
      }
      .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #e8dfcc; }
      .stTabs [data-baseweb="tab"] {
        font-family: 'Lora', Georgia, serif;
        color: #6e5640;
        padding-bottom: 8px;
      }
      .stTabs [aria-selected="true"] {
        color: #3a2f24 !important;
        border-bottom: 2px solid #9a8567 !important;
      }
      div[data-testid="stChatMessage"] {
        background-color: #fefcf6;
        border: 1px solid #ece2cb;
        border-radius: 8px;
        padding: 12px 16px;
      }
      div[data-testid="stChatInput"] textarea {
        background-color: #fefcf6;
        border: 1px solid #c9b896;
        color: #3a2f24;
      }
      .stTextInput input,
      .stFileUploader,
      .stExpander {
        background-color: #fefcf6;
        border-color: #e8dfcc !important;
      }
      a, a:visited { color: #6e5640; }
      code { background-color: #ede2cd; color: #3a2f24; padding: 1px 5px; border-radius: 3px; }
      blockquote {
        border-left: 2px solid #9a8567;
        padding-left: 12px;
        color: #6e5640;
      }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)


# ---------- session state ----------

if "libraries" not in st.session_state:
    st.session_state.libraries = list_libraries()
if "current_library" not in st.session_state:
    st.session_state.current_library = None
if "chats" not in st.session_state:
    st.session_state.chats = {}  # library_name -> list[{"role", "content"}]
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "show_create" not in st.session_state:
    st.session_state.show_create = False
if "pending_web_query" not in st.session_state:
    st.session_state.pending_web_query = None


def current_chat() -> list[dict]:
    lib = st.session_state.current_library
    if not lib:
        return []
    return st.session_state.chats.setdefault(lib, [])


# ---------- sidebar ----------

with st.sidebar:
    st.title("Universal RAG")
    st.caption("Upload PDFs, ask questions, get cited answers.")
    st.divider()

    st.subheader("Libraries")

    if not st.session_state.libraries:
        st.info("No libraries yet. Create one to get started.")
    else:
        selected = st.radio(
            "Active library",
            options=st.session_state.libraries,
            index=(
                st.session_state.libraries.index(st.session_state.current_library)
                if st.session_state.current_library in st.session_state.libraries
                else 0
            ),
            label_visibility="collapsed",
        )
        if selected != st.session_state.current_library:
            st.session_state.current_library = selected
            st.session_state.last_citations = []
            st.session_state.pending_web_query = None
            st.rerun()

    if st.button("Create library", use_container_width=True):
        st.session_state.show_create = not st.session_state.show_create

    if st.session_state.show_create:
        new_name = st.text_input(
            "Name",
            placeholder="ml-papers",
            key="new_lib_name",
            label_visibility="collapsed",
        )
        if st.button("Create", use_container_width=True, key="create_lib_submit"):
            name = (new_name or "").strip()
            if not name:
                st.warning("Please enter a name.")
            else:
                ok, msg = create_library_api(name)
                if ok:
                    st.success(msg)
                    st.session_state.libraries = list_libraries()
                    st.session_state.current_library = name
                    st.session_state.show_create = False
                    st.rerun()
                else:
                    st.error(msg)

# ---------- main ----------

if not st.session_state.current_library:
    st.header("Welcome to Universal RAG")
    st.markdown(
        "Create a library in the sidebar to get started.\n\n"
        "Each library is its own collection of PDFs you can chat with. "
        "Answers come grounded with source citations, and you can opt into web search when your library doesn't have enough info."
    )
else:
    lib = st.session_state.current_library
    chat = current_chat()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(lib)
    with col2:
        if chat:
            pdf_bytes = build_chat_pdf(lib, chat, st.session_state.last_citations)
            st.download_button(
                "Export PDF",
                data=pdf_bytes,
                file_name=f"{lib}-chat.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    chat_tab, files_tab = st.tabs(["Chat", "Files"])

    # ---------- chat tab ----------
    with chat_tab:
        if not chat:
            st.info("Ask a question below to start chatting with this library.")

        for msg in chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.pending_web_query:
            if st.button("Search the web", type="primary"):
                query = st.session_state.pending_web_query
                with st.spinner("Searching the web..."):
                    data = ask_api(lib, query, allow_web_search=True)
                chat.append({"role": "assistant", "content": format_bot_message(data), "status": data.get("status")})
                st.session_state.last_citations = data.get("citations", [])
                st.session_state.pending_web_query = None
                st.rerun()

        st.divider()
        with st.expander(
            f"Sources ({len(st.session_state.last_citations)})",
            expanded=False,
        ):
            citations = st.session_state.last_citations
            if not citations:
                st.caption("Sources for the most recent answer will appear here.")
            else:
                for i, c in enumerate(citations, start=1):
                    if c.get("page") is not None:
                        label = f"`{c['source']}` (p.{c['page']})"
                    elif c.get("title"):
                        label = f"**{c['title']}** — `{c['source']}`"
                    else:
                        label = f"`{c.get('source') or 'unknown'}`"
                    snippet = (c.get("snippet") or "").replace("\n", " ").strip()
                    if len(snippet) > 240:
                        snippet = snippet[:237] + "..."
                    with st.container(border=True):
                        st.markdown(f"**{i}.** {label}")
                        if snippet:
                            st.caption(snippet)

    # ---------- files tab ----------
    with files_tab:
        st.subheader("Upload a PDF")
        uploaded = st.file_uploader(
            "Drop a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"upload_{lib}",
        )
        if uploaded is not None:
            with st.spinner(f"Ingesting {uploaded.name}..."):
                ok, msg = upload_pdf_api(lib, uploaded.getvalue(), uploaded.name)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.divider()

        files = list_library_files(lib)
        st.subheader(f"Files in `{lib}` ({len(files)})")
        if not files:
            st.caption("No files yet. Upload a PDF above to get started.")
        else:
            for f in files:
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{f['filename']}**")
                    with c2:
                        st.caption(f"{f['chunk_count']} chunks")

    # ---------- chat input (sticky bottom, outside tabs so always visible) ----------
    query = st.chat_input(f"Ask {lib} a question...")
    if query:
        chat.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            data = ask_api(lib, query, allow_web_search=False)
        chat.append({"role": "assistant", "content": format_bot_message(data), "status": data.get("status")})
        st.session_state.last_citations = data.get("citations", [])
        st.session_state.pending_web_query = query if data.get("status") == "needs_web_search" else None
        st.rerun()
