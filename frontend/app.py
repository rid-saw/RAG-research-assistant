import os

import requests
import streamlit as st

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


# ---------- page config ----------

st.set_page_config(
    page_title="Universal RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
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
    st.title("📚 Universal RAG")
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

    if st.button("➕ Create library", use_container_width=True):
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

    # Upload only after a library is selected
    if st.session_state.current_library:
        st.divider()
        st.subheader(f"Upload to `{st.session_state.current_library}`")
        uploaded = st.file_uploader(
            "Drop a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"upload_{st.session_state.current_library}",
        )
        if uploaded is not None:
            with st.spinner(f"Ingesting {uploaded.name}..."):
                ok, msg = upload_pdf_api(
                    st.session_state.current_library,
                    uploaded.getvalue(),
                    uploaded.name,
                )
            if ok:
                st.success(msg)
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
    st.header(f"💬 {lib}")

    chat = current_chat()

    if not chat:
        st.info("Ask a question below to start chatting with this library.")

    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Web search button surfaces only when last bot reply asked for it
    if st.session_state.pending_web_query:
        if st.button("🌐 Search the web", type="primary"):
            query = st.session_state.pending_web_query
            with st.spinner("Searching the web..."):
                data = ask_api(lib, query, allow_web_search=True)
            chat.append({"role": "assistant", "content": format_bot_message(data)})
            st.session_state.last_citations = data.get("citations", [])
            st.session_state.pending_web_query = None
            st.rerun()

    # Chat input (sticky bottom)
    query = st.chat_input(f"Ask {lib} a question...")
    if query:
        chat.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            data = ask_api(lib, query, allow_web_search=False)
        chat.append({"role": "assistant", "content": format_bot_message(data)})
        st.session_state.last_citations = data.get("citations", [])
        st.session_state.pending_web_query = query if data.get("status") == "needs_web_search" else None
        st.rerun()

    # ---------- sources ----------
    st.divider()
    with st.expander(
        f"📎 Sources ({len(st.session_state.last_citations)})",
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
