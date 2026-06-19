import os

import requests
from nicegui import ui

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
            f"Grader's verdict: *{reason}*\n\n"
            f"Click **Search the web** below to retry with web sources."
        )
    if status == "answered_from_web":
        prefix = f"*Searched the web for:* `{rewritten}`\n\n" if rewritten else ""
        return f"{prefix}{answer}"
    if status == "refused":
        prefix = f"*Tried searching the web for:* `{rewritten}`\n\n" if rewritten else ""
        return f"{prefix}{answer}"
    return answer or "(no response)"


# ---------- state ----------

state: dict = {
    "current_library": None,
    "chat_history": [],
    "last_query": "",
    "last_citations": [],
    "show_web_button": False,
}


# ---------- refreshable views ----------

@ui.refreshable
def render_chat():
    if not state["chat_history"]:
        ui.label("Ask a question to start.").classes("italic text-stone-500 p-12 text-center w-full")
        return
    for msg in state["chat_history"]:
        with ui.row().classes("w-full mb-3"):
            if msg["role"] == "user":
                ui.element("div").classes("flex-1")
                with ui.card().tight().classes("max-w-2xl px-6 py-4").style(
                    "background-color: #e6dec9; border-color: #d8d0bb;"
                ):
                    ui.markdown(msg["content"])
            else:
                with ui.card().tight().classes("max-w-2xl px-6 py-4").style(
                    "background-color: #fdfbf7; border-color: #e6dec9;"
                ):
                    ui.markdown(msg["content"])
                ui.element("div").classes("flex-1")


@ui.refreshable
def render_sources():
    citations = state["last_citations"]
    ui.label("Sources for last answer").classes("text-2xl font-semibold heading-serif mb-2")
    if not citations:
        ui.label("No sources yet — ask a question above.").classes("italic text-stone-500")
        return
    for i, c in enumerate(citations, start=1):
        if c.get("page") is not None:
            label = f"`{c['source']}` (p.{c['page']})"
        elif c.get("title"):
            label = f"**{c['title']}** — `{c['source']}`"
        else:
            label = f"`{c.get('source') or 'unknown'}`"
        snippet = (c.get("snippet") or "").replace("\n", " ").strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        with ui.card().classes("w-full p-5").style("background-color: #fdfbf7; border: 1px solid #e6dec9;"):
            ui.markdown(f"**{i}.** {label}")
            if snippet:
                ui.markdown(f"> {snippet}").classes("text-stone-600")


@ui.refreshable
def render_web_button():
    if state["show_web_button"]:
        ui.button("Search the web", on_click=on_search_web).props("color=primary outline").classes("mt-2")


# ---------- event handlers ----------

def refresh_library_select():
    libs = list_libraries()
    current = state["current_library"] if state["current_library"] in libs else (libs[0] if libs else None)
    library_select.set_options(libs, value=current)
    state["current_library"] = current


def on_create_library():
    name = (new_lib_input.value or "").strip()
    if not name:
        ui.notify("Please enter a library name.", type="warning")
        return
    ok, msg = create_library_api(name)
    if ok:
        ui.notify(msg)
        new_lib_input.value = ""
        state["current_library"] = name
        refresh_library_select()
    else:
        ui.notify(msg, type="negative")


def on_upload(e):
    library = state["current_library"]
    if not library:
        ui.notify("Pick or create a library first.", type="warning")
        return
    file_bytes = e.content.read()
    ok, msg = upload_pdf_api(library, file_bytes, e.name)
    ui.notify(msg, type="positive" if ok else "negative")


def on_send():
    query = (msg_input.value or "").strip()
    if not query:
        return
    library = state["current_library"]
    if not library:
        ui.notify("Pick or create a library first.", type="warning")
        return

    state["chat_history"].append({"role": "user", "content": query})
    msg_input.value = ""
    render_chat.refresh()

    data = ask_api(library, query, allow_web_search=False)
    state["chat_history"].append({"role": "assistant", "content": format_bot_message(data)})
    state["last_citations"] = data.get("citations", [])
    state["last_query"] = query
    state["show_web_button"] = data.get("status") == "needs_web_search"

    render_chat.refresh()
    render_sources.refresh()
    render_web_button.refresh()


def on_search_web():
    library = state["current_library"]
    query = state["last_query"]
    if not query or not library:
        return
    data = ask_api(library, query, allow_web_search=True)
    state["chat_history"].append({"role": "assistant", "content": format_bot_message(data)})
    state["last_citations"] = data.get("citations", [])
    state["show_web_button"] = False
    render_chat.refresh()
    render_sources.refresh()
    render_web_button.refresh()


def on_library_change(e):
    state["current_library"] = e.value


# ---------- page ----------

ui.colors(primary="#8b7e60", secondary="#d8d0bb", accent="#c4bba2")

ui.add_head_html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Lora:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  body {
    background-color: #f5f0e6;
    color: #2a2520;
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 15px;
    line-height: 1.65;
  }
  .nicegui-content { background-color: #f5f0e6; }
  .heading-serif {
    font-family: 'Lora', Georgia, serif !important;
    letter-spacing: -0.01em;
    color: #2a2520;
  }
  h1, h2, h3, h4, h5 {
    font-family: 'Lora', Georgia, serif;
    color: #2a2520;
    letter-spacing: -0.01em;
  }
  .q-card {
    background-color: #fdfbf7;
    border: 1px solid #e6dec9;
    border-radius: 12px;
    box-shadow: none !important;
  }
  .q-field__control {
    background-color: #fdfbf7 !important;
    border-radius: 12px !important;
  }
  .q-field--outlined .q-field__control:before {
    border-color: #e6dec9 !important;
    border-radius: 12px !important;
  }
  .q-field--outlined.q-field--focused .q-field__control:after {
    border-color: #8b7e60 !important;
  }
  .q-btn {
    font-weight: 500;
    text-transform: none;
    border-radius: 12px;
    padding: 10px 22px;
    box-shadow: none !important;
    letter-spacing: 0;
  }
  .q-btn:before { box-shadow: none !important; }
  .q-uploader {
    background-color: #fdfbf7;
    border: 1px dashed #c4bba2;
    border-radius: 12px;
    box-shadow: none !important;
  }
  .q-uploader__header {
    background-color: #e6dec9 !important;
    color: #2a2520 !important;
    border-top-left-radius: 11px;
    border-top-right-radius: 11px;
  }
  .q-separator { background-color: #e6dec9 !important; }
  blockquote {
    border-left: 2px solid #c4bba2;
    padding-left: 12px;
    color: #6b6256;
    margin: 8px 0;
  }
  code {
    background-color: #f0ebe0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.92em;
  }
</style>
""")

_initial_libs = list_libraries()
if _initial_libs:
    state["current_library"] = _initial_libs[0]

with ui.row().classes("w-full max-w-7xl mx-auto p-10 gap-12 items-start"):
    # Sidebar
    with ui.column().classes("w-80 gap-6"):
        ui.label("Universal RAG").classes("text-4xl font-bold heading-serif")
        ui.label("Upload PDFs, ask questions, get cited answers.").classes("text-sm text-stone-600 leading-relaxed")

        ui.label("Upload").classes("text-2xl font-semibold heading-serif mt-6")
        ui.upload(
            on_upload=on_upload,
            auto_upload=True,
            label="Drop a PDF",
        ).props('accept=".pdf" flat').classes("w-full")

        ui.label("Library").classes("text-2xl font-semibold heading-serif mt-8")
        library_select = ui.select(
            options=_initial_libs,
            value=state["current_library"],
            label="Active library",
            on_change=on_library_change,
        ).classes("w-full")
        new_lib_input = ui.input(label="New library", placeholder="ml-papers").classes("w-full mt-3")
        ui.button("Create library", on_click=on_create_library).props("color=primary").classes("w-full mt-3")

    # Main chat
    with ui.column().classes("flex-1 min-w-0 gap-4"):
        with ui.card().classes("w-full min-h-[600px] p-6").style("background-color: #fdfbf7;"):
            render_chat()

        with ui.row().classes("w-full gap-3 items-center mt-2"):
            msg_input = (
                ui.input(placeholder="Ask a question...")
                .classes("flex-1")
                .on("keydown.enter", on_send)
            )
            ui.button("Send", on_click=on_send).props("color=primary")

        render_web_button()

ui.separator().classes("my-8")
with ui.column().classes("w-full max-w-7xl mx-auto px-10 pb-12 gap-4"):
    render_sources()


ui.run(host="127.0.0.1", port=7860, title="Universal RAG", dark=False, show=False, reload=False)
