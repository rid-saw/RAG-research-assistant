import os
from pathlib import Path

import gradio as gr
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = ["", "**Sources:**"]
    for i, c in enumerate(citations, start=1):
        if c.get("page") is not None:
            label = f"{c['source']} (p.{c['page']})"
        elif c.get("title"):
            label = f"{c['title']} — {c['source']}"
        else:
            label = c.get("source") or "unknown"
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


def _format_bot_message(data: dict) -> str:
    status = data.get("status", "")
    answer = data.get("answer", "")
    reason = data.get("reason")
    rewritten = data.get("rewritten_query")
    citations_md = _format_citations(data.get("citations", []))

    if status == "answered":
        return f"{answer}\n{citations_md}"

    if status == "needs_web_search":
        return (
            f"⚠️ Not enough info in your library to answer confidently.\n\n"
            f"**Grader's verdict:** {reason}\n\n"
            f"Click **Search the web** below to retry with web sources."
        )

    if status == "answered_from_web":
        prefix = f"🔍 *Searched the web for:* `{rewritten}`\n\n" if rewritten else ""
        return f"{prefix}{answer}\n{citations_md}"

    if status == "refused":
        prefix = f"🔍 *Tried searching the web for:* `{rewritten}`\n\n" if rewritten else ""
        return f"{prefix}{answer}\n{citations_md}"

    return answer or "(no response)"


def list_libraries() -> list[str]:
    try:
        r = requests.get(f"{BACKEND_URL}/api/libraries", timeout=10)
        r.raise_for_status()
        return [lib["name"] for lib in r.json().get("libraries", [])]
    except Exception:
        return []


def create_library(name: str):
    name = (name or "").strip()
    if not name:
        return "Please enter a library name.", gr.update()
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/libraries",
            json={"name": name},
            timeout=30,
        )
        r.raise_for_status()
    except requests.HTTPError:
        detail = r.json().get("detail", "Unknown error")
        return f"Error: {detail}", gr.update()
    except Exception as e:
        return f"Backend unreachable: {e}", gr.update()

    libs = list_libraries()
    return f"Created library `{name}`.", gr.update(choices=libs, value=name)


def upload_pdf(library: str, file_path: str | None):
    if not library:
        return "Pick or create a library first."
    if not file_path:
        return "No file selected."
    filename = Path(file_path).name
    try:
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/api/libraries/{library}/documents",
                files={"file": (filename, f, "application/pdf")},
                timeout=TIMEOUT,
            )
        r.raise_for_status()
    except requests.HTTPError:
        detail = r.json().get("detail", "Unknown error")
        return f"Error: {detail}"
    except Exception as e:
        return f"Backend unreachable: {e}"
    data = r.json()
    return f"Ingested **{data['filename']}**: {data['pages']} pages → {data['chunks_added']} chunks."


def _ask(library: str, query: str, allow_web_search: bool) -> dict:
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
    except requests.HTTPError:
        detail = r.json().get("detail", "Unknown error")
        return {"status": "refused", "answer": f"Backend error: {detail}", "citations": [], "reason": None, "rewritten_query": None}
    except Exception as e:
        return {"status": "refused", "answer": f"Backend unreachable: {e}", "citations": [], "reason": None, "rewritten_query": None}


def on_send(library: str, query: str, history: list, last_query_state: str):
    if not query or not query.strip():
        return history, "", last_query_state, gr.update(visible=False)

    history = history + [{"role": "user", "content": query}]

    if not library:
        history = history + [{"role": "assistant", "content": "Pick or create a library first."}]
        return history, "", last_query_state, gr.update(visible=False)

    data = _ask(library, query, allow_web_search=False)
    history = history + [{"role": "assistant", "content": _format_bot_message(data)}]
    show_web = data.get("status") == "needs_web_search"
    return history, "", query, gr.update(visible=show_web)


def on_search_web(library: str, history: list, last_query_state: str):
    if not last_query_state:
        return history, gr.update(visible=False)
    data = _ask(library, last_query_state, allow_web_search=True)
    history = history + [{"role": "assistant", "content": _format_bot_message(data)}]
    return history, gr.update(visible=False)


with gr.Blocks(title="Universal RAG") as app:
    gr.Markdown(
        "# Universal RAG\n"
        "Upload PDFs to a library, then ask questions. If your library can't answer, "
        "you'll be prompted to search the web."
    )

    with gr.Row():
        library_dd = gr.Dropdown(
            choices=list_libraries(),
            label="Library",
            interactive=True,
            scale=2,
        )
        new_lib_name = gr.Textbox(label="New library name", placeholder="ml-papers", scale=2)
        create_lib_btn = gr.Button("Create library", scale=1)
    create_lib_status = gr.Markdown("")

    with gr.Row():
        upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
    upload_status = gr.Markdown("")

    chatbot = gr.Chatbot(height=420, label="Chat")
    last_query = gr.State("")

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask a question...", scale=4, show_label=False)
        send_btn = gr.Button("Send", scale=1, variant="primary")

    web_search_btn = gr.Button("🔍 Search the web", visible=False, variant="secondary")

    create_lib_btn.click(
        create_library,
        inputs=[new_lib_name],
        outputs=[create_lib_status, library_dd],
    )
    upload.upload(
        upload_pdf,
        inputs=[library_dd, upload],
        outputs=[upload_status],
    )
    send_btn.click(
        on_send,
        inputs=[library_dd, msg, chatbot, last_query],
        outputs=[chatbot, msg, last_query, web_search_btn],
    )
    msg.submit(
        on_send,
        inputs=[library_dd, msg, chatbot, last_query],
        outputs=[chatbot, msg, last_query, web_search_btn],
    )
    web_search_btn.click(
        on_search_web,
        inputs=[library_dd, chatbot, last_query],
        outputs=[chatbot, web_search_btn],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
