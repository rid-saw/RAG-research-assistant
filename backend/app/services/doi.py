"""Resolve a DOI to an open-access PDF.

Strategy:
  1. arXiv DOIs (10.48550/arXiv.X) get a direct fast path to arxiv.org.
  2. Everything else hits Unpaywall, which returns the best open-access
     PDF location for a given DOI. Unpaywall is free and only requires
     a contact email in the request.

Raises DOIError if no open-access PDF could be found. The error carries
the paper metadata (title, authors) when known so the UI can show
"Found 'Attention Is All You Need' but no OA PDF is available."
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

import requests

from app.core.config import settings

ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
USER_AGENT = "RAG-Research-Assistant/1.0 (https://github.com/)"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


class DOIError(Exception):
    def __init__(self, message: str, metadata: dict | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


def _normalise_doi(raw: str) -> str:
    s = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\s\-\.]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:120] or "paper"


def _download(url: str) -> bytes:
    r = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT}, allow_redirects=True)
    r.raise_for_status()
    if not r.content:
        raise DOIError("Downloaded PDF was empty.")
    return r.content


def _arxiv_metadata(arxiv_id: str) -> dict:
    """Title + author list from arxiv.org's Atom API."""
    try:
        r = requests.get(
            f"http://export.arxiv.org/api/query?id_list={arxiv_id}",
            timeout=10,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        root = ET.fromstring(r.text)
        entry = root.find("a:entry", ARXIV_NS)
        if entry is None:
            raise ValueError("no entry")
        title_el = entry.find("a:title", ARXIV_NS)
        title = (title_el.text or "").strip() if title_el is not None else arxiv_id
        authors = [
            (a.find("a:name", ARXIV_NS).text or "").strip()
            for a in entry.findall("a:author", ARXIV_NS)
            if a.find("a:name", ARXIV_NS) is not None
        ]
        return {"title": title, "authors": authors, "doi": f"10.48550/arXiv.{arxiv_id}"}
    except Exception:
        return {"title": arxiv_id, "authors": [], "doi": f"10.48550/arXiv.{arxiv_id}"}


def fetch_doi_pdf(raw_doi: str) -> tuple[bytes, str, dict]:
    """Resolve a DOI to (pdf_bytes, filename, metadata).

    Raises DOIError on lookup failure. If the DOI was valid but no OA PDF
    is available, the error carries the metadata so the UI can be helpful.
    """
    doi = _normalise_doi(raw_doi)
    if not doi or "/" not in doi:
        raise DOIError("That doesn't look like a DOI (try 10.xxxx/yyyy).")

    # 1) arXiv fast path
    m = ARXIV_DOI_RE.match(doi)
    if m:
        arxiv_id = m.group(1)
        meta = _arxiv_metadata(arxiv_id)
        try:
            pdf_bytes = _download(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        except Exception as e:
            raise DOIError(f"Could not download from arXiv: {e}", meta) from e
        filename = _sanitize_filename(meta.get("title") or arxiv_id) + ".pdf"
        return pdf_bytes, filename, meta

    # 2) Unpaywall
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": settings.unpaywall_email},
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.RequestException as e:
        raise DOIError(f"Could not reach Unpaywall: {e}") from e

    if r.status_code == 404:
        raise DOIError(f"DOI '{doi}' was not recognised.")
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        raise DOIError(f"Unpaywall error ({r.status_code}): {e}") from e

    data = r.json()
    title = (data.get("title") or "").strip()
    authors = [
        " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
        for a in (data.get("z_authors") or [])
    ]
    authors = [a for a in authors if a]
    meta = {"title": title, "authors": authors, "doi": doi}

    oa = data.get("best_oa_location") or {}
    pdf_url = oa.get("url_for_pdf") or oa.get("url")
    if not pdf_url:
        raise DOIError("No open-access PDF found for this DOI.", meta)

    try:
        pdf_bytes = _download(pdf_url)
    except Exception as e:
        raise DOIError(f"Could not download PDF: {e}", meta) from e

    filename = (
        _sanitize_filename(title) + ".pdf"
        if title
        else doi.replace("/", "-") + ".pdf"
    )
    return pdf_bytes, filename, meta
