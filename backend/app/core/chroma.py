"""Single shared chromadb client.

chromadb's PersistentClient has a thread-safety bug: when multiple
PersistentClient instances are constructed for the same path in parallel
(e.g. several /api/libraries* calls arriving on the same first frontend
load), the shared-system registry races and throws KeyError on its own
internal lookup.

functools.lru_cache isn't enough here — Python's lru_cache serialises
*storage* but lets multiple threads execute the wrapped function
concurrently on a cache miss. So we use an explicit double-checked lock
to guarantee chromadb.PersistentClient is only ever called once.
"""

import threading

import chromadb

from app.core.config import settings

_lock = threading.Lock()
_client: chromadb.api.ClientAPI | None = None


def get_chroma_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client
