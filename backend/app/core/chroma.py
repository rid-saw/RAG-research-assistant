"""Single cached chromadb client.

chromadb's PersistentClient has a thread-safety issue: when multiple
PersistentClient instances are created for the same path concurrently
(e.g. the frontend firing several /api/libraries calls in parallel),
the shared-system registry races and throws KeyError.

Workaround: instantiate one client at module level and reuse it
everywhere. Also hand the same client to langchain's Chroma wrapper so
they don't open competing instances.
"""

from functools import lru_cache

import chromadb

from app.core.config import settings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)
