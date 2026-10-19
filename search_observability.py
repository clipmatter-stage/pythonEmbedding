"""Per-request provider counters; no prompts, credentials or response text."""
from contextlib import contextmanager
from contextvars import ContextVar
_events=ContextVar('embedding_events',default=None)
def record(event):
    target=_events.get()
    if target is not None:target.append(event)
@contextmanager
def capture_embeddings():
    events=[];token=_events.set(events)
    try:yield events
    finally:_events.reset(token)
