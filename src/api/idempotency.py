import time
import threading
from typing import Optional, Dict, Any

class IdempotencyStore:
    """
    Thread-safe in-memory TTL store for API Idempotency Keys (DDIA pattern).
    Stores cached JSON responses keyed by X-Idempotency-Key.
    """
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key not in self._store:
                return None
            item = self._store[key]
            if time.time() > item['expires_at']:
                del self._store[key]
                return None
            return item['data']

    def set(self, key: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = {
                'data': data,
                'expires_at': time.time() + self.ttl_seconds
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

idempotency_store = IdempotencyStore()
