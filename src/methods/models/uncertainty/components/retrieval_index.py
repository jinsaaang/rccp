from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def as_float32_matrix(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return values.reshape(values.shape[0], -1).astype(np.float32)


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = as_float32_matrix(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
    return (values / norms).astype(np.float32)


def _next_capacity(current: int, needed: int) -> int:
    capacity = max(int(current), 64)
    while capacity < int(needed):
        capacity *= 2
    return capacity


@dataclass
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray


class RetrievalIndex:
    def fit(self, keys: np.ndarray):
        raise NotImplementedError

    def add(self, keys: np.ndarray):
        raise NotImplementedError

    def search(self, query_keys: np.ndarray, k: int) -> SearchResult:
        raise NotImplementedError

    def range_search(self, query_keys: np.ndarray, radius: float) -> SearchResult:
        raise NotImplementedError("range_search is not implemented for this backend")

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


class ExactRetrievalIndex(RetrievalIndex):
    def __init__(self, metric: str = "cosine"):
        self.metric = str(metric).lower()
        if self.metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported retrieval metric: {metric}")
        self.keys: Optional[np.ndarray] = None
        self._keys_cos: Optional[np.ndarray] = None
        self._size = 0

    def _active_keys(self) -> np.ndarray:
        if self.keys is None:
            raise ValueError("Retrieval index is empty")
        return self.keys[: self._size]

    def _active_keys_cos(self) -> np.ndarray:
        if self._keys_cos is None:
            raise ValueError("Retrieval index has no normalized keys")
        return self._keys_cos[: self._size]

    def _ensure_capacity(self, needed: int, dim: int):
        current = 0 if self.keys is None else int(self.keys.shape[0])
        if current >= needed:
            return
        new_capacity = _next_capacity(current, needed)
        new_keys = np.empty((new_capacity, dim), dtype=np.float32)
        new_keys_cos = np.empty((new_capacity, dim), dtype=np.float32) if self.metric == "cosine" else None
        if self._size > 0:
            new_keys[: self._size] = self.keys[: self._size]
            if self.metric == "cosine":
                new_keys_cos[: self._size] = self._keys_cos[: self._size]
        self.keys = new_keys
        self._keys_cos = new_keys_cos

    def fit(self, keys: np.ndarray):
        keys = as_float32_matrix(keys)
        size = int(keys.shape[0])
        capacity = _next_capacity(0, size)
        self.keys = np.empty((capacity, keys.shape[1]), dtype=np.float32)
        self.keys[:size] = keys
        if self.metric == "cosine":
            self._keys_cos = np.empty((capacity, keys.shape[1]), dtype=np.float32)
            self._keys_cos[:size] = l2_normalize_rows(keys)
        else:
            self._keys_cos = None
        self._size = size
        return self

    def add(self, keys: np.ndarray):
        keys = as_float32_matrix(keys)
        if self.keys is None:
            return self.fit(keys)
        if keys.shape[1] != self.keys.shape[1]:
            raise ValueError(f"Key dimension changed from {self.keys.shape[1]} to {keys.shape[1]}")
        old_size = self._size
        new_size = old_size + int(keys.shape[0])
        self._ensure_capacity(new_size, keys.shape[1])
        self.keys[old_size:new_size] = keys
        if self.metric == "cosine":
            self._keys_cos[old_size:new_size] = l2_normalize_rows(keys)
        self._size = new_size
        return self

    def search(self, query_keys: np.ndarray, k: int) -> SearchResult:
        if self.keys is None or self._size == 0:
            raise ValueError("Retrieval index is empty")
        query_keys = as_float32_matrix(query_keys)
        keys = self._active_keys()
        k = min(max(int(k), 1), self._size)
        if self.metric == "cosine":
            query = l2_normalize_rows(query_keys)
            scores = query @ self._active_keys_cos().T
        else:
            diff = query_keys[:, None, :] - keys[None, :, :]
            scores = -np.linalg.norm(diff, axis=2)
        top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        return SearchResult(indices=top_idx.astype(np.int64), scores=top_scores.astype(np.float32))

    def save(self, path: str):
        if self.keys is None or self._size == 0:
            raise ValueError("Cannot save an empty retrieval index")
        np.savez(Path(path), metric=self.metric, keys=self._active_keys())

    def load(self, path: str):
        data = np.load(Path(path), allow_pickle=False)
        self.metric = str(data["metric"])
        self.fit(data["keys"])
        return self


class FaissRetrievalIndex(RetrievalIndex):
    def __init__(self, metric: str = "cosine"):
        self.metric = str(metric).lower()
        if self.metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported retrieval metric: {metric}")
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss is required for FaissRetrievalIndex") from exc
        self._faiss = faiss
        self.keys: Optional[np.ndarray] = None
        self.index = None
        self._size = 0

    def _active_keys(self) -> np.ndarray:
        if self.keys is None:
            raise ValueError("Retrieval index is empty")
        return self.keys[: self._size]

    def _ensure_capacity(self, needed: int, dim: int):
        current = 0 if self.keys is None else int(self.keys.shape[0])
        if current >= needed:
            return
        new_capacity = _next_capacity(current, needed)
        new_keys = np.empty((new_capacity, dim), dtype=np.float32)
        if self._size > 0:
            new_keys[: self._size] = self.keys[: self._size]
        self.keys = new_keys

    def _new_index(self, dim: int):
        if self.metric == "cosine":
            return self._faiss.IndexFlatIP(dim)
        return self._faiss.IndexFlatL2(dim)

    def fit(self, keys: np.ndarray):
        keys = as_float32_matrix(keys)
        size = int(keys.shape[0])
        self._size = 0
        self._ensure_capacity(size, int(keys.shape[1]))
        self.keys[:size] = keys
        self._size = size
        self.index = self._new_index(int(keys.shape[1]))
        index_keys = l2_normalize_rows(keys) if self.metric == "cosine" else keys
        self.index.add(np.ascontiguousarray(index_keys))
        return self

    def add(self, keys: np.ndarray):
        keys = as_float32_matrix(keys)
        if self.keys is None or self.index is None:
            return self.fit(keys)
        if keys.shape[1] != self.keys.shape[1]:
            raise ValueError(f"Key dimension changed from {self.keys.shape[1]} to {keys.shape[1]}")
        old_size = self._size
        new_size = old_size + int(keys.shape[0])
        self._ensure_capacity(new_size, keys.shape[1])
        self.keys[old_size:new_size] = keys
        self._size = new_size
        index_keys = l2_normalize_rows(keys) if self.metric == "cosine" else keys
        self.index.add(np.ascontiguousarray(index_keys))
        return self

    def search(self, query_keys: np.ndarray, k: int) -> SearchResult:
        if self.keys is None or self.index is None or self._size == 0:
            raise ValueError("Retrieval index is empty")
        query_keys = as_float32_matrix(query_keys)
        k = min(max(int(k), 1), self._size)
        index_query = l2_normalize_rows(query_keys) if self.metric == "cosine" else query_keys
        raw_scores, indices = self.index.search(np.ascontiguousarray(index_query), k)
        if self.metric == "l2":
            scores = -np.sqrt(np.maximum(raw_scores, 0.0))
        else:
            scores = raw_scores
        return SearchResult(indices=indices.astype(np.int64), scores=scores.astype(np.float32))

    def save(self, path: str):
        if self.keys is None or self.index is None or self._size == 0:
            raise ValueError("Cannot save an empty retrieval index")
        base = Path(path)
        np.savez(base.with_suffix(".keys.npz"), metric=self.metric, keys=self._active_keys())
        self._faiss.write_index(self.index, str(base.with_suffix(".faiss")))

    def load(self, path: str):
        base = Path(path)
        data = np.load(base.with_suffix(".keys.npz"), allow_pickle=False)
        self.metric = str(data["metric"])
        self.keys = as_float32_matrix(data["keys"])
        self._size = int(self.keys.shape[0])
        self.index = self._faiss.read_index(str(base.with_suffix(".faiss")))
        return self


def create_retrieval_index(backend: str, metric: str) -> RetrievalIndex:
    backend = str(backend).lower()
    if backend in {"exact", "exact_full", "numpy"}:
        return ExactRetrievalIndex(metric=metric)
    if backend in {"faiss", "faiss_flat", "faiss_topm"}:
        try:
            return FaissRetrievalIndex(metric=metric)
        except ImportError:
            return ExactRetrievalIndex(metric=metric)
    raise ValueError(f"Unsupported retrieval backend: {backend}")
