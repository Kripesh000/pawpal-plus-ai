"""Retrieval over the local veterinary reference notes (a small, dependency-free BM25 index)."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import RetrievedReference

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were with when can may not".split())
DOG_ONLY_NOTE = " (dog-specific source; applicability to cats not verified)"


def tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric tokens without stop-words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    """One retrievable passage (a `##` section of a reference note)."""

    ref_id: str
    title: str
    source: str
    url: str
    species: str
    tags: str
    section: str
    text: str


def _parse_note(path: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    meta: Dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        _, front, body = raw.split("---", 2)
        for line in front.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    if "id" not in meta or "title" not in meta:
        raise ValueError(f"{os.path.basename(path)} is missing required 'id'/'title' front matter.")
    chunks: List[Chunk] = []
    for part in re.split(r"(?m)^## ", body):
        part = part.strip()
        if not part:
            continue
        heading, _, text = part.partition("\n")
        chunks.append(
            Chunk(meta["id"], meta["title"], meta.get("source", ""), meta.get("url", ""),
                  meta.get("species", "both"), meta.get("tags", ""), heading.strip(), text.strip())
        )
    return chunks


class KnowledgeBase:
    """Loads reference notes from a directory and answers keyword queries with citations."""

    def __init__(self, directory: Optional[str] = DEFAULT_DIR) -> None:
        self.chunks: List[Chunk] = []
        self.load_error = ""
        if directory is not None:
            try:
                for name in sorted(os.listdir(directory)):
                    if name.endswith(".md"):
                        self.chunks.extend(_parse_note(os.path.join(directory, name)))
            except (OSError, ValueError) as exc:
                self.chunks = []
                self.load_error = f"{type(exc).__name__}: {exc}"
        self._docs = [tokenize(f"{c.title} {c.tags} {c.tags} {c.section} {c.text}") for c in self.chunks]
        self._df: Counter = Counter()
        for doc in self._docs:
            self._df.update(set(doc))
        self._avg = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0.0

    @classmethod
    def empty(cls) -> "KnowledgeBase":
        """A knowledge base with no documents (used as the no-retrieval baseline)."""
        return cls(directory=None)

    def __len__(self) -> int:
        return len(self.chunks)

    def _bm25(self, query_tokens: List[str], idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        doc = self._docs[idx]
        if not doc:
            return 0.0
        tf = Counter(doc)
        score = 0.0
        n = len(self._docs)
        for token in query_tokens:
            if token not in tf:
                continue
            idf = math.log(1 + (n - self._df[token] + 0.5) / (self._df[token] + 0.5))
            score += idf * tf[token] * (k1 + 1) / (tf[token] + k1 * (1 - b + b * len(doc) / self._avg))
        return score

    def retrieve(self, query: str, species: Optional[str] = None, top_k: int = 2, min_score: float = 1.0) -> List[RetrievedReference]:
        """Return up to top_k passages for `query`, each with its citation."""
        q = tokenize(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            if species == "dog" and chunk.species == "cat":
                continue  # filter BEFORE ranking so excluded passages cannot use up a top-k slot
            score = self._bm25(q, i)
            if score >= min_score:
                scored.append((score, i))
        scored.sort(reverse=True)
        results: List[RetrievedReference] = []
        for score, i in scored[:top_k]:
            c = self.chunks[i]
            source = c.source
            if species == "cat" and c.species == "dog":
                source += DOG_ONLY_NOTE
            snippet = c.text.replace("\n", " ")
            results.append(
                RetrievedReference(c.ref_id, c.title, source, c.url, c.section,
                                   snippet[:280] + ("..." if len(snippet) > 280 else ""), round(score, 2))
            )
        return results
