"""Small, inspectable local retrieval index for seeded FinOps knowledge documents."""
from __future__ import annotations

import re
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_KB_CANDIDATES = [Path(__file__).resolve().parents[2] / "knowledge", Path(__file__).resolve().parents[1] / "knowledge"]
KB_ROOT = next((path for path in _KB_CANDIDATES if path.exists()), _KB_CANDIDATES[0])
_index: tuple[list[dict[str, Any]], TfidfVectorizer, Any, tuple[tuple[str, int], ...]] | None = None


def _load_index():
    global _index
    files = sorted(KB_ROOT.glob("*.md")) if KB_ROOT.exists() else []
    signature = tuple((str(path), path.stat().st_mtime_ns) for path in files)
    if _index and _index[3] == signature:
        return _index
    documents = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", text, re.M)
        source_match = re.search(r"^## Source\s*\n(.+)$", text, re.M)
        source_line = source_match.group(1).strip() if source_match else ""
        url_match = re.search(r"https?://\S+", source_line)
        chunks = []
        for part in re.split(r"\n\s*\n", text):
            cleaned = re.sub(r"^#{1,6}\s+[^\n]+\n", "", part.strip()).strip()
            if len(cleaned) > 45:
                chunks.append(cleaned)
        for chunk in chunks:
            documents.append({"title": title_match.group(1) if title_match else path.stem.replace("-", " "),
                              "text": chunk, "source": url_match.group(0).rstrip(".,") if url_match else str(path),
                              "document": path.name})
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([doc["text"] for doc in documents]) if documents else None
    _index = documents, vectorizer, matrix, signature
    return _index


def retrieve(question: str, limit: int = 3) -> dict[str, Any]:
    documents, vectorizer, matrix, _ = _load_index()
    if not documents or matrix is None:
        return {"answer": "The local knowledge index has no documents yet. Add Markdown sources under the knowledge directory and reindex.", "sources": [], "retrieved": 0}
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix).ravel()
    selected = [int(i) for i in scores.argsort()[::-1] if scores[i] >= 0.05][:max(1, min(limit, 5))]
    if not selected:
        return {"answer": "I couldn't find a relevant passage in the indexed FinOps documents. Try a more specific question or add an authoritative source.", "sources": [], "retrieved": 0}
    passages = [documents[i] for i in selected]
    context = "\n\n".join(doc["text"] for doc in passages)
    answer = context
    method = "local TF-IDF retrieval with extractive response; no external LLM used"
    api_key, model = os.getenv("FINOPTIX_LLM_API_KEY"), os.getenv("FINOPTIX_LLM_MODEL")
    if api_key and model:
        endpoint = os.getenv("FINOPTIX_LLM_BASE_URL", "https://api.openai.com/v1/chat/completions").rstrip("/")
        body = json.dumps({"model": model, "temperature": 0, "messages": [
            {"role": "system", "content": "Answer the user's cloud FinOps knowledge question using only the supplied retrieved passages. If they do not support an answer, say so. Do not invent prices, savings, source claims, or operational facts. Application financial calculations are handled elsewhere."},
            {"role": "user", "content": f"Question:\n{question}\n\nRetrieved passages:\n{context}"}
        ]}).encode()
        req = urllib.request.Request(endpoint, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                generated = json.loads(response.read())
            answer = generated["choices"][0]["message"]["content"].strip()
            method = f"local TF-IDF retrieval with configured {model} generation"
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
            method = "local TF-IDF retrieval; optional generation failed, showing retrieved passages"
    sources = [{"title": doc["title"], "reference": doc["source"], "document": doc["document"], "relevance": round(float(scores[i]), 3)} for i, doc in zip(selected, passages)]
    return {"answer": answer, "sources": sources, "retrieved": len(sources), "method": method}


def sources() -> list[dict[str, str]]:
    documents, _, _, _ = _load_index()
    unique = {}
    for doc in documents:
        unique[doc["document"]] = {"title": doc["title"], "document": doc["document"], "reference": doc["source"]}
    return list(unique.values())
