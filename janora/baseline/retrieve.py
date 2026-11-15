"""TF-IDF retrieval over scheme records (BASELINE stand-in for the team's retriever/reranker)."""
from __future__ import annotations
import json
from sklearn.feature_extraction.text import TfidfVectorizer


class SchemeIndex:
    def __init__(self, schemes: list[dict]):
        self.schemes = schemes
        self.by_id = {s["id"]: s for s in schemes}
        self.states = sorted({s["state"] for s in schemes if s.get("state")})
        docs = [f"{s['name']} {s['name']} {s.get('category', '')} {' '.join(s.get('topics') or [])} {s.get('description', '')}"
                for s in schemes]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.X = self.vec.fit_transform(docs)

    @classmethod
    def from_json(cls, path: str = "data/rag_schemes.json") -> "SchemeIndex":
        return cls(json.load(open(path, encoding="utf-8")))

    def search(self, query: str, k: int = 50, state: str | None = None) -> list[tuple[dict, float]]:
        """Metadata pre-filter: if the user's state is known keep central schemes + that state's schemes."""
        scores = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1]
        hits = []
        for i in order:
            s = self.schemes[i]
            if state and s.get("state") and s["state"] != state:
                continue
            hits.append((s, float(scores[i])))
            if len(hits) == k:
                break
        return hits
