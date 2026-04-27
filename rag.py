import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@dataclass
class Recipe:
    title: str
    ingredients: list[str]
    instructions: str
    steps: int
    score: float = 0.0

class RecipeRetriever:
    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        model_name: str = EMBEDDING_MODEL,
    ):
        index_path = os.path.join(index_dir, "index.faiss")
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Please run `python ingest.py` first."
            )
        self._index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self._metadata: list[dict] = pickle.load(f)
        self._model = SentenceTransformer(model_name)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        ingredient_filter: Optional[list[str]] = None,
        max_steps: Optional[int] = None,
    ) -> list[Recipe]:
        vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
        fetch_k = k * 10 if (ingredient_filter or max_steps is not None) else k
        fetch_k = min(fetch_k, self._index.ntotal)
        scores, indices = self._index.search(vec, fetch_k)
        scores = scores[0]
        indices = indices[0]
        results: list[Recipe] = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            recipe = Recipe(
                title=meta["title"],
                ingredients=meta["ingredients"],
                instructions=meta["instructions"],
                steps=meta["steps"],
                score=float(score),
            )
            if max_steps is not None and recipe.steps > max_steps:
                continue
            if ingredient_filter:
                ings_lower = " ".join(recipe.ingredients).lower()
                if not all(kw.lower() in ings_lower for kw in ingredient_filter):
                    continue
            results.append(recipe)
            if len(results) >= k:
                break
        return results

    @staticmethod
    def parse_query_filters(query: str) -> dict:
        q_lower = query.lower()
        max_steps = None
        if re.search(r"\b(less time|fewer steps?|quick|fast|simple|easy)\b", q_lower):
            max_steps = 6
        ingredient_filter: list[str] = []
        patterns = [
            r"(?:with|using|contain(?:ing)?|that (?:has|have|uses?))\s+([a-z\s,]+?)(?:\s+and\b|\s+that\b|$|\.|,)",
        ]
        for pat in patterns:
            match = re.search(pat, q_lower)
            if match:
                raw = match.group(1)
                parts = re.split(r",| and ", raw)
                ingredient_filter = [p.strip() for p in parts if len(p.strip()) > 2]
                break
        return {"ingredient_filter": ingredient_filter, "max_steps": max_steps}
