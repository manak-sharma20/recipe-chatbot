

import json
import os
import pickle
import re
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_FILE = os.path.join("data", "recipes_raw_nosource_ar.json")
INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 512


def clean_ingredient(ingredient: str) -> str:
    """Strip trailing ' ADVERTISEMENT' and whitespace."""
    return re.sub(r"\s*ADVERTISEMENT\s*$", "", ingredient).strip()


def count_steps(instructions: str) -> int:
    """
    Estimate number of cooking steps.
    Split on sentence-ending punctuation or numbered list markers.
    """
    if not instructions:
        return 0
    # Split on newlines first, then count non-empty lines
    lines = [l.strip() for l in instructions.split("\n") if l.strip()]
    if len(lines) > 1:
        return len(lines)
    # Fallback: split on '. '
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", instructions) if s.strip()]
    return max(1, len(sentences))


def build_chunk(title: str, ingredients: list[str], instructions: str) -> str:
    """Build the text chunk used for embedding."""
    clean_ings = [clean_ingredient(i) for i in ingredients]
    return (
        f"Recipe: {title}\n"
        f"Ingredients: {', '.join(clean_ings)}\n"
        f"Instructions: {instructions}"
    )



def main():
    print(f"Loading data from {DATA_FILE} …")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for recipe_id, recipe in raw.items():
        title = recipe.get("title", "").strip()
        ingredients = recipe.get("ingredients") or []
        instructions = recipe.get("instructions", "").strip()

        if not title or not instructions:
            continue

        clean_ings = [clean_ingredient(i) for i in ingredients]
        steps = count_steps(instructions)
        chunk = build_chunk(title, ingredients, instructions)

        records.append(
            {
                "id": recipe_id,
                "title": title,
                "ingredients": clean_ings,
                "instructions": instructions,
                "steps": steps,
                "chunk": chunk,
            }
        )

    print(f"  → {len(records):,} valid recipes loaded.")

    
    print(f"Loading embedding model '{EMBEDDING_MODEL}' …")
    model = SentenceTransformer(EMBEDDING_MODEL)

    chunks = [r["chunk"] for r in records]
    print(f"Encoding {len(chunks):,} recipe chunks in batches of {BATCH_SIZE} …")

    all_embeddings = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs)
        done = min(start + BATCH_SIZE, len(chunks))
        pct = done / len(chunks) * 100
        print(f"  {done:>6,} / {len(chunks):,}  ({pct:.1f}%)", end="\r")

    print()
    embeddings = np.vstack(all_embeddings).astype("float32")

    
    faiss.normalize_L2(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal:,} vectors (dim={dim}).")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"Index saved → {INDEX_PATH}")




if __name__ == "__main__":
    main()
