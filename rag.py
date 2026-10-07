from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INDEX_DIR = Path("rag_index")

INDEX_PATH = INDEX_DIR / "index.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 4


# ============================================================
# RAG
# ============================================================

class HyperionRAG:

    def __init__(self):

        if not INDEX_PATH.exists():
            raise RuntimeError(
                f"RAG index not found: {INDEX_PATH}"
            )

        if not CHUNKS_PATH.exists():
            raise RuntimeError(
                f"RAG chunks not found: {CHUNKS_PATH}"
            )

        print(
            "Loading RAG embedding model..."
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Loading FAISS index..."
        )

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        with open(
            CHUNKS_PATH,
            "rb",
        ) as file:

            self.chunks = pickle.load(
                file
            )

        print(
            f"RAG loaded: {len(self.chunks)} chunks"
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ):

        if not query.strip():
            return []

        # ----------------------------------------------------
        # Embed query
        # ----------------------------------------------------

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        # ----------------------------------------------------
        # FAISS similarity search
        # ----------------------------------------------------

        scores, indices = self.index.search(
            embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "score": float(score),
                }
            )

        return results


    # ========================================================
    # CONTEXT
    # ========================================================

    def build_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ):

        results = self.search(
            query,
            top_k=top_k,
        )

        if not results:
            return ""

        parts = []

        for i, result in enumerate(
            results,
            start=1,
        ):

            source = result["source"]
            page = result["page"]
            score = result["score"]

            location = source

            if page is not None:
                location += (
                    f", page {page}"
                )

            parts.append(
                f"""
SOURCE {i}
Document: {location}
Similarity: {score:.3f}

{result["text"]}
""".strip()
            )

        return "\n\n---\n\n".join(
            parts
        )


# ============================================================
# GLOBAL INSTANCE
# ============================================================

rag = HyperionRAG()


# ============================================================
# SIMPLE FUNCTION
# ============================================================

def search_knowledge(
    query: str,
    top_k: int = DEFAULT_TOP_K,
):

    return rag.search(
        query,
        top_k=top_k,
    )


def get_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
):

    return rag.build_context(
        query,
        top_k=top_k,
    )