from typing import List
import numpy as np
from app.models.evolution import ChangeMetrics

class EmbeddingService:
    def __init__(self):
        """Initialize embedding backend lazily.

        If transformers/torch are unavailable, fall back to a lightweight heuristic
        embedding that works in lean CI environments.
        """
        self._use_transformers = False
        self._tokenizer = None
        self._model = None
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self._model = AutoModel.from_pretrained("microsoft/codebert-base")
            self._use_transformers = True
            self._torch = torch
        except Exception:
            # Transformers/torch not available; use numpy-only fallback
            self._use_transformers = False
            self._torch = None

    def get_embedding(self, text: str):
        """Generate an embedding for a given text.

        Uses transformers if available; otherwise returns a simple heuristic
        embedding based on character n-gram counts.
        """
        if self._use_transformers and self._tokenizer is not None and self._model is not None:
            torch = self._torch
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Fallback: simple 8-dim heuristic embedding using byte histogram
        arr = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
        if arr.size == 0:
            return np.zeros(8, dtype=np.float32)
        bins = np.linspace(0, 256, 9)
        hist, _ = np.histogram(arr, bins=bins)
        hist = hist.astype(np.float32)
        hist /= (np.linalg.norm(hist) + 1e-8)
        return hist

    def analyze_commit_importance(
        self,
        commit_message: str,
        changes: List[ChangeMetrics]
    ) -> float:
        """Analyze the semantic importance of a commit."""
        change_summary = " ".join([f"{c.change_type} in {c.file_path}" for c in changes])
        text_to_embed = f"{commit_message} {change_summary}".strip()

        embedding = self.get_embedding(text_to_embed)
        # Importance score: L2 norm of embedding
        importance = float(np.linalg.norm(embedding))
        return importance

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate the semantic similarity between two texts using cosine similarity."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if denom == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / denom)