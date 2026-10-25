"""Sentence-transformer embeddings (local, free). Vectors are L2-normalised so inner product == cosine."""
import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None  # lazy: importing/loading the model takes a few seconds

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")
