import hashlib
import math

from tg_radar.config import Settings


class Embedder:
    dim = 1024

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def _fallback_embedding(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, text: str) -> list[float]:
        if not self.settings.embeddings_enabled:
            return self._fallback_embedding(text)
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.settings.embedding_model)
        return list(next(self._model.embed([text])))
