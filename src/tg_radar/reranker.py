from tg_radar.config import Settings
from tg_radar.schemas import SearchHit


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rerank(self, query: str, hits: list[SearchHit], limit: int) -> list[SearchHit]:
        if not self.settings.reranker_enabled:
            return hits[:limit]
        query_terms = {x.lower() for x in query.split() if len(x) > 2}
        for hit in hits:
            text = hit.text.lower()
            lexical_overlap = sum(1 for term in query_terms if term in text)
            hit.score += lexical_overlap * 0.05
            hit.score_reasons.append("local_rerank_overlap")
        return sorted(hits, key=lambda h: h.score, reverse=True)[:limit]
