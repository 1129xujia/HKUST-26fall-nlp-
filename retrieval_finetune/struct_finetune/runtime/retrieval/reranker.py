"""Cross-Encoder reranking for fused retrieval candidates."""

from typing import Any, Protocol
import re
import sys

import numpy as np

from .schemas import FusedHit


ACKNOWLEDGEMENT_PATTERN = re.compile(
    r"\s*(?:yes|no|yeah|yep|nope|yes it is|no it is not|i do|i don't|"
    r"i am|i am not|correct|that's right)[.!?\s]*",
    flags=re.IGNORECASE,
)


class Reranker(Protocol):
    def rerank(
        self,
        main_query: str,
        sub_queries: list[str],
        candidates: list[FusedHit],
    ) -> list[FusedHit]: ...


class NoOpReranker:
    """Keep RRF order for ablation experiments or graceful fallback."""

    def rerank(
        self,
        main_query: str,
        sub_queries: list[str],
        candidates: list[FusedHit],
    ) -> list[FusedHit]:
        for candidate in candidates:
            candidate.rerank_score = candidate.rrf_score
        return list(candidates)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        model: Any | None = None,
        query_strategy: str = "identity",
        use_sub_queries: bool = True,
        skip_acknowledgements: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        if query_strategy not in {"identity", "current_first_user_history"}:
            raise ValueError(f"Unsupported reranker query strategy: {query_strategy}")
        self.query_strategy = query_strategy
        self.use_sub_queries = use_sub_queries
        self.skip_acknowledgements = skip_acknowledgements
        if model is None:
            # Keep importing this module cheap for unit tests and non-reranking modes.
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device=device)
            if sys.platform == "darwin" and str(model.model.device).startswith("cpu"):
                import torch

                torch.set_num_threads(1)
                # This macOS/Torch build produced NaNs in Linear for finite
                # safetensors-backed weights. Owned tensor storage fixes the
                # observed issue without changing parameter values or dtype.
                with torch.no_grad():
                    for parameter in model.model.parameters():
                        parameter.data = parameter.data.clone()
        self.model = model

    def _prepare_main_query(self, main_query: str) -> str:
        if self.query_strategy == "identity":
            return main_query
        user_turns = [
            item.strip()
            for item in re.findall(r"^User:\s*(.+)$", main_query, flags=re.MULTILINE)
            if item.strip()
        ]
        if not user_turns:
            return main_query
        return "\n".join([user_turns[-1], *user_turns[:-1]])

    @staticmethod
    def _latest_user_turn(main_query: str) -> str:
        user_turns = re.findall(r"^User:\s*(.+)$", main_query, flags=re.MULTILINE)
        return user_turns[-1].strip() if user_turns else main_query.strip()

    def rerank(
        self,
        main_query: str,
        sub_queries: list[str],
        candidates: list[FusedHit],
    ) -> list[FusedHit]:
        if not candidates:
            return []

        if (
            self.skip_acknowledgements
            and ACKNOWLEDGEMENT_PATTERN.fullmatch(self._latest_user_turn(main_query))
        ):
            for candidate in candidates:
                candidate.rerank_score = candidate.rrf_score
            return list(candidates)

        prepared_main_query = self._prepare_main_query(main_query)
        queries = [prepared_main_query]
        if self.use_sub_queries:
            queries.extend(sub_queries)
        pairs = [
            (query, candidate.passage.index_text)
            for candidate in candidates
            for query in queries
        ]
        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        ).reshape(len(candidates), len(queries))
        if not np.isfinite(scores).all():
            raise RuntimeError("Cross-Encoder returned non-finite scores; refusing invalid ranking")

        for candidate, row in zip(candidates, scores, strict=True):
            best_index = int(np.argmax(row))
            candidate.rerank_score = float(row[best_index])
            best_query = queries[best_index]
            if best_query in candidate.matched_queries:
                candidate.matched_queries.remove(best_query)
            candidate.matched_queries.insert(0, best_query)

        return sorted(
            candidates,
            key=lambda item: (
                -item.rerank_score,
                -item.rrf_score,
                item.passage.passage_id,
            ),
        )
