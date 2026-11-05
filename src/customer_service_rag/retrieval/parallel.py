from __future__ import annotations

import asyncio
from time import perf_counter

from customer_service_rag.interfaces import SearchBranch
from customer_service_rag.schemas import RetrievalHit


async def parallel_search(
    branches: list[SearchBranch], queries: list[tuple[str, str]], top_k: int,
) -> tuple[list[list[RetrievalHit]], dict[str, float]]:
    async def execute(branch: SearchBranch, query_id: str, text: str):
        start = perf_counter()
        hits = await asyncio.to_thread(branch.search, text, top_k, query_id)
        return hits, f"{branch.name}/{query_id}", (perf_counter() - start) * 1000

    results = await asyncio.gather(*[
        execute(branch, query_id, text)
        for branch in branches
        for query_id, text in queries
    ])
    return [result[0] for result in results], {
        key: elapsed for _, key, elapsed in results
    }
