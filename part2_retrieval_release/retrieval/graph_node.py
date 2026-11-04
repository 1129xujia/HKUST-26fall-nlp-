"""Thin adapter that exposes the retrieval pipeline as a LangGraph node."""

from typing import Any, Callable

from .pipeline import HybridRetrievalPipeline
from .schemas import RetrievalRequest


def make_retrieval_node(
    pipeline: HybridRetrievalPipeline,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    # 工厂函数捕获已经加载好的 pipeline；父 Graph 每次只传本轮 state。
    def retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
        request = RetrievalRequest(
            query=state["query"],
            sub_queries=state.get("sub_queries", []),
            domain=state.get("domain"),
            top_k=state.get("retrieval_top_k", pipeline.config.final_top_k),
            trace_id=state.get("trace_id"),
        )
        evidence = pipeline.retrieve(request)
        # 只返回 evidence 增量，让 LangGraph 合并回父 state，不覆盖其他组员的字段。
        return {"evidence": [item.to_dict() for item in evidence]}

    return retrieval_node
