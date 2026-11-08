"""Optional one-node subgraph wrapper for parent LangGraph integration."""

from typing import Any
from typing_extensions import TypedDict

from .graph_node import make_retrieval_node
from .pipeline import HybridRetrievalPipeline


class RetrievalGraphState(TypedDict, total=False):
    # total=False 只描述可能字段，不做运行时校验；真正校验由 Pipeline 完成。
    query: str
    sub_queries: list[str]
    domain: str | None
    retrieval_top_k: int
    trace_id: str | None
    evidence: list[dict[str, Any]]


def build_retrieval_subgraph(pipeline: HybridRetrievalPipeline):
    """Return a compiled subgraph that can be added to the parent graph."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as error:
        raise RuntimeError("Install langgraph before building the subgraph") from error

    # 该子图只负责检索。query optimizer 的回环应由父 Graph 控制。
    builder = StateGraph(RetrievalGraphState)
    builder.add_node("retrieve_evidence", make_retrieval_node(pipeline))
    builder.add_edge(START, "retrieve_evidence")
    builder.add_edge("retrieve_evidence", END)
    return builder.compile()
