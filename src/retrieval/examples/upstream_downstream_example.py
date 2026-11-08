"""演示 Part 1 -> Part 2 -> Part 3 的最小接口，不绑定具体业务实现。"""

import argparse
from typing import Any, TypedDict

from retrieval.config import RetrievalConfig
from retrieval.graph_node import make_retrieval_node
from retrieval.pipeline import HybridRetrievalPipeline


class ParentState(TypedDict, total=False):
    # Part 1 写入的字段
    query: str
    sub_queries: list[str]
    domain: str | None
    retrieval_top_k: int
    trace_id: str | None
    # Part 2 写入、Part 3 读取的字段
    evidence: list[dict[str, Any]]
    # Part 3 最终写入的字段
    answer: str


def build_part2_node(artifact_dir: str = "artifacts"):
    """应用启动时构建一次；不要在每个用户请求里重复加载模型。"""
    config = RetrievalConfig.for_mode("rrf", artifact_dir=artifact_dir)
    pipeline = HybridRetrievalPipeline.from_artifacts(
        artifact_dir=artifact_dir,
        config=config,
        device="cpu",
    )
    return make_retrieval_node(pipeline)


def example_upstream_state() -> ParentState:
    """模拟上游 query planner 的输出。"""
    return {
        "query": "How do I replace a lost driver's license?",
        "sub_queries": ["lost license replacement documents"],
        "domain": "dmv",
        "retrieval_top_k": 6,
        "trace_id": "dialog-12:turn-4",
    }


def example_downstream_generator(state: ParentState) -> dict[str, str]:
    """模拟下游：保持 evidence 排序，并把正文交给 LLM 生成答案。"""
    evidence = state.get("evidence", [])
    context = "\n\n".join(
        f"[{item['passage_id']}] {item['title']} > {item['section_title']}\n{item['text']}"
        for item in evidence
    )
    # 真实项目在这里调用 LLM；示例只展示下游能稳定获得的上下文。
    return {"answer": f"Generator receives {len(evidence)} passages:\n{context}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    retrieve_evidence = build_part2_node(args.artifact_dir)
    state = example_upstream_state()
    state.update(retrieve_evidence(state))
    state.update(example_downstream_generator(state))
    print(state["answer"])
