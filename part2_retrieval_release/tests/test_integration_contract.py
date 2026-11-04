from retrieval.config import RetrievalConfig
from retrieval.graph_node import make_retrieval_node
from retrieval.schemas import Evidence


class Pipeline:
    config = RetrievalConfig()

    def retrieve(self, request):
        return [Evidence("p", "d", "dmv", "T", "S", "text", ["1"], ["bm25"], request.query)]


def test_graph_node_maps_state_to_public_contract() -> None:
    node = make_retrieval_node(Pipeline())
    result = node({"query": "license", "sub_queries": [], "domain": "dmv", "retrieval_top_k": 2})
    assert result["evidence"][0]["passage_id"] == "p"


def test_compiled_langgraph_invokes_public_retrieval_contract() -> None:
    from retrieval.subgraph import build_retrieval_subgraph

    result = build_retrieval_subgraph(Pipeline()).invoke({"query": "license"})
    assert result["evidence"][0]["passage_id"] == "p"
