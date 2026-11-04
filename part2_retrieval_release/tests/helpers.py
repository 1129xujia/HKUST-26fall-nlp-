from retrieval.schemas import Passage, SearchHit


def passage(identifier: str = "p1", *, doc: str = "d1", text: str = "Useful text", domain: str = "dmv") -> Passage:
    return Passage(identifier, doc, domain, "Title", "Section", [], text, ["1"])


def hit(identifier: str, retriever: str, rank: int, *, doc: str = "d1", text: str = "Useful text") -> SearchHit:
    return SearchHit(passage(identifier, doc=doc, text=text), retriever, rank, 1.0 / rank, "query")
