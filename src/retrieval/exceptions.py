"""Public exceptions raised by the retrieval module."""


class RetrievalError(RuntimeError):
    """Base class for retrieval module failures."""


class InvalidRetrievalRequest(ValueError):
    """The graph or caller supplied an invalid retrieval request."""


class RetrievalIndexNotFound(RetrievalError):
    """A required Dense or BM25 index artifact is missing."""


class RetrievalModelMismatch(RetrievalError):
    """The configured model does not match the model used to build the index."""
