import pytest
from rag_system.utils import Domain, RetrievedChunk
from rag_system.vector_store import VectorStoreManager
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="module")
def manager():
    """Initializes the vector store once for all tests in this module."""
    m = VectorStoreManager(data_dir="data/")
    m.initialize()
    return m


# --- utils.py ---
def test_domain_values():
    """Verifies that each Domain enum maps to the expected string value."""
    assert Domain.TECHNICAL.value == "technical"
    assert Domain.BUSINESS.value == "business"
    assert Domain.COMPLIANCE.value == "compliance"


# --- vector_store.py ---
def test_all_domains_initialized(manager):
    """After initialization, all three domain indices should exist in the store"""
    assert Domain.TECHNICAL in manager.stores
    assert Domain.BUSINESS in manager.stores
    assert Domain.COMPLIANCE in manager.stores


def test_search_returns_retrieved_chunks(manager):
    """Search should return a non-empty list of RetrievedChunk objects"""
    results = manager.search("deployment", Domain.TECHNICAL)
    assert len(results) > 0
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_search_length_matches_top_k(manager):
    """When top_k=2, search should return exactly 2 results"""
    results = manager.search("deployment", Domain.TECHNICAL, top_k=2)
    assert len(results) == 2


def test_search_chunks_have_correct_domain(manager):
    """Every chunk returned from a domain search should be tagged with that domain"""
    results = manager.search("approval workflow", Domain.BUSINESS)
    for chunk in results:
        assert chunk.domain == Domain.BUSINESS


def test_search_chunks_have_metadata(manager):
    """Each returned chunk should carry a non-empty chunk_id and correct domain in its metadata"""
    results = manager.search("security policy", Domain.COMPLIANCE)
    for chunk in results:
        assert chunk.chunk_id != ""
        assert chunk.metadata.get("domain") == "compliance"


def test_get_retriever(manager):
    """The LangChain retriever interface should return documents when invoked"""
    retriever = manager.get_retriever(Domain.TECHNICAL)
    docs = retriever.invoke("CI/CD pipeline")
    assert len(docs) > 0
