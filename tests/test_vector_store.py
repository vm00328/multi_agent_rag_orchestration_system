import pytest
from rag_system.utils import Domain, RetrievedChunk
from rag_system.vector_store import VectorStoreManager
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="module")
def manager():
    m = VectorStoreManager(data_dir="data/")
    m.initialize()
    return m


# --- utils.py ---
def test_domain_values():
    assert Domain.TECHNICAL.value == "technical"
    assert Domain.BUSINESS.value == "business"
    assert Domain.COMPLIANCE.value == "compliance"


# --- vector_store.py ---
def test_all_domains_initialized(manager):
    assert Domain.TECHNICAL in manager.stores
    assert Domain.BUSINESS in manager.stores
    assert Domain.COMPLIANCE in manager.stores


def test_search_returns_retrieved_chunks(manager):
    results = manager.search("deployment", Domain.TECHNICAL)
    assert len(results) > 0
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_search_length_matches_top_k(manager):
    results = manager.search("deployment", Domain.TECHNICAL, top_k=2)
    assert len(results) == 2


def test_search_chunks_have_correct_domain(manager):
    results = manager.search("approval workflow", Domain.BUSINESS)
    for chunk in results:
        assert chunk.domain == Domain.BUSINESS


def test_search_chunks_have_metadata(manager):
    results = manager.search("security policy", Domain.COMPLIANCE)
    for chunk in results:
        assert chunk.chunk_id != ""
        assert chunk.metadata.get("domain") == "compliance"


def test_get_retriever(manager):
    retriever = manager.get_retriever(Domain.TECHNICAL)
    docs = retriever.invoke("CI/CD pipeline")
    assert len(docs) > 0
