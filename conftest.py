import sys
import pytest
from pathlib import Path
from rag_system.vector_store import VectorStoreManager


# in order for Python to find our modules when running tests, we need to add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def manager():
    """Initialize the vector store once for the entire test suite."""
    m = VectorStoreManager(data_dir="data/")
    m.initialize()
    return m
