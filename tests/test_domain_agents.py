from unittest.mock import MagicMock
from rag_system.utils import Domain
from rag_system.domain_agents import DomainAgent, DOMAIN_PROMPTS


def test_domain_prompts_exist_for_all_domains():
    for domain in Domain:
        assert domain in DOMAIN_PROMPTS
        assert len(DOMAIN_PROMPTS[domain]) > 0


def test_agent_stores_correct_system_prompt(manager):
    llm = MagicMock()
    agent = DomainAgent(Domain.BUSINESS, manager, llm)
    assert agent.system_prompt == DOMAIN_PROMPTS[Domain.BUSINESS]


def test_agent_holds_vector_store(manager):
    llm = MagicMock()
    agent = DomainAgent(Domain.COMPLIANCE, manager, llm)
    assert agent.vector_store is manager


def test_agent_search_returns_real_similarity_scores(manager):
    # After the dynamic-retrieval refactor, DomainAgent uses VectorStoreManager.search()
    # so similarity_score is a real FAISS score (not the old hardcoded 0.0).
    results = manager.search("deployment process", Domain.TECHNICAL, top_k=3)
    assert len(results) == 3
    assert all(r.similarity_score > 0 for r in results)


def test_agent_query_passes_top_k_to_search(manager):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="answer")
    agent = DomainAgent(Domain.TECHNICAL, manager, llm)
    agent.vector_store = MagicMock()
    agent.vector_store.search.return_value = []

    agent.query("anything", top_k=7)

    agent.vector_store.search.assert_called_once_with(
        "anything", Domain.TECHNICAL, top_k=7
    )


def test_all_domains_can_create_agents(manager):
    llm = MagicMock()
    for domain in Domain:
        agent = DomainAgent(domain, manager, llm)
        assert agent.domain == domain
        assert agent.vector_store is manager
