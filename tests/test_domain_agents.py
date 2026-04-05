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


def test_agent_has_retriever(manager):
    llm = MagicMock()
    agent = DomainAgent(Domain.COMPLIANCE, manager, llm)
    assert agent.retriever is not None


def test_agent_retriever_returns_docs(manager):
    llm = MagicMock()
    agent = DomainAgent(Domain.TECHNICAL, manager, llm)
    docs = agent.retriever.invoke("deployment process")
    assert len(docs) > 0


def test_all_domains_can_create_agents(manager):
    llm = MagicMock()
    for domain in Domain:
        agent = DomainAgent(domain, manager, llm)
        assert agent.domain == domain
        assert agent.retriever is not None
