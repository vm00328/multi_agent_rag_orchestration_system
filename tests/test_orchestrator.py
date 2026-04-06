from unittest.mock import MagicMock, patch
from rag_system.utils import (
    Domain,
    AgentResponse,
    Citation,
    ClassificationResult,
    OrchestratorResult,
)
from rag_system.orchestrator import Orchestrator, SYNTHESIS_PROMPT


# Helper functions to create mock classification results and agent responses for testing
def _make_classification(domains, sub_queries, query="test query"):
    """Helper to build a ClassificationResult for testing."""
    return ClassificationResult(
        query=query,
        domains=domains,
        sub_queries=sub_queries,
        confidence_scores={
            "technical": 0.0,
            "business": 0.0,
            "compliance": 0.0,
        },  # default scores (testing the orchestrator, not the classifier)
        is_multi_domain=len(domains) > 1,
    )


# Helper to create an AgentResponse with a specified number of citations for testing
def _make_agent_response(domain, answer="Test answer", num_citations=1):
    """Helper to build an AgentResponse for testing."""
    citations = [
        Citation(
            chunk_id=f"{domain.value}_{i:03d}",
            source=f"{domain.value}_knowledge.pdf",
            domain=domain,
            excerpt=f"Excerpt {i} from {domain.value}",
        )
        for i in range(num_citations)
    ]
    return AgentResponse(
        domain=domain,
        answer=answer,
        citations=citations,
        retrieved_chunks=[],
    )


# Initialization
@patch("rag_system.orchestrator.QueryClassifier")  # mock_classifier_cls (second param)
@patch("rag_system.orchestrator.DomainAgent")  # mock_agent_cls (first param)
# Testing that the Orchestrator's constructor correctly initializes its QueryClassifier and DomainAgent instances, ensuring that the LLM is passed to the classifier and that one agent is created for each domain
def test_orchestrator_creates_classifier_and_agents(
    mock_agent_cls, mock_classifier_cls
):
    llm = MagicMock()
    vsm = MagicMock()
    orchestrator = Orchestrator(llm, vsm)

    mock_classifier_cls.assert_called_once_with(llm)
    assert mock_agent_cls.call_count == 3  # one per domain


# Classification routing
@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing that when the Orchestrator's query method is called, it first invokes the QueryClassifier to classify the user query, ensuring that the classification step is executed as part of the query processing pipeline
def test_query_calls_classifier(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification([], {})
    mock_classifier_cls.return_value.classify.return_value = classification

    orchestrator = Orchestrator(llm, vsm)
    orchestrator.query("test query")

    orchestrator.classifier.classify.assert_called_once_with(
        "test query"
    )  # verifying the classifier is called with the original user query


# Agent routing
@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing that the orchestrator correctly routes sub-queries to the appropriate domain agents based on the classification result
def test_query_routes_to_correct_agents(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification(
        [Domain.TECHNICAL, Domain.COMPLIANCE],
        {"technical": "deploy microservice", "compliance": "compliance checks"},
    )
    mock_classifier_cls.return_value.classify.return_value = classification

    tech_response = _make_agent_response(Domain.TECHNICAL, "Tech answer")
    compliance_response = _make_agent_response(Domain.COMPLIANCE, "Compliance answer")

    orchestrator = Orchestrator(llm, vsm)
    orchestrator.agents[Domain.TECHNICAL] = MagicMock()
    orchestrator.agents[Domain.TECHNICAL].query.return_value = tech_response
    orchestrator.agents[Domain.COMPLIANCE] = MagicMock()
    orchestrator.agents[Domain.COMPLIANCE].query.return_value = compliance_response
    orchestrator.agents[Domain.BUSINESS] = MagicMock()

    # Mock the synthesis LLM call
    mock_synthesis = MagicMock()  # two domains are used -> synthesis should be called
    mock_synthesis.content = "Synthesized answer"
    orchestrator.llm.invoke.return_value = mock_synthesis

    orchestrator.query("test query")

    orchestrator.agents[Domain.TECHNICAL].query.assert_called_once()
    orchestrator.agents[Domain.COMPLIANCE].query.assert_called_once()
    orchestrator.agents[Domain.BUSINESS].query.assert_not_called()


@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing not just that the right agents were called, but that each received its specific sub-query from the classification
def test_query_passes_sub_queries_to_agents(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification(
        [Domain.TECHNICAL, Domain.COMPLIANCE],
        {"technical": "deploy microservice", "compliance": "compliance checks"},
    )
    mock_classifier_cls.return_value.classify.return_value = classification

    tech_response = _make_agent_response(Domain.TECHNICAL)
    compliance_response = _make_agent_response(Domain.COMPLIANCE)

    orchestrator = Orchestrator(llm, vsm)
    orchestrator.agents[Domain.TECHNICAL] = MagicMock()
    orchestrator.agents[Domain.TECHNICAL].query.return_value = tech_response
    orchestrator.agents[Domain.COMPLIANCE] = MagicMock()
    orchestrator.agents[Domain.COMPLIANCE].query.return_value = compliance_response

    mock_synthesis = MagicMock()
    mock_synthesis.content = "Synthesized"
    orchestrator.llm.invoke.return_value = mock_synthesis

    orchestrator.query("test query")

    orchestrator.agents[Domain.TECHNICAL].query.assert_called_once_with(
        "deploy microservice"
    )
    orchestrator.agents[Domain.COMPLIANCE].query.assert_called_once_with(
        "compliance checks"
    )


# Response structure
@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing that the final OrchestratorResult contains the correct classification, agent responses, and flattened citations from all agents
def test_query_returns_synthesized_response(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification(
        [Domain.BUSINESS],
        {"business": "business approvals"},
    )
    mock_classifier_cls.return_value.classify.return_value = classification

    business_response = _make_agent_response(Domain.BUSINESS, "Business answer")

    orchestrator = Orchestrator(llm, vsm)
    orchestrator.agents[Domain.BUSINESS] = MagicMock()
    orchestrator.agents[Domain.BUSINESS].query.return_value = business_response

    result = orchestrator.query("test query")

    assert isinstance(result, OrchestratorResult)
    assert (
        result.classification is classification
    )  # same object returned by the mock classifier
    assert len(result.agent_responses) == 1
    assert result.agent_responses[0].domain == Domain.BUSINESS


# Citation aggregation
@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing that all citations from all domain agents are correctly collected and included in the final OrchestratorResult, ensuring no citations are lost in the aggregation process
def test_query_collects_all_citations(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification(
        [Domain.TECHNICAL, Domain.COMPLIANCE],
        {"technical": "tech query", "compliance": "compliance query"},
    )
    mock_classifier_cls.return_value.classify.return_value = classification

    tech_response = _make_agent_response(Domain.TECHNICAL, "Tech", num_citations=2)
    compliance_response = _make_agent_response(
        Domain.COMPLIANCE, "Compliance", num_citations=3
    )

    orchestrator = Orchestrator(llm, vsm)
    orchestrator.agents[Domain.TECHNICAL] = MagicMock()
    orchestrator.agents[Domain.TECHNICAL].query.return_value = tech_response
    orchestrator.agents[Domain.COMPLIANCE] = MagicMock()
    orchestrator.agents[Domain.COMPLIANCE].query.return_value = compliance_response

    mock_synthesis = MagicMock()
    mock_synthesis.content = "Synthesized"
    orchestrator.llm.invoke.return_value = mock_synthesis

    result = orchestrator.query("test query")

    assert len(result.citations) == 5  # 2 tech + 3 compliance


# Single-domain synthesis behaviour
@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing that when only one domain is relevant, the orchestrator returns the agent's answer directly without calling the synthesis LLM, ensuring efficiency in single-domain scenarios
def test_single_domain_skips_synthesis_llm_call(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification(
        [Domain.TECHNICAL],
        {"technical": "deploy microservice"},
    )
    mock_classifier_cls.return_value.classify.return_value = classification

    tech_response = _make_agent_response(Domain.TECHNICAL, "Direct tech answer")

    orchestrator = Orchestrator(llm, vsm)
    orchestrator.agents[Domain.TECHNICAL] = MagicMock()
    orchestrator.agents[Domain.TECHNICAL].query.return_value = tech_response

    result = orchestrator.query("test query")

    # LLM should NOT be called for synthesis (only classifier used it)
    orchestrator.llm.invoke.assert_not_called()
    assert result.answer == "Direct tech answer"


# Multi-domain synthesis behaviour
@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing that when multiple domains are relevant, the orchestrator correctly calls the synthesis LLM to combine the agent responses, ensuring that multi-domain queries trigger the expected synthesis process
def test_multi_domain_calls_synthesis_llm(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification(
        [Domain.TECHNICAL, Domain.COMPLIANCE],
        {"technical": "tech query", "compliance": "compliance query"},
    )
    mock_classifier_cls.return_value.classify.return_value = classification

    tech_response = _make_agent_response(Domain.TECHNICAL, "Tech answer")
    compliance_response = _make_agent_response(Domain.COMPLIANCE, "Compliance answer")

    orchestrator = Orchestrator(llm, vsm)
    orchestrator.agents[Domain.TECHNICAL] = MagicMock()
    orchestrator.agents[Domain.TECHNICAL].query.return_value = tech_response
    orchestrator.agents[Domain.COMPLIANCE] = MagicMock()
    orchestrator.agents[Domain.COMPLIANCE].query.return_value = compliance_response

    mock_synthesis = MagicMock()
    mock_synthesis.content = "Combined answer"
    orchestrator.llm.invoke.return_value = mock_synthesis

    result = orchestrator.query("test query")

    orchestrator.llm.invoke.assert_called_once()
    assert result.answer == "Combined answer"

    # Verifying the synthesis prompt reaches the LLM
    call_args = orchestrator.llm.invoke.call_args[0][0]  # messages list
    assert call_args[0].content == SYNTHESIS_PROMPT


# Edge case - the classifier returns no relevant domains to the query
@patch("rag_system.orchestrator.QueryClassifier")
@patch("rag_system.orchestrator.DomainAgent")
# Testing that if the classifier returns no relevant domains, the orchestrator returns a fallback response without calling any agents or the synthesis LLM, ensuring graceful handling of unclassifiable queries
def test_empty_domains_returns_fallback(mock_agent_cls, mock_classifier_cls):
    llm = MagicMock()
    vsm = MagicMock()

    classification = _make_classification([], {})
    mock_classifier_cls.return_value.classify.return_value = classification

    orchestrator = Orchestrator(llm, vsm)
    result = orchestrator.query("irrelevant query")

    assert isinstance(result, OrchestratorResult)
    assert result.agent_responses == []
    assert result.citations == []
