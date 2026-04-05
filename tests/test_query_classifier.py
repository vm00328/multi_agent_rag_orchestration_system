import json
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import SystemMessage, HumanMessage
from rag_system.utils import Domain, ClassificationResult
from rag_system.query_classifier import (
    QueryClassifier,
    CLASSIFICATION_PROMPT,
    CONFIDENCE_THRESHOLD,
)


def _make_mock_llm(response_json: dict) -> MagicMock:
    """Create a mock LLM that returns a JSON string as response.content."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(response_json)
    llm.invoke.return_value = mock_response
    return llm


# Construction
def test_classifier_stores_llm_and_prompt():
    llm = MagicMock()
    classifier = QueryClassifier(llm)
    assert classifier.llm is llm
    assert classifier.system_prompt == CLASSIFICATION_PROMPT


# Message building
def test_classify_builds_correct_messages():
    llm = _make_mock_llm(
        {
            "confidence_scores": {"technical": 0.9, "business": 0.0, "compliance": 0.0},
            "sub_queries": {"technical": "test", "business": "", "compliance": ""},
        }
    )
    classifier = QueryClassifier(llm)
    classifier.classify("test query")

    llm.invoke.assert_called_once()
    messages = llm.invoke.call_args[0][0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == CLASSIFICATION_PROMPT
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "test query"


# JSON extraction
def test_extract_json_bare():
    classifier = QueryClassifier(MagicMock())
    raw = '{"confidence_scores": {}}'
    assert classifier._extract_json(raw) == '{"confidence_scores": {}}'


def test_extract_json_markdown_fenced():
    classifier = QueryClassifier(MagicMock())
    raw = '```json\n{"confidence_scores": {}}\n```'
    assert classifier._extract_json(raw) == '{"confidence_scores": {}}'


# Response parsing
def test_parse_response_multi_domain():
    classifier = QueryClassifier(MagicMock())
    raw = json.dumps(
        {
            "confidence_scores": {
                "technical": 0.9,
                "business": 0.1,
                "compliance": 0.85,
            },
            "sub_queries": {
                "technical": "process for deploying a new microservice",
                "business": "",
                "compliance": "compliance checks needed for microservice deployment",
            },
        }
    )
    result = classifier._parse_response(
        raw, "deploy a microservice with compliance checks"
    )

    assert result.domains == [Domain.TECHNICAL, Domain.COMPLIANCE]
    assert result.is_multi_domain is True
    assert "technical" in result.sub_queries
    assert "compliance" in result.sub_queries
    assert "business" not in result.sub_queries
    assert result.confidence_scores["technical"] == 0.9
    assert result.confidence_scores["business"] == 0.1
    assert result.confidence_scores["compliance"] == 0.85


def test_parse_response_single_domain():
    classifier = QueryClassifier(MagicMock())
    raw = json.dumps(
        {
            "confidence_scores": {
                "technical": 0.0,
                "business": 0.95,
                "compliance": 0.1,
            },
            "sub_queries": {
                "technical": "",
                "business": "business approvals for data processing workflow",
                "compliance": "",
            },
        }
    )
    result = classifier._parse_response(raw, "What approvals do I need?")

    assert result.domains == [Domain.BUSINESS]
    assert result.is_multi_domain is False
    assert "business" in result.sub_queries
    assert len(result.sub_queries) == 1


def test_parse_response_no_domains_above_threshold():
    classifier = QueryClassifier(MagicMock())
    raw = json.dumps(
        {
            "confidence_scores": {"technical": 0.1, "business": 0.1, "compliance": 0.1},
            "sub_queries": {"technical": "", "business": "", "compliance": ""},
        }
    )
    result = classifier._parse_response(raw, "hello world")

    assert result.domains == []
    assert result.is_multi_domain is False
    assert result.sub_queries == {}


def test_parse_response_invalid_json_raises():
    classifier = QueryClassifier(MagicMock())
    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        classifier._parse_response("not json at all", "some query")


# Full classify flow
def test_classify_returns_classification_result():
    llm = _make_mock_llm(
        {
            "confidence_scores": {"technical": 0.8, "business": 0.0, "compliance": 0.7},
            "sub_queries": {
                "technical": "troubleshoot API performance issues",
                "business": "",
                "compliance": "security policies for API troubleshooting",
            },
        }
    )
    classifier = QueryClassifier(llm)
    result = classifier.classify(
        "How do I troubleshoot API issues while following security policies?"
    )

    assert isinstance(result, ClassificationResult)
    assert Domain.TECHNICAL in result.domains
    assert Domain.COMPLIANCE in result.domains
    assert result.is_multi_domain is True


def test_classify_preserves_original_query():
    query = "What business approvals are required for a new data processing workflow?"
    llm = _make_mock_llm(
        {
            "confidence_scores": {"technical": 0.4, "business": 0.9, "compliance": 0.2},
            "sub_queries": {
                "technical": "data processing workflow implementation",
                "business": "business approvals for new data processing workflow",
                "compliance": "",
            },
        }
    )
    classifier = QueryClassifier(llm)
    result = classifier.classify(query)

    assert result.query == query
