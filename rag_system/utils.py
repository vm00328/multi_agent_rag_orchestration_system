from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List


class Domain(Enum):
    TECHNICAL = "technical"
    BUSINESS = "business"
    COMPLIANCE = "compliance"


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store."""

    content: str
    domain: Domain
    source: str
    chunk_id: str
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class Citation:
    """A citation linking a response back to its source document."""

    chunk_id: str
    source: str
    domain: Domain
    excerpt: str


@dataclass
class AgentResponse:
    """Structured response from a domain agent."""

    domain: Domain
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float = 0.0
    success: bool = True


@dataclass
class ClassificationResult:
    """Result of classifying a user query into one or more knowledge domains."""

    query: str
    domains: List[Domain]
    sub_queries: Dict[str, str]
    confidence_scores: Dict[str, float]
    is_multi_domain: bool


@dataclass
class OrchestratorResult:
    """Final synthesized response from the orchestrator."""

    answer: str  # the final synthesized text combining all domain answers
    classification: ClassificationResult  #  full classification result for traceability (domains, confidence scores, sub-queries)
    agent_responses: List[AgentResponse]  # domain agent responses (answer + citations)
    citations: List[Citation]  # flattened list of all citations from all agents
    total_latency_ms: float = 0.0
    success: bool = True


# Dynamic retrieval strategy
BASE_TOP_K, MAX_TOP_K = 4, 10


class FeedbackStore:
    """Collects per-domain feedback scores and exposes rolling trust scores."""

    def __init__(self):
        self._ratings: Dict[Domain, List[float]] = {d: [] for d in Domain}

    def record(self, domain: Domain, score: float) -> None:
        self._ratings[domain].append(max(0.0, min(1.0, score)))

    def trust_scores(self) -> Dict[str, float]:
        return {
            d.value: (sum(r) / len(r)) if (r := self._ratings[d]) else 1.0
            for d in Domain
        }
