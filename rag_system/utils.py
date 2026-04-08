from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List
import time
from functools import wraps


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


# Dynamic retrieval strategy
BASE_TOP_K, MAX_TOP_K = 4, 10


def compute_top_k(query: str, num_domains: int) -> int:
    """
    Picks top_k from two query-level signals:
      - query length (words): longer query -> more chunks (broader context)
      - num_domains routed:   more domains -> more chunks per domain (each domain gets less LLM attention, so itneeds stronger evidence to compensate)
    """
    word_count = len(query.split())
    if word_count <= 10:
        length_bonus = 0
    elif word_count <= 20:
        length_bonus = 1
    else:
        length_bonus = 2

    domain_bonus = max(0, num_domains - 1)  # 0 for 1 domain, 1 for 2, 2 for 3
    return min(BASE_TOP_K + length_bonus + domain_bonus, MAX_TOP_K)


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
