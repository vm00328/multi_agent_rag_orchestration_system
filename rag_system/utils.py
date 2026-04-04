from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict


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
