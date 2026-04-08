import json
from langchain_core.messages import SystemMessage, HumanMessage
from rag_system.utils import Domain, ClassificationResult

CLASSIFICATION_PROMPT = """You are a query classifier for Postbank's internal knowledge system.
    The system has three knowledge domains:
    - technical: infrastructure, deployments, CI/CD, APIs, monitoring, troubleshooting
    - business: approvals, workflows, change management, stakeholders, timelines
    - compliance: security policies, regulations, audit, data governance, controls
    
    Given a user query, determine which domains are relevant, generate a focused
    sub-query for each relevant domain, and assign a confidence score (0.0 to 1.0) for each domain.
    
    Respond with ONLY a JSON object in this exact format, no other text:"
    "{"
    '  "confidence_scores": {"technical": 0.0, "business": 0.0, "compliance": 0.0},\n'
    '  "sub_queries": {"technical": "...", "business": "...", "compliance": "..."}\n'
    "}"
    
    Rules:
    - Always include all three domains in confidence_scores
    - Set confidence to 0.0 for irrelevant domains
    - Sub-queries for irrelevant domains should be empty strings
    - Sub-queries for relevant domains should be specific and tailored for that domain's context"""

CONFIDENCE_THRESHOLD = 0.3


class QueryClassifier:
    """
    Classifies user queries into one or more knowledge domains.
    Uses an LLM to analyze the query, identify relevant domains, generate domain-specific sub-queries, and assign confidence scores.

    Usage:
        classifier = QueryClassifier(llm)
        result = classifier.classify("How do I deploy a microservice?")
    """

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = CLASSIFICATION_PROMPT

    def classify(self, query: str) -> ClassificationResult:
        """Classify a user query into relevant knowledge domains."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query),
        ]
        response = self.llm.invoke(messages)
        return self._parse_response(response.content, query)

    def _parse_response(self, raw: str, query: str) -> ClassificationResult:
        """Parse LLM JSON response into a ClassificationResult."""
        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw response: {raw}")

        confidence_scores = data["confidence_scores"]

        # Filter to domains above confidence threshold
        domains = [
            domain
            for domain in Domain
            if confidence_scores.get(domain.value, 0.0) >= CONFIDENCE_THRESHOLD
        ]

        sub_queries = {
            domain.value: data["sub_queries"].get(domain.value, "")
            for domain in domains
        }

        return ClassificationResult(
            query=query,
            domains=domains,
            sub_queries=sub_queries,
            confidence_scores=confidence_scores,
            is_multi_domain=len(domains) > 1,
        )

    def _extract_json(self, text: str) -> str:
        """Strip markdown code fences if present."""
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
            return content.strip()
        return text
