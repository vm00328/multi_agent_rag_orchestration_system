from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from rag_system.utils import Domain, AgentResponse, OrchestratorResult
from rag_system.vector_store import VectorStoreManager
from rag_system.domain_agents import DomainAgent
from rag_system.query_classifier import QueryClassifier

SYNTHESIS_PROMPT = """You are a knowledge synthesizer for Postbank's internal knowledge system.
    You will receive answers from multiple domain specialists (technical, business, compliance).
    Combine them into a single, coherent response that:
    - Addresses the original question completely
    - Preserves specific details from each domain
    - Highlights connections between domains where relevant
    - Uses clear section headings when covering multiple domains
    
    If domain answers contradict each other, explicitly flag the conflict, present both perspectives, and recommend following the higher-confidence domain's guidance."
    Do not fabricate information beyond what the specialists provided."""


class Orchestrator:
    """
    Central orchestrator for the multi-agent RAG system.
    Coordinates the full pipeline: query classification, routing sub-queries to domain agents, response collection and final answer synthesis.
    Usage:
        orchestrator = Orchestrator(llm, vector_store_manager)
        response = orchestrator.query("How do I deploy a microservice?")
    """

    def __init__(self, llm, vector_store_manager: VectorStoreManager):
        self.llm = llm
        self.classifier = QueryClassifier(llm)
        self.agents: Dict[Domain, DomainAgent] = {
            domain: DomainAgent(domain, vector_store_manager, llm) for domain in Domain
        }

    def query(self, user_query: str) -> OrchestratorResult:
        """
        Runs the full RAG pipeline for a user query.
        Steps:
            1. Classifies the query into relevant domains
            2. Routes domain-specific sub-queries to the appropriate agents
            3. Collects agent responses and flatten citations
            4. Synthesizes a final answer
        """
        # Query Classification
        classification = self.classifier.classify(user_query)

        # Handling no relevant domains
        if not classification.domains:
            return OrchestratorResult(
                answer="I couldn't identify relevant knowledge domains for your query. Please try rephrasing or be more specific.",
                classification=classification,
                agent_responses=[],
                citations=[],
            )

        # Routing to domain agents and collecting responses
        agent_responses: List[AgentResponse] = []
        for domain in classification.domains:
            sub_query = classification.sub_queries.get(domain.value, user_query)
            response = self.agents[domain].query(sub_query)
            agent_responses.append(response)

        # Flattenning citations from all agents
        citations = [
            citation for response in agent_responses for citation in response.citations
        ]

        # Response Synthesis
        answer = self._synthesize(
            user_query, classification.confidence_scores, agent_responses
        )

        return OrchestratorResult(
            answer=answer,
            classification=classification,
            agent_responses=agent_responses,
            citations=citations,
        )

    def _synthesize(
        self,
        user_query: str,
        confidence_scores: Dict[str, float],
        agent_responses: List[AgentResponse],
    ) -> str:
        """Combines domain agent answers into a single final response."""
        # Single domain - no need for an extra LLM call
        if len(agent_responses) == 1:
            return agent_responses[0].answer

        # Multi-domain - synthesize via LLM
        domain_answers = "\n\n".join(
            f"[{response.domain.value.upper()} | confidence: {confidence_scores.get(response.domain.value, 0.0):.2f}]\n{response.answer}"
            for response in agent_responses
        )

        messages = [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(
                content=f"Original question: {user_query}\n\n"
                f"Domain specialist answers:\n{domain_answers}"
            ),
        ]

        response = self.llm.invoke(messages)
        return response.content
