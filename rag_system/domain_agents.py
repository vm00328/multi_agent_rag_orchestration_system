from langchain_core.messages import SystemMessage, HumanMessage
from torch import chunk
from rag_system.utils import Domain, AgentResponse, Citation, RetrievedChunk, BASE_TOP_K
from rag_system.vector_store import VectorStoreManager

# Each domain agent gets a tailored system prompt so the LLM answers from the perspective of the right specialist
DOMAIN_PROMPTS = {
    Domain.TECHNICAL: (
        """You are a senior platform engineer at Postbank.
        Answer the question using only the provided context from the technical documentation.
        Be specific and reference actual tools, systems, and procedures mentioned in the context.
        Do not fabricate any information that is not present in the context. If you don't know, say you don't know."""
    ),
    Domain.BUSINESS: (
        """You are a business process specialist at Postbank.
        Answer the question using only the provided context from the business process handbook.
        Be specific about approval workflows, timelines, and stakeholder responsibilities.
        Do not fabricate any information that is not present in the context. If you don't know, say you don't know."""
    ),
    Domain.COMPLIANCE: (
        """You are a compliance and security officer at Postbank.
        Answer the question using only the provided context from the compliance manual.
        Be specific about policies, regulations, and required controls.
        Do not fabricate any information that is not present in the context. If you don't know, say you don't know."""
    ),
}


class DomainAgent:
    """
    A RAG agent for a specific knowledge domain.

    Uses a dynamic retrieval startegy based on the query's complexity and the domain's characteristics.:
    1. Retrieves top_k chunks from the domain's FAISS index (k varies by query complexity)
    2. Filters chunks by similarity score threshold to remove low-relevance results
    3. Builds a prompt and calls the LLM

    Usage:
        agent = DomainAgent(Domain.TECHNICAL, vector_store_manager, llm)
        response = agent.query("How do I deploy a microservice?")
    """

    def __init__(self, domain: Domain, vector_store_manager: VectorStoreManager, llm):
        self.domain = domain
        self.vector_store = vector_store_manager
        self.llm = llm
        self.system_prompt = DOMAIN_PROMPTS[domain]

    def query(self, question: str, top_k: int = BASE_TOP_K) -> AgentResponse:
        """
        Retrieves relevant context and generates a grounded answer.

        Steps:
            1. Vector store search returns the top_k most relevant chunks
            2. Chunks are formatted as context for the LLM
            3. LLM receives a systemm message (domain role) and a human message (context + question)
            4. Returns response with citations and similarity scores for traceability
        """
        # Retrieving relevant chunks
        retrieved_chunks = self.vector_store.search(question, self.domain, top_k=top_k)

        # Formatting the retrieved chunks as context
        context = "\n\n".join(chunk.content for chunk in retrieved_chunks)

        # Building messages and calling the LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self.llm.invoke(messages)

        # Building citations and retrieved chunks for traceability
        citations = [
            Citation(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                domain=self.domain,
                excerpt=chunk.content[:100],  # First 100 chars as excerpt
            )
            for chunk in retrieved_chunks
        ]

        return AgentResponse(
            domain=self.domain,
            answer=response.content,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
        )
