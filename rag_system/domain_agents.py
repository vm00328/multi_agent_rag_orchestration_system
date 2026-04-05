from langchain_core.messages import SystemMessage, HumanMessage
from rag_system.utils import Domain, AgentResponse, Citation, RetrievedChunk
from rag_system.vector_store import VectorStoreManager

# Each domain agent gets a tailored system prompt so the LLM answers from the perspective of the right specialist
DOMAIN_PROMPTS = {
    Domain.TECHNICAL: (
        "You are a senior platform engineer at Postbank. "
        "Answer the question using only the provided context from the technical documentation. "
        "Be specific and reference actual tools, systems, and procedures mentioned in the context. "
        "Do not fabricate any information that is not present in the context. If you don't know, say you don't know."
    ),
    Domain.BUSINESS: (
        "You are a business process specialist at Postbank. "
        "Answer the question using only the provided context from the business process handbook. "
        "Be specific about approval workflows, timelines, and stakeholder responsibilities. "
        "Do not fabricate any information that is not present in the context. If you don't know, say you don't know."
    ),
    Domain.COMPLIANCE: (
        "You are a compliance and security officer at Postbank. "
        "Answer the question using only the provided context from the compliance manual. "
        "Be specific about policies, regulations, and required controls. "
        "Do not fabricate any information that is not present in the context. If you don't know, say you don't know."
    ),
}


class DomainAgent:
    """
    A RAG agent for a specific knowledge domain.

    Uses the following pattern:
    1. Retrieve relevant chunks from the domain's FAISS index
    2. Build a prompt with SystemMessage + HumanMessage
    3. Call the LLM

    Usage:
        agent = DomainAgent(Domain.TECHNICAL, vector_store_manager, llm)
        response = agent.query("How do I deploy a microservice?")
    """

    def __init__(self, domain: Domain, vector_store_manager: VectorStoreManager, llm):
        self.domain = domain
        self.retriever = vector_store_manager.get_retriever(domain)
        self.llm = llm
        self.system_prompt = DOMAIN_PROMPTS[domain]

    def query(self, question: str) -> AgentResponse:
        """
        Retrieves relevant context and generates a grounded answer.

        Steps:
            1. Retriever fetches the most relevant chunks from the vector store
            2. Chunks are formatted as context in a human message
            3. LLM receives a system message (domain role) + human message (context + question)
            4. Response is returned with citations for traceability
        """
        # Retrieving relevant chunks
        docs = self.retriever.invoke(question)

        # Formatting the retrieved chunks as context
        context = "\n\n".join(doc.page_content for doc in docs)

        # Building messages and call the LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self.llm.invoke(messages)

        # Building citations and retrieved chunks for traceability
        citations = []
        retrieved_chunks = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", "")

            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    source=doc.metadata.get("source", ""),
                    domain=self.domain,
                    excerpt=doc.page_content[:100],  # First 100 chars as excerpt
                )
            )

            retrieved_chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    domain=self.domain,
                    source=doc.metadata.get("source", ""),
                    chunk_id=chunk_id,
                    similarity_score=0.0,
                    metadata=doc.metadata,
                )
            )

        return AgentResponse(
            domain=self.domain,
            answer=response.content,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
        )
