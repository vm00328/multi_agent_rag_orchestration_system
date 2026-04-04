from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_system.utils import Domain, RetrievedChunk


# Domain-to-filename mapping
DOMAIN_FILES = {
    Domain.TECHNICAL: "technical_knowledge.pdf",
    Domain.BUSINESS: "business_knowledge.pdf",
    Domain.COMPLIANCE: "compliance_knowledge.pdf",
}


class VectorStoreManager:
    """
    Manages per-domain (technical, business, compliance) FAISS vector stores.
    Each domain gets its own FAISS index built from PDF documents embedded with Qwen3-Embedding-0.6B.

    Usage:
        manager = VectorStoreManager(data_dir="data/")
        manager.initialize()
        results = manager.search("deployment process", Domain.TECHNICAL)
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            # Normalizing the output vectors to L2 unit length for improved cosine similarity search
            encode_kwargs={"normalize_embeddings": True},
        )
        self.stores: dict[Domain, FAISS] = {}

    def initialize(self) -> None:
        """Loads all domain PDFs, chunks them, and builds FAISS indices."""
        for domain, filename in DOMAIN_FILES.items():
            path = self.data_dir / filename

            # Loading PDF pages
            loader = PyPDFLoader(str(path))
            pages = loader.load()

            # Chunking w/ overlap to preserve context across chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)

            # Tagging each chunk with domain metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["domain"] = domain.value
                chunk.metadata["chunk_id"] = f"{domain.value}_{i:03d}"

            # Building FAISS index for this domain
            self.stores[domain] = FAISS.from_documents(chunks, self.embeddings)
            print(f"[{domain.value}] Indexed {len(chunks)} chunks from {filename}")

    def search(
        self, query: str, domain: Domain, top_k: int = 4
    ) -> list[RetrievedChunk]:
        """
        Searches a domain's vector store and returns the top_k most relevant chunks.
        """
        if domain not in self.stores:
            raise ValueError(f"Domain '{domain.value}' not initialized.")

        results = self.stores[domain].similarity_search_with_score(query, k=top_k)

        return [
            RetrievedChunk(
                content=doc.page_content,
                domain=domain,
                source=doc.metadata.get("source", ""),
                chunk_id=doc.metadata.get("chunk_id", ""),
                similarity_score=float(score),
                metadata=doc.metadata,
            )
            for doc, score in results
        ]

    def get_retriever(self, domain: Domain, top_k: int = 4):
        """
        Returns a LangChain retriever for a domain. Will be used by the domain agents.
        """
        return self.stores[domain].as_retriever(search_kwargs={"k": top_k})
