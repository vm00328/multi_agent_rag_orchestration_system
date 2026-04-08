# Multi-Agent RAG Orchestration System

A simulated multi-agent Retrieval-Augmented Generation (RAG) system that demonstrates intelligent query routing, agent coordination, and response synthesis across multiple knowledge domains.

---

## Overview

This system routes complex natural language queries to specialized domain agents - **Technical**, **Business**, and **Compliance** - each backed by its own vector store. An orchestrator agent coordinates retrieval across relevant domains, resolves conflicts, and synthesizes a unified, cited response.

The project also includes a systems design document addressing the **Self-Service Paradox**: how to build a single automation platform that satisfies both non-technical business users and power users who need full coding flexibility.

---

## Repository Structure

```
multi_agent_rag_orchestration_system/
├── rag_system/
│   ├── __init__.py
│   ├── domain_agents.py       # Three domain agents (technical/business/compliance)
│   ├── orchestrator.py        # Coordinates agents; synthesises final response
│   ├── query_classifier.py    # Routes queries to relevant domain(s)
│   └── utils.py               # Classes
│   ├── vector_store.py        # FAISS index management and embedding utilities
├── data/
│   ├── business_knowledge.pdf
│   └── compliance_knowledge.pdf
│   ├── technical_knowledge.pdf
└── docs/
    └── architecture.md
├── notebooks/
│   └── demo.ipynb             # End-to-end walkthrough of all scenarios
├── systems_design/
│   └── self_service_paradox.md
├── tests/
│   └── test_domain_agents.py
│   └── test_orchestrator.py
│   └── test_query_classifier.py
│   └── test_vector_store.py
├── conftest.py
├── README.md
├── requirements.txt
```

---

## Tech Stack

| Component | Library | Role |
|-----------|---------|------|
| Framework | `langchain`, `langchain-community` | RAG infrastructure and agent abstractions |
| LLM | OpenAI chat model | Classification, synthesis, query decomposition |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` via HuggingFace | Document and query embeddings |
| Vector Store | `faiss-cpu` via LangChain | Per-domain similarity search |
| Testing | `pytest` | Scenario and edge-case coverage |
| Notebook | `jupyter` | Interactive demo |

---

## Setup

### Installation

```bash
git clone https://github.com/vm00328/multi_agent_rag_orchestration_system.git
cd multi_agent_rag_orchestration_system
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY=your_key_here
```

### Running the Demo Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

### Running Tests

```bash
pytest tests/test_scenarios.py -v
```

---

## How It Works

### Agent Flow
```
          ┌──────────────────────────┐
          │        User query        │
          └─────────────┬────────────┘
                        │
                        |
          ┌──────────────────────────┐
          │      Query classifier    │
          │  Identify domains &      │
          │  decompose sub-queries   │
          └─────────────┬────────────┘
                        │
  |-------------------------------------------------------|
  |  Orchestrator: route · coordinate · collect           |
  |                        │                              |
  |          ┌─────────────┼─────────────┐                |
  |          ▼             ▼             ▼                |
  |  ┌──────────────┐ ┌──────────┐ ┌────────────┐         |
  |  │  Technical   │ │ Business │ │ Compliance │         |
  |  │    agent     │ │  agent   │ │   agent    │         |
  |  └──────┬───────┘ └────┬─────┘ └─────┬──────┘         |
  |         │              │             │                |
  |         ▼              ▼             ▼                |
  |  ┌──────────────┐ ┌──────────┐ ┌────────────┐         |
  |  │ FAISS index  │ │  FAISS   │ │   FAISS    │         |
  |  │  technical   │ │ business │ │ compliance │         |
  |  └──────┬───────┘ └────┬─────┘ └─────┬──────┘         |
  |         └──────────────┴─────────────┘                |
  |                        │  retrieved chunks            |
  |  ─────────────────────────────────────────────────    |
  |  Conflict resolution · confidence · citation tracking |
  --------------------------------------------------------|
                        │
                        |
          ┌──────────────────────────----┐
          │   Synthesised response       │
          │  Unified answer + citations  │
          └────────────────────────── ---┘
```
### Key Capabilities

**Query Classification & Decomposition**: An LLM chain with a structured prompt identifies which domains a query requires and breaks it into focused sub-queries per domain.

**Dynamic Retrieval**: Each domain agent looks up and retrieves information from its own FAISS index. Search parameters (top-k, similarity threshold) adjust based on query complexity.

**Conflict Resolution**: Where domain agents return contradictory information, the orchestrator applies confidence scoring and flags ambiguities rather than silently merging them.

**Citation Tracking**: LangChain retrieval chains return source documents alongside answers. Every synthesised response is traceable back to specific chunks with appropriate metadata.

**Knowledge Updates**: New documents can be added at runtime via FAISS `add_documents()`, simulating a live-updating knowledge base.

**Feedback Mechanism**: A simulated feedback loop adjusts per-domain retrieval weights based on response quality signals.

---

## Test Scenarios

The three assignment scenarios covered in `tests/test_scenarios.py`:

1. *"What's the process for deploying a new microservice and what compliance checks are needed?"*
2. *"How do I troubleshoot API performance issues while following our security policies?"*
3. *"What business approvals are required for implementing a new data processing workflow?"*

Additional edge cases: ambiguous queries, single-domain queries, and no-match queries.

---

## Systems Design: The Self-Service Paradox

`systems_design/self_service_paradox.md` addresses the challenge of building a **single unified automation platform** for two very different user groups:

- **Business Users** who need drag-and-drop simplicity, templates, and guided wizards
- **Power Users** who need direct API access, custom logic, and full coding flexibility

The document covers architecture design, UX strategy (progressive disclosure), technical implementation, and long-term platform governance. 

See [`systems_design/self_service_paradox.md`](systems_design/self_service_paradox.md) for the full write-up.

---

## Design Decisions

1. **LangChain Framework**

LangChain abstractions handle the RAG plumbing (retrieval chains, vector store integration, document loaders). Custom logic is added only where LangChain doesn't provide what's needed, keeping the codebase lean.

2. **Synthetic PDF data**

The synthetic data was generated in the form of PDF files as this file format is most widely used for internal company documents.

3. **Embedding Model**

The Qwen3-Embedding-0.6B model is chosen due to it ranking very high on the MTEB leaderboard (rank #16) and being exceptionally lightweight compared to other models that achieve similar performance. It offers strong, instruction-aware performance despite its lightweight size, supporting over 100 languages, 32k context, and user-defined dimensions for high-speed, cost-effective embeddings.
In addition, it is open-source which offers flexibility and cost savings. A proprietary embedding model increases the risk of vendor lock-in and higher costs. 

4. **Embedding Normalization**

Embedding normalization scales vector embeddings to a consistent unit length (L2 norm), preserving their direction while making similarity comparisons (cosine similarity) consistent and fair. It is crucial for preventing popularity bias, improving similarity search accuracy, and ensuring that results are driven by vector direction rather than magnitude.

5. **Vector Store**

FAISS is highly optimized for performance, scalable, memory efficient and serverless.

6. **One FAISS index per domain**

Rather than a single shared index, per-domain isolation keeps retrieval precise and makes it straightforward to update or replace one domain's knowledge without touching others.

7. **Text Splitter**

`RecursiveCharacterTextSplitter` has been chosen for chunking because it respects natural text boundaries (paragraphs -> sentences -> words), which improves retrieval coherence compared to fixed-size splitting. A potential downside is that it might be slower than a simple Character Text Splitter, however, for the number of documents we are working with the advantages outweigh the disadvantages. 

8. **Cross-domain conflict resolution**

I have decided to handle potential conflicts between the domain agents directly as part of the SYNTHESIS_PROMPT, because:

- this is the most straightforward and efficient way of approaching this problem

- we could compare the semantic similarity between the responses of the agents but the problem is that low semantic similarity doesn't mean contradiction - the technical and compliance domains should talk about different things. This way we risk getting false positives constantly.

9. **Saving FAISS indices to disk**

I have decided to save FAISS indices to disk in a project sub-folder 'faiss_indices'. This way I do not have to recompute the indices every time I run the demo notebook. This helps with computational efficiency and allows the Jupyter notebook to be executed considerably faster.

10. **Adding new content to an existing domain's vector store**

New content not added if very similar content already exists (threshold set at 0.95).

11. **Single Domain - No extra LLM call**

When combining the answers from the domain agent answers into a single final response, if only one domain was being used, its output 
is directly used as the final 'synthesized' answer to avoid making an extra LLM call.

12. **Dynamic Retrieval Strategy**

```top_k``` is adjusted once per query based on the query's length and the number of routed domains.

- longer query -> more chunks (broader context)
- more domains -> more chunks per domain (each domain gets less LLM attention, so it needs stronger evidence to compensate)

## Limitations

1. **Contextual Compression**

*Example: text is divided into equal parts (chunks), each 800 characters long. Given a query, the retriever returns the relevant chunk. This chunk, however, might contain both relevant and irrelevant information. Incorporating unrelated content in the LLM prompt can be problematic because it may distract the LLM from focusing on essential details and it consumes space in the prompt that could be allocated to more relevant information.*

With compression, only the information relevant to the query is returned. Compressing here refers to using an LLM to rewrite the retrieved chunk so that it contains only information relevant to the query. This way, the chunks are smaller, and more chunks can be used as contextual information to generate the final answer.

2. **Chunk Size & Overlap**

Due to the time constraint of the project, experimenting with different chunk sizes is limited.

3. **Conflict Resolution**

Conflict resolution is prompt-based, not model-based scoring