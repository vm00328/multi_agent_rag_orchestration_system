# Multi-Agent RAG Orchestration System | Project Plan v1

## Philosophy

- **LangChain-first**: I will use LangChain abstractions and build custom logic where LangChain doesn't provide what we need.
- **Simple synthetic data**: Three documents, one per domain (technical, business, compliance).
- **Incremental delivery**: Throughout the project's SDLC this plan will be used as a reference point for sprints. Changes to the plan can be made throught various stages of the project depending on progress. Therefore, this plan outlines version 1 of the project.

---

## Tech Stack

| Component       | Library                          | Role                                |
|-----------------|----------------------------------|-------------------------------------|
| Framework       | `langchain`, `langchain-community` | RAG infrastructure, abstractions   |
| LLM             | `langchain/openai`   | Classification, synthesis, decomposition |
| Embeddings      | `Qwen/Qwen3-Embedding-0.6B` via HuggingFace | Document & query embeddings |
| Vector Store    | `faiss-cpu` via LangChain        | Similarity search                   |
| Testing         | `pytest`                         | Test scenarios                      |
| Notebook        | `jupyter`                        | Demo                                |

---

## Repository Structure

```
multi_agent_rag_orchestration_system/
├── README.md
├── requirements.txt
├── rag_system/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── query_classifier.py
│   ├── domain_agents.py
│   ├── vector_store.py
│   └── utils.py
├── data/
│   ├── technical_knowledge.pdf
│   ├── business_knowledge.pdf
│   └── compliance_knowledge.pdf
├── notebooks/demo.ipynb
├── systems_design/self_service_paradox.md
├── tests/test_scenarios.py
└── docs/architecture.md
```

---

## Implementation Steps

### Step 1: Synthetic Data
I first need to create 3 markdown documents:
- `data/technical_knowledge.md` — Microservice deployment, API troubleshooting, CI/CD
- `data/business_knowledge.md` — Approval workflows, change management, data processing procedures
- `data/compliance_knowledge.md` — Security policies, audit checklists, data governance

### Step 2: Vector Store + Embeddings
- LangChain FAISS integration with an embedding model (possibly a Qwen3 family model like Qwen3-Embedding-0.6B)
- One FAISS index per domain
- `RecursiveCharacterTextSplitter` for chunking - (preserves larger units like paragraphs, then sentences, and finally words)
- Each chunk carries metadata: `{domain, source, section}`
- Implement in `rag_system/vector_store.py`

### Step 3: Agents via LangChain
- **Domain Agents** (`domain_agents.py`): Each agent = a LangChain RetrievalQA chain pointing at its own FAISS store
- **Query Classifier** (`query_classifier.py`): LLM chain with structured prompt -> returns target domains + sub-queries
- **Orchestrator** (`orchestrator.py`): Classify -> route -> collect -> synthesize. LangGraph or simple Python coordination.

### Step 4: Advanced Features
Layer on after core works:
- Citation tracking (LangChain retrieval chains already return source docs)
- Knowledge updates (FAISS `add_documents()`)
- Performance monitoring (timing decorators)
- Conflict resolution (confidence scoring in orchestrator)
- Feedback mechanism (simulated weight adjustments)

### Step 5: Tests
- `tests/test_scenarios.py` using pytest
- Three assignment scenarios:
  - "What's the process for deploying a new microservice and what compliance checks are needed?"
  - "How do I troubleshoot API performance issues while following our security policies?"
  - "What business approvals are required for implementing a new data processing workflow?"
- Edge cases: ambiguous query, single-domain query, no-match query

### Step 6: Demo Notebook
- `notebooks/demo.ipynb`
- End-to-end walkthrough of all test scenarios
- Show intermediate outputs (classification, routing, retrieval, synthesis)
- Demonstrate knowledge update flow
- Display performance metrics

### Step 7: Documentation
- `docs/architecture.md` — Architecture diagram, agent flow, design decisions
- `README.md` — Overview, setup, how to run, design rationale

---

## Priority Order

Data -> Vector Store -> Domain Agents-> Classifier-> Orchestrator -> Advanced Features -> Tests -> Demo -> Docs