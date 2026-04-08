# Purpose

This document explains the internal architecture of the multi-agent RAG orchestration system. It focuses on component responsibilities, end-to-end query flow, retrieval design, synthesis logic, and the main technical trade-offs behind the implementation.

---

# System Overview

The system is a simulated multi-agent Retrieval-Augmented Generation (RAG) pipeline built around three domain-specific agents:

- **Technical**
- **Business**
- **Compliance**

Each domain is backed by its own vector store and prompt specialization. A central orchestrator coordinates the full pipeline:

1. classify the incoming user query
2. identify the relevant domains
3. decompose the query into domain-specific sub-queries
4. retrieve evidence from each domain’s knowledge base
5. collect domain answers and citations
6. synthesize a final response

This architecture is designed to demonstrate intelligent query routing, modular domain specialization, citation traceability, and conflict-aware response synthesis.

---

## 1. Query Classifier (`query_classifier.py`)

The `QueryClassifier` is responsible for the first stage of the pipeline.

Its responsibilities are:

- classify which knowledge domains are relevant to the user query
- assign confidence scores per domain
- generate domain-specific sub-queries when the original query spans multiple areas

This enables the system to avoid sending every query to every domain agent. Instead, only the relevant agents are invoked, which improves both relevance and efficiency.

### Why this component exists

Without a classifier, the system would either:
- retrieve from all domains unnecessarily, increasing noise
- or rely on a single shared retriever, reducing domain separation and explainability

The classifier therefore acts as the routing layer between the user query and the domain-specific RAG agents.

---

## 2. Domain Agents (`domain_agents.py`)

Each `DomainAgent` encapsulates the retrieval and answer-generation logic for one domain.

Its responsibilities are:

- receive a domain-specific sub-query
- retrieve relevant chunks from the domain’s vector store
- generate a focused answer using the domain prompt
- return both answer text and source citations

Each agent has:
- a dedicated retrieval scope
- a domain-specific prompt
- access only to the knowledge base of its assigned domain

### Why this component exists

The goal is to simulate domain-specialized expertise. Technical, business, and compliance domains often contain different terminology, priorities, and answer styles. Keeping them separate improves retrieval quality and creates a more realistic multi-agent design.

---

## 3. Vector Store Manager (`vector_store.py`)

The `VectorStoreManager` handles document ingestion, chunking, embedding, indexing, retrieval, and updates.

Its responsibilities are:

- load source documents for each domain
- split them into retrieval-friendly chunks
- generate embeddings
- build or load FAISS indices
- perform similarity search for queries
- support simulated knowledge updates by adding new content

The design uses **one FAISS index per domain** rather than one shared index.

### Why this component exists

Per-domain isolation keeps retrieval focused and makes it easy to:

- maintain clear boundaries between knowledge domains
- replace or update one domain independently
- avoid irrelevant cross-domain retrieval
- inspect domain-level behavior during testing and debugging

---

## 4. Orchestrator (`orchestrator.py`)

The `Orchestrator` is the central coordination component of the system.

Its responsibilities are:

- invoke the query classifier
- determine which domain agents should be called
- apply the dynamic retrieval strategy
- route sub-queries to the relevant agents
- collect agent responses and citations
- synthesize a unified final answer
- record feedback for trust-score tracking

The orchestrator is intentionally lightweight in terms of business logic. It does not perform retrieval itself; instead, it coordinates the specialized components.

### Why this component exists

The orchestrator is what makes the system multi-agent rather than a collection of unrelated retrievers. It provides the control layer that turns multiple independent domain responses into a single user-facing answer.

---

## 5. Shared Data Structures (`utils.py`)

`utils.py` contains the shared data structures and helper types used across the system.

- `Domain`
- `Citation`
- `AgentResponse`
- `ClassificationResult`
- `OrchestratorResult`
- `FeedbackStore`

These structures standardize how information is passed between components and make the pipeline easier to test and reason about.

---

# End-to-End Query Flow

The full request lifecycle works as follows:

1. **User submits a query**  
   The system receives a natural-language question.

2. **Query classification**  
   The `QueryClassifier` determines which domains are relevant and may decompose the query into specialized sub-queries.

3. **Dynamic retrieval sizing**  
   The `Orchestrator` computes a retrieval depth (`top_k`) based on:
   - query length
   - number of routed domains

4. **Domain routing**  
   The orchestrator sends each sub-query to the corresponding `DomainAgent`.

5. **Per-domain retrieval**  
   Each domain agent retrieves relevant chunks from its own FAISS index.

6. **Per-domain answer generation**  
   Each agent generates a domain-specific answer grounded in its retrieved context and returns citations.

7. **Response collection**  
   The orchestrator collects all agent responses and flattens citations into a single list.

8. **Final synthesis**  
   If only one domain is involved, the system returns that answer directly.  
   If multiple domains are involved, the orchestrator invokes the LLM again to synthesize the domain responses into one coherent answer.

9. **Feedback recording**  
   Optional user feedback can be recorded and converted into rolling trust scores for future synthesis context.

---

# Retrieval Design

## Per-Domain Vector Stores

The system uses one vector store per domain instead of one global index.

### Benefits

- cleaner domain boundaries
- simpler debugging
- better retrieval precision
- easier updates to one knowledge base without affecting others

This design also aligns well with the simulated multi-agent setup, where each agent represents a distinct knowledge domain.

## Chunking Strategy

Documents are split into chunks before embedding and indexing. The implementation uses recursive splitting so that chunk boundaries follow natural text structure more closely than fixed-size slicing.

This improves retrieval coherence because retrieved chunks are more likely to contain semantically meaningful information.

## Dynamic Retrieval Strategy

The orchestrator adjusts `top_k` dynamically based on query complexity.

The heuristic considers:
- **query length**: longer queries usually require broader context
- **number of routed domains**: multi-domain queries need stronger evidence per domain because the synthesis stage must compare and combine answers

This is a lightweight way to simulate adaptive retrieval behavior without adding a separate reranking or retrieval controller layer.

## Knowledge Updates

The system supports simulated knowledge updates by adding new documents or chunks into an existing domain’s vector store.

This demonstrates how the knowledge base can evolve without rebuilding the entire system from scratch.

## Synthesis and Conflict Resolution

The synthesis stage is responsible for converting multiple domain-specific responses into one final answer.

## Single-Domain Queries

If only one domain is relevant, the orchestrator returns that agent’s answer directly. This avoids an unnecessary extra LLM call and keeps the pipeline more efficient.

## Multi-Domain Queries

If multiple domains are involved, the orchestrator constructs a synthesis prompt containing:

- the original user question
- domain confidence scores from the classifier
- historical domain trust scores from the feedback mechanism
- each domain agent’s answer

The synthesis prompt instructs the model to:

- combine the answers coherently
- preserve domain-specific details
- highlight cross-domain connections
- flag contradictions explicitly
- prefer higher-confidence guidance when conflicts occur

## Why prompt-based conflict handling was chosen

Conflict resolution is handled primarily in the synthesis prompt rather than by a separate contradiction detection module.

This choice keeps the implementation simple and practical for the assignment. A more explicit semantic contradiction detector could be added later, but it would also increase complexity and risk false positives in cases where domain answers differ because they focus on different aspects of the same problem.

---

# Core Data Models

The most important data structures in the system are:

## `ClassificationResult`
Represents the classifier output:
- selected domains
- domain confidence scores
- domain-specific sub-queries

## `AgentResponse`
Represents the output of a domain agent:
- domain
- answer
- citations

## `Citation`
Represents traceable source information associated with retrieved content.

## `OrchestratorResult`
Represents the final system output:
- synthesized answer
- classification result
- all domain responses
- flattened citations

These models make the pipeline explicit and support both debugging and testing.

---

# Feedback Mechanism

The system includes a lightweight simulated feedback loop through `FeedbackStore`.

Its purpose is to:

- record per-domain quality scores after a query
- maintain rolling trust scores by domain
- expose those trust scores during synthesis

This does not fully retrain or tune the system, but it provides a simple mechanism to simulate how user feedback could influence domain weighting over time.

---

# Testing Approach

The test suite validates the main architectural behaviors of the system.

The tests cover:

- query classification behavior
- domain agent behavior
- orchestrator routing and synthesis
- vector store functionality
- assignment scenario coverage
- edge cases such as ambiguous queries, single-domain queries, and no-match queries

The goal of the tests is not only correctness, but also confidence that the components interact as intended.

---

# Technical Trade-offs

## 1. Per-domain isolation vs shared retrieval

Using one index per domain improves clarity and precision, but it may reduce flexibility compared with a more unified retrieval layer that can reason across all documents at once.

## 2. Prompt-based synthesis vs structured aggregation

The final answer is synthesized through an LLM prompt rather than a deterministic merging strategy. This improves fluency and allows conflict-aware summarization, but also makes the final step less deterministic.

---

# Limitations

Key limitations include:

- the system uses synthetic internal knowledge rather than real enterprise documents
- conflict resolution is prompt-driven rather than model-based or rule-based
- retrieval quality has limited tuning beyond chunking and `top_k`
- there is no dedicated reranking stage
- the system is not designed as a production deployment with APIs, auth, monitoring, or serving infrastructure

---

# Future Improvements

Several extensions would make the architecture stronger and more production-ready:

- add contextual compression to reduce irrelevant retrieved text
- introduce a reranking stage after initial retrieval
- replace prompt-only conflict handling with explicit contradiction checks
- expand the feedback mechanism into persistent retrieval weighting or evaluation-based tuning