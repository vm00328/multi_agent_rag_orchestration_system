**Design Desicions:**

1. **Embedding Model**

The Qwen3-Embedding-0.6B model is chosen due to it ranking very high on the MTEB leaderboard and being exceptionally lightweight compared to other models that achieve similar performance.

It offers strong, instruction-aware performance (rank #16 on MTEB) despite its lightweight size, supporting over 100 languages, 32k context, and user-defined dimensions for high-speed, cost-effective embeddings.

In addition, it is open-source which offers flexibility and cost savings. A proprietary embedding model increases the risk of vendor lock-in and higher costs. 

2. **Embedding Normalization**

Embedding normalization scales vector embeddings to a consistent unit length (L2 norm), preserving their direction while making similarity comparisons (cosine similarity) consistent and fair. It is crucial for preventing popularity bias, improving similarity search accuracy, and ensuring that results are driven by vector direction rather than magnitude.

3. **Vector Database**

FAISS is highly optimized for performance, scalable, memory efficient and serverless.

4. **Text Splitter**

Choice: RecursiveCharacterTextSplitter

Method: Attempts to split using a sequence of characters ("\n\n", "\n", " ", "") recursively until the chunk size is met.

Pros: Highly recommended by and as it keeps paragraphs, sentences, and words together, leading to better retrieval results.

Cons: Slightly slower than Character Text Splitter.


**Limitations:**

1. **Contextual Compression**

*Example: text is divided into equal parts (chunks), each 800 characters long. Given a query, the retriever returns the relevant chunk. This chunk, however, might contain both relevant and irrelevant information. Incorporating unrelated content in the LLM prompt can be problematic because it may distract the LLM from focusing on essential details and it consumes space in the prompt that could be allocated to more relevant information.*

With compression, only the information relevant to the query is returned. Compressing here refers to using an LLM to rewrite the retrieved chunk so that it contains only information relevant to the query. This way, the chunks are smaller, and more chunks can be used as contextual information to generate the final answer.

2. **Chunk Size & Overlap**

Due to the time constraint of the project, experimenting with different chunk sizes is limited.