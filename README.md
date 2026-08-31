# 20206RAG
RAG Colab Code

# Enterprise Hybrid-RAG Textbook Search Engine 📚🔍

An advanced **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** system built with Python, engineered to bypass the structural blind spots of traditional vector search engines. By combining dense semantic embeddings with sparse keyword tokenization, this system accurately indexes and queries complex technical definitions, jargon, and academic context from the `FineWeb-Edu` dataset.

---

## 🏗️ System Architecture

Traditional vector-only RAG often fails to catch explicit, precise keywords (like exact definitions, acronyms, or proper nouns). This enterprise pattern uses a parallel search and rank topology:

┌──> [1. Semantic Search: FAISS (Cosine)] ──┐

[User Query] ─────────┤├──> [3. Cross-Encoder Reranker] ──> [4. Llama 3.1 Study Guide]

└──> [2. Keyword Search: BM25 (Exact)] ─────┘
1. **Dense Retrieval (Concepts)**: Uses `all-MiniLM-L6-v2` and normalized **FAISS** index vectors to locate broad conceptual matches, synonyms, and search intent via Cosine Similarity.
2. **Sparse Retrieval (Keywords)**: Uses the statistical **BM25 algorithm** (`rank_bm25`) to run exact string matches across text document tokens, pulling precise terms.
3. **Cross-Attention Reranking**: Merges the results from both pathways, deduplicates them, and feeds them through a localized **Cross-Encoder model** (`bge-reranker-base`) to score structural contextual relevance relative to the query.
4. **Structured Compilation**: Passes the highest-scoring reranked text segments into a custom prompt wrapper handled by `Llama-3.1-8B-Instruct` via the **Hugging Face Serverless API** to output an executive Study Guide.

---

## 🛠️ Tech Stack & Key Components

* **Source Dataset:** [HuggingFaceFW/fineweb-edu](https://huggingface.co) (`sample-10BT`)
* **Vector DB (Semantic):** [FAISS-CPU](https://github.com)
* **Keyword Index (Lexical):** [Rank-BM25](https://github.com)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 Dimensions)
* **Reranker Model:** `BAAI/bge-reranker-base` (Cross-Encoder attention scoring) [3]
* **Generation LLM:** `meta-llama/Llama-3.1-8B-Instruct` (Hugging Face Serverless Endpoint)

---

## 🚀 Getting Started

### 1. Installation

Install the entire hybrid text retrieval and validation stack in your Python environment or Google Colab notebook:

```bash
pip install datasets sentence-transformers faiss-cpu rank_bm25 huggingface_hub
```

### 2. Configuration

Make sure your free Hugging Face **User Access Token** is authorized and referenced in your engine execution script:

```python
HF_TOKEN = "your_free_hugging_face_token_here"
```

### 3. Running the Engine

Submit your complex academic questions to kick off parallel routing and compile a custom study guide:

```python
# Query your hybrid textbook engine
run_hybrid_rag_search(
    user_query="What are the specific definitions and characteristics of thermodynamic processes?",
    top_n_candidates=10,
    final_top_k=2
)
```

---

## 🎯 Production Engineering Highlights

* **Candidate Pool Fusion:** Collects the top `N` candidates from both dense and sparse indexes, merging them via a Python `set` operation to safely eliminate duplicate entries before passing them to the reranker.
* **Cross-Encoder Attention vs. Bi-Encoders:** Rather than comparing isolated vector distances independently, the `bge-reranker` processes the query and document together through an attention matrix. This provides a highly accurate contextual score at the cost of slight local computational overhead.
* **Asynchronous Serverless Generation:** Offloads the final heavy text synthesis step to cloud endpoints, keeping local RAM requirements tiny enough to run on standard, free-tier hardware environments.
