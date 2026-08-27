# 20206RAG
RAG Colab Code

# Standard RAG Textbook Assistant 📚🤖

A lightweight, high-precision **Retrieval-Augmented Generation (RAG)** pipeline built with Python. This system connects an open-weight Large Language Model (`Llama-3.1-8B`) to a local vector database populated with high-quality educational data from the `FineWeb-Edu` dataset.

This repository demonstrates how to build a fully localized semantic search index and route contextual prompt packages to a free cloud LLM endpoint securely—fully accessible globally, including from region-restricted networks like Hong Kong.

---

## 🏗️ System Architecture

The core pipeline bypasses model hallucinations by forcing the LLM to follow an "open-book exam" workflow:

[User Query] ──> [1. Vector Embedding (MiniLM)] ──> [2. Local Search (FAISS Cosine)] ──> [3. Prompt Injection] ──> [4. Cloud LLM Generation]

1. **Local Ingestion**: Downloads a clean, streaming slice of educational data from Hugging Face.
2. **Dense Vector Mapping**: Uses `all-MiniLM-L6-v2` to transform text passages into 384-dimensional mathematical arrays.
3. **Cosine Similarity Search**: Normalizes vectors using `faiss.normalize_L2` and indexes them inside an inner product space (`IndexFlatIP`) for perfect semantic alignment.
4. **Context Grounding**: Strips away outside model knowledge by enforcing a strict system instruction matrix and a low temperature (`0.2`).

---

## 🛠️ Tech Stack & Key Components

* **Source Dataset:** [HuggingFaceFW/fineweb-edu](https://huggingface.co) (`sample-10BT`)
* **Vector Database:** [FAISS-CPU](https://github.com)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 Dimensions)
* **Generation LLM:** `meta-llama/Llama-3.1-8B-Instruct` (Hugging Face Serverless API)

---

## 🚀 Getting Started

### 1. Installation

Install the required core dependencies inside your Python environment or Google Colab notebook:

```bash
pip install datasets sentence-transformers faiss-cpu huggingface_hub
```

### 2. Configuration

Generate a free **User Access Token** (with `Read` permissions) from your Hugging Face Account Settings and add it to your pipeline:

```python
HF_TOKEN = "your_free_hugging_face_token_here"
```

### 3. Usage Example

Initialize the pipeline and submit standard text queries to pull grounded educational insights:

```python
# Execute the search and generation workflow
run_rag_query("What book or author is being discussed in these documents?", top_k=2)
```

---

## 🐛 Troubleshooting & Key Fixes Included

* **FAISS Capitalization Bug:** Corrected legacy function calls to use `faiss.normalize_L2` (capital **L**) to prevent method execution crashes.
* **Matrix Output Flattening:** Resolves `TypeError: only integer scalar arrays can be converted to a scalar index` by explicitly slicing the multidimensional matrix output arrays returned from `index.search()`.
* **Dynamic SDK Type Response Unpacking:** Includes an automated type-unpacker fallback block that correctly handles both standard dictionary outputs and complex Pydantic `ChatCompletionOutput` elements seamlessly across different versions of the `huggingface_hub` client.
