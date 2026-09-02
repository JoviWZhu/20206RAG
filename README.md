# 20206RAG
RAG Colab Code

# Open Educational AI Stack: From RAG to QLoRA Fine-Tuning 📚🤖

A complete, production-grade curriculum demonstrating modern architectures in Large Language Model (LLM) engineering. This repository spans four evolutionary stages of development—progressing from basic semantic information retrieval to interactive pipelines, high-precision hybrid enterprise search engines, and parameter-efficient model training.

All pipelines utilize high-density natural language open data from the [HuggingFaceFW/FineWeb-Edu](https://huggingface.co) dataset and are optimized for global deployment, including region-restricted runtimes.

---

## 🗺️ Project Roadmap & Directory Layout

The repository is organized into four standalone, production-ready modules:

Use code with caution.├── 📁 1-standard-rag/      # Semantic Dense Vector RAG via FAISS & Cosine Similarity├── 📁 2-quiz-bot/          # Dynamic "Quiz Me" Bot utilizing Multi-Angle Prompting├── 📁 3-hybrid-search/     # Enterprise Dense-Sparse Retrieval with BGE Cross-Attention Reranking└── 📁 4-teacher-sft/       # 4-Bit Parameter-Efficient Fine-Tuning (QLoRA) using SFTTrainer
---

## 🛠️ Deep Dive: The Four Core Milestones

### 1. Standard RAG Textbook Assistant (`/1-standard-rag`)
Implements a baseline vector-retrieval pipeline. It maps raw textbook data into a local vector database and uses an "open-book exam" prompt template to ground model outputs and eliminate hallucinations.
* **Core Stack:** `sentence-transformers/all-MiniLM-L6-v2` (384 Dimensions), `faiss-cpu` (`IndexFlatIP`).
* **Key Innovation:** Uses strict L2 normalization boundary vectors to transition standard Euclidean distances into pure Cosine angular searches.

### 2. Interactive "Quiz Me" Bot (`/2-quiz-bot`)
An interactive study companion that transforms static data rows into an endless testing game loop. It dynamically drafts multi-tiered questions and executes factual grading.
* **Core Stack:** Native Python random utility loops, `meta-llama/Llama-3.1-8B-Instruct`.
* **Key Innovation:** **Multi-Angle Prompting** matrix. It rotates through diverse pedagogical profiles (Definition tests, Misconception tracking, Real-World application) to stretch a minimal memory dataset footprint into an infinitely varied game lifecycle.

### 3. Enterprise Hybrid Search Engine (`/3-hybrid-search`)
Addresses the structural blind spots of standard vector databases by running dense semantic searches and lexical keyword searches in parallel.
* **Core Stack:** `rank_bm25` (Okapi framework), `BAAI/bge-reranker-base` Cross-Encoder.
* **Key Innovation:** Uses dual-path processing to catch exact technical jargon/proper nouns, combining results into an attention-based reranker to isolate maximum contextual density before text generation.

### 4. Gated "Teacher" Model Fine-Tuning (`/4-teacher-sft`)
Steps away from prompting constraints to permanently alter an AI model's internal neural network weights. It trains a base model to inherently communicate with the tone, discipline, and style of a professional educator.
* **Core Stack:** `google/gemma-2-2b-it`, Hugging Face `trl` (`SFTTrainer` & `SFTConfig`), `peft` (LoRA/QLoRA), `bitsandbytes`.
* **Key Innovation:** Implements 4-bit NormalFloat (`nf4`) quantization configs to run a complete optimization loop comfortably within free-tier consumer or cloud GPU environments (like a Google Colab T4 GPU node).

---

## ⚙️ Global Technical Fixes & Engineering Lessons Included

This repository maintains rigorous compliance with the latest Hugging Face API updates. Key patches natively handled in the code tree include:
* **FAISS Capitalization Normalization:** Patched legacy function mismatches by mapping strict case-sensitive `faiss.normalize_L2` methods.
* **Matrix Output Unpacking:** Resolves indexing TypeErrors by flattening multidimensional arrays returned from raw FAISS search executions.
* **SDK Response Type Fallbacks:** Integrates runtime conditional type-checking to parse both object attributes and raw dictionary schema outputs returned across changing serverless network routers.
* **Modern TRL Framework Alignment:** Fully adapted to modern Hugging Face tokenization architectures by migrating sequence processing arguments into unified `SFTConfig` layouts and updating token tracking arrays to use explicit `processing_class` properties.

---

## 🚀 Setup & Global Configuration

### 1. Ingestion Requirements
Install the unified library matrix inside your target execution node:
```bash
pip install datasets transformers trl peft accelerate bitsandbytes sentence-transformers faiss-cpu rank_bm25 huggingface_hub
```

### 2. Authorization Setup
To execute training loops against gated weight trees (like Gemma), ensure your environment reads a classic Hugging Face token with active `Write` permissions:
```python
# Pass directly to pretrained parameters to bypass proxy barriers
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", token="your_classic_write_token_here")
```
Developer: Jovi Zhu

GitHub: @JoviWZhu

LinkedIn: https://www.linkedin.com/in/jovizhu
