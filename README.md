# 20206RAG
RAG Colab Code

# 🤖 Dual-Module Retrieval-Augmented Generation (RAG) Suite

An open-source, modular GenAI repository demonstrating end-to-end **Retrieval-Augmented Generation (RAG)** architectures for conversational Q&A and automated structured assessment generation.

---

## 🌟 Overview

This repository contains two independent, production-oriented RAG applications developed in **Google Colab** and organized into dedicated, self-contained sub-modules:

1. **[RAG Bot](./rag-bot)** — A conversational Q&A engine engineered for semantic search, context retrieval, and grounded response generation over custom document collections.
2. **[RAG Quiz](./rag-quiz)** — An automated question-and-assessment generator that ingests documents to produce context-verified quizzes, answers, and evaluation rubrics in structured formats (JSON/Markdown).

## 🗺️ Project Roadmap & Future Enhancements

- [ ] **Hybrid Search:** Combine sparse (BM25) and dense vector search to improve keyword-specific retrieval accuracy.
- [ ] **Advanced RAG Patterns:** Integrate re-ranking models (Cross-Encoders) and contextual compression.
- [ ] **Automated RAG Evaluation:** Implement evaluation frameworks (e.g., Ragas) to measure context precision, recall, and answer faithfulness.
- [ ] **API & UI Deployment:** Wrap execution pipelines in a FastAPI backend with a lightweight Streamlit web interface.

---

## 📂 Repository Architecture

```text
.
├── README.md               # Top-level landing page & suite overview
├── rag-bot/                # Module 1: Conversational Q&A Bot
│   ├── README.md           # Module-specific documentation & architecture
│   └── rag_bot.ipynb       # Main Colab execution notebook
└── rag-quiz/               # Module 2: Automated Quiz & Assessment Engine
    ├── README.md           # Module-specific documentation & schema formats
    └── rag_quiz.ipynb      # Main Colab execution notebook

```
Developer: Jovi Zhu

GitHub: @JoviWZhu

LinkedIn: https://www.linkedin.com/in/jovizhu
