Markdown
# 20206RAG 📚🤖

### Metadata-Filtered Hybrid RAG & Structured LLM Pipeline

A production-grade, end-to-end Retrieval-Augmented Generation (RAG) and document metadata extraction pipeline. Built to bridge the gap between unstructured educational text parsing, attribute-filtered multi-stage search, and structured schema extraction, this codebase leverages modern vector indexing, sparse keyword lookup, cross-encoder reranking, and Google AI SDK primitives.

All components are optimized for minimal memory overhead and seamless execution within resource-constrained environments like Google Colab.

---

## 🗺️ Pipeline Architecture & System Workflow

The architecture operates as a multi-stage engine divided into structured metadata extraction, hybrid candidate selection, pre-filtering, and cross-encoder reranking before passing contextual payloads to the LLM layer:

[ Raw Educational Text ]

│

▼

[ Metadata Extraction (Gemini SDK + Pydantic Schema) ] ──► (Domain, Sub-topic, Audience)

│

▼

┌────────────────────────────────────────────────────────┐

│                        User Query Input                │

└────────────────────────────────────────────────────────┘

│

┌─────┴────────────────────────────────────────────────┐

▼                                                      ▼

[ Pathway A: Dense Retrieval ]             [ Pathway B: Sparse Retrieval ]


sentence-transformers                     - rank_bm25 (Okapi BM25)

FAISS Vector Index (L2 Norm)                  - Metadata Pre-Filtering

Metadata Pre-Filtering                        - Lexical Keyword Match

│                                                      │

└──────────────────────┬───────────────────────────────┘

▼

[ Dual-Candidate Pool Fusion & Deduplication ]

│

▼

[ Cross-Encoder Reranking (BAAI/bge) ]

│

▼

[ Top-K Context Assembly & Prompt Grounding ]

│

▼

[ Final LLM Generation via Google AI (gemini-3.1-flash-lite) ]



---

## 🛠️ Key Technical Modules

### 1. Schema-Driven Metadata Extraction Engine
Extracts granular academic attributes from raw text chunks into strict JSON formats to enable targeted downstream filtering.
* **Core Stack:** `google-genai` SDK, `Pydantic` (`BaseModel`), `gemini-3.1-flash-lite`
* **Key Innovation:** Enforces schema validity at the API layer via `response_schema` and Pydantic `Literal` enumerations. It safely fallbacks to regex string cleaning and custom object parsing if standard JSON payloads encounter edge-case whitespace errors.

### 2. Attribute-Prefiltered Hybrid Search
Solves the semantic drift problem in standard vector search by applying strict metadata constraints before similarity scoring.
* **Core Stack:** `faiss-cpu`, `rank_bm25`, `sentence-transformers`
* **Key Innovation:** Features a two-pass pre-filtering engine (`passes_filter`) that evaluates candidate metadata inline across both dense vector search and sparse keyword lookups. Over-fetching strategies (`overfetch_k`) ensure high candidate density even under aggressive filter conditions.

### 3. Cross-Encoder Context Reranking
Isolates maximum contextual precision from fused candidate pools.
* **Core Stack:** `BAAI/bge-reranker-base` / `SentenceTransformer` CrossEncoder
* **Key Innovation:** Evaluates query-document pairs jointly with full cross-attention scoring. This eliminates false-positive vector hits and selects only the top $K$ most informative contexts for the final prompt payload.

### 4. Grounded Response Generation
Generates strict, hallucination-free answers derived exclusively from retrieved search context.
* **Core Stack:** Google AI SDK (`types.GenerateContentConfig`), system instruction grounding
* **Key Innovation:** Uses system-level instructional boundaries to restrict response generation to retrieved context, returning standardized fallbacks (`I cannot find the answer in the provided documents.`) when knowledge gaps occur.

---

## ⚙️ Engineering & Runtime Patch Highlights

* **Robust JSON Extraction Fixes:** Eliminates `json.loads()` string parsing exceptions by integrating `response.parsed.model_dump()` alongside regex-based code-fence strippers (`re.sub`).
* **FAISS Over-fetching Strategy:** Implements adaptive top-$K$ over-fetching to maintain retrieval yield when metadata pre-filters drop non-matching items.
* **Google AI API Alignment:** Standardized configuration layers using `types.GenerateContentConfig` for unified handling of system instructions, temperature limits, max token bounds, and structured JSON outputs.

---

## 🚀 Quickstart Guide

### Prerequisites
Install the required dependencies:
```bash
pip install google-genai pydantic faiss-cpu rank-bm25 sentence-transformers numpy

```

Set your API Key:

Bash
export GEMINI_API_KEY="your-google-ai-api-key"
Basic Execution
Python
from google import genai

# Initialize Client
client = genai.Client()

# Run Metadata-Filtered Search & Generation
run_attribute_hybrid_rag_search(
    user_query="How do granitic intrusions form in Chile?",
    attribute_filter={"domain": "STEM"},
    top_n_candidates=5,
    final_top_k=2
)
