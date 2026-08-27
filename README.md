# 20206RAG
RAG Colab Code

# Interactive AI "Quiz Me" Bot 🎓🤖

An interactive, educational quiz system that turns static textbook data from the `FineWeb-Edu` dataset into a dynamic, endless study companion. It generates diverse academic questions and handles factual grading using a Retrieval-Augmented Generation (RAG) framework.

This project features a **Multi-Angle Prompting** strategy, allowing a modest data footprint (e.g., 100 documents) to yield hundreds of unique question variations.

---

## 🏗️ How It Works

The system splits the interactive lifecycle into two distinct execution steps:

### 1. Question Generation Phase

[Random Passage Pick] ──> [Random Profile Select (e.g., Real-World Application)] ──> [LLM Generates Question]

The script picks a random chunk of text from the database and pairs it with a randomly selected pedagogical style (Definitions, Misconceptions, Real-World Applications, or Cause/Effect). This is sent to the LLM to format a unique question.

### 2. Rigorous Grading Phase

[User Types Answer] ──> [Hidden Source Context Injected] ──> [LLM Evaluates Factuality] ──> [Grounded Grade & Feedback]When you submit an answer, the backend acts as a strict academic grader. It references the hidden source text passage, checks for factual alignment, flags missing components, and gives structural feedback labeled with `[CORRECT]`, `[PARTIALLY CORRECT]`, or `[INCORRECT]`.

---

## 🛠️ Tech Stack & Key Components

* **Source Dataset:** [HuggingFaceFW/fineweb-edu](https://huggingface.co) (`sample-10BT`)
* **Core Engine Framework:** Native Python `random` allocation utilities
* **Generation & Grading LLM:** `meta-llama/Llama-3.1-8B-Instruct` via the Hugging Face Serverless API

---

## 🚀 Getting Started

### 1. Installation

Install the required core libraries inside your execution environment or Google Colab notebook:

```bash
pip install datasets huggingface_hub
```

### 2. Configuration

Make sure you add your free Hugging Face **User Access Token** to authenticate the API streams:

```python
HF_TOKEN = "your_free_hugging_face_token_here"
```

### 3. Playing the Quiz Loop

Run the question generator, capture user input natively inside your console, and feed it into the evaluation processor:

```python
# Step 1: Spin up a fresh question from a random textbook angle
generate_quiz_question()

# Step 2: Open an interactive text box right inside your cell
your_response = input("Type your answer to the question above: ")

# Step 3: Let the RAG model grade your response against the hidden textbook data
grade_quiz_answer(your_response)
```

---

## 🎯 Optimization Features

* **Multi-Angle Prompting:** Includes an automated matrix of multiple "professor profiles" (e.g., isolating core definitions vs. searching for conceptual misconceptions). This ensures that even if a document is selected twice, the resulting question feels completely fresh.
* **Low-Temperature Grading:** Locks down the grading execution framework to a low temperature setting (`0.2`). This prevents the LLM from being overly lenient or hallucinating grading rules outside of the provided source textbook context.
