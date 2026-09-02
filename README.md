# 20206RAG
RAG Colab Code

# Parameter-Efficient Teacher Model Fine-Tuning (QLoRA) 🎓🔥

This module demonstrates how to permanently alter the internal neural network communication weights of a foundational Large Language Model (`Gemma-2-2b-it`). By executing **Supervised Fine-Tuning (SFT)** on high-density data from the `FineWeb-Edu` dataset, the model shifts its default vocabulary, structure, and persona to act inherently like a disciplined, professional educator.

Unlike Retrieval-Augmented Generation (RAG) which passes temporary notes to a frozen model, fine-tuning modifies the model's actual base brain to adopt a specific tone without needing external context manuals injected into every prompt.

---

## 🏗️ Deep Learning Architecture

Full parameter fine-tuning of multi-billion parameter networks requires hundreds of gigabytes of VRAM. This pipeline implements **QLoRA (Quantized Low-Rank Adaptation)** to compress the workload by 75%, allowing full training to execute on a single consumer or free cloud GPU (such as an NVIDIA T4 node).

[Base Model Weights] ──(Frozen in 4-bit NF4 Precision)──┐

├──> [SFTTrainer Gradient Update Loop]

[LoRA Adapter Layers]──(Active 16-bit BFloat Learning)──┘



1. **4-Bit Quantization**: Compresses the base model down to **NormalFloat 4 (NF4)** parameters via `bitsandbytes`. This drops the VRAM baseline footprint drastically so it fits easily within a ~15GB memory envelope.
2. **LoRA Adapters**: Freezes the base model completely. Instead of optimizing all 2 billion parameters, it injects small, low-rank matrix pairs (Rank `r=16`, `Alpha=32`) into the target attention projection layers (`q_proj`, `v_proj`). Only these tiny adapter layers learn during training.
3. **Double Quantization**: Compresses the quantization constants themselves, reclaiming an extra `0.4 GB` of vital VRAM cache headroom.

---

## 🛠️ Tech Stack & Key Components

* **Base Foundational Model:** [google/gemma-2-2b-it](https://huggingface.co) (Gated Repo)
* **Training Dataset:** [HuggingFaceFW/fineweb-edu](https://huggingface.co) (`sample-10BT` text split)
* **Optimization Framework:** Hugging Face `trl` (`SFTTrainer` & `SFTConfig`)
* **Quantization & Adaption Backend:** `peft` & `bitsandbytes`

---

## 🚀 Getting Started

### 1. Ingestion Prerequisites

Install the underlying deep learning compilation stack inside your environment or Google Colab notebook:

```bash
pip install datasets transformers trl peft accelerate bitsandbytes
```
*CRITICAL: Ensure your runtime hardware accelerator is switched to **T4 GPU** before execution. Running this script in a standard CPU layout will result in an immediate runtime crash.*

### 2. Gated Credential Authentication

Because `gemma-2-2b-it` is a gated model repository, ensure you have clicked "Accept" on its Hugging Face page. Then, pass a classic Hugging Face token with **Write Permissions** directly into your variable block:

```python
HF_TOKEN = "your_classic_write_permission_token_here"
```

### 3. Execution Lifecycle

Run the script cell. The `SFTTrainer` will ingest 500 textbook passages, convert them to native datasets via `Dataset.from_list()`, map the text structural instructions row-by-row, and kick off the 50-step optimization loop.

The training phase naturally requires intense backward gradient calculus passes and typically completes in **6 to 10 minutes** on a standard free-tier T4 GPU.

---

## 🎯 Production Engineering Highlights & Framework Patches

The codebase contains strict structural engineering patterns to align with the absolute latest Hugging Face TRL library versions:
* **Row-by-Row Dataset Mapping:** Replaces legacy bulk array functions with a granular `.map(formatting_prompts_func)` wrapper. This fixes internal `trl` validation `AttributeError` conflicts by preventing tokenization utilities from checking list objects instead of primitive text strings.
* **Unified SFTConfig Class Migration:** Adapts to the modern TRL API framework by moving positional argument limits (like `max_length`) out of the raw trainer initialization block and into the dedicated `SFTConfig` parameter layout.
* **Processing Class Standardization:** Upgrades the legacy `tokenizer` parameters to use the modern, multi-modal compliant `processing_class` variable argument tags.
* **Prompt Slice Inference Output:** Features an optimized post-training validation block (`outputs[inputs["input_ids"].shape:]`) that cleanly slices away input conversation sequences to ensure only the raw generated teacher answers print directly onto your console logs.

---

Developer: Jovi Zhu

GitHub: @JoviWZhu

LinkedIn: https://www.linkedin.com/in/jovizhu
