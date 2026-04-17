# NVIDIA Solutions Engineering Workshop — April 2026
### Saudi Energy | Generative AI with NVIDIA Technology


## Overview

This workshop covers practical, hands-on Generative AI use cases relevant to the energy sector using NVIDIA's inference infrastructure, foundation models, and APIs. Participants will build and run two end-to-end AI applications — a Retrieval-Augmented Generation (RAG) pipeline and a multi-agent data analysis assistant — and review production deployment blueprints for on-premises NVIDIA infrastructure.

No local GPU is required for the hands-on labs in the RAG folder.



---

## Workshop Contents

```
se_workshop_april_2026/
├── README.md                        ← You are here
├── examples/
│   ├── 5_mins_rag_no_gpu/           ← Lab 1: Minimal RAG pipeline
│   └── data-analysis-agent/         ← Lab 2: Multi-agent data 
    └── ai-podcast-assistant/        ← Lab 3: Audio and Image Processing
└── nims_blueprints/
    └── NVIDIA_NIMS_H200x2_V1.0.pdf  ← Reference: On-premises deployment blueprint
```

---

## Lab 1 — 5-Minute RAG (No GPU Required)

**Location:** [examples/5_mins_rag_no_gpu/](examples/5_mins_rag_no_gpu/)

A minimal, self-contained Retrieval-Augmented Generation application that demonstrates the full RAG pipeline in a single Python file, running entirely on NVIDIA's hosted APIs — no local GPU needed.

### What it does

1. **Ingest documents** — Upload files (PDF, DOCX, TXT, and more) via a Streamlit sidebar. Documents are loaded from disk using LangChain's `DirectoryLoader`.
2. **Chunk and embed** — Text is split into 512-character overlapping chunks and converted to semantic vectors using NVIDIA's `nv-embedqa-e5-v5` embedding model.
3. **Index with FAISS** — Chunks are stored in a FAISS vector index and serialized to disk as `vectorstore.pkl` for persistence across sessions.
4. **Answer questions** — User questions are matched against the index via similarity search; the top-k relevant chunks are injected as context into a prompt sent to `meta/llama3-70b-instruct`, which streams a grounded answer back through the UI.

### Models used

| Role | Model |
|------|-------|
| LLM (chat) | `meta/llama3-70b-instruct` via NVIDIA API Catalog |
| Embeddings | `nvidia/nv-embedqa-e5-v5` via NVIDIA API Catalog |



### Key concepts covered

- Document ingestion and chunking strategies
- Semantic vector embeddings and similarity search
- Context-augmented prompting (RAG pattern)
- Cloud-hosted inference with NVIDIA API Catalog

---

## Lab 2 — Multi-Agent Data Analysis Assistant

**Location:** [examples/data-analysis-agent/](examples/data-analysis-agent/)

A production-style multi-agent application that lets users upload any CSV dataset and ask natural language questions — receiving data query results, generated plots, and reasoned explanations powered by NVIDIA's Nemotron reasoning models.

### What it does

Users upload a CSV file (a Titanic dataset is included as an example), then ask questions in natural language such as:
- *"Show me the age distribution"*
- *"What is the average fare by passenger class?"*
- *"Plot survival rate by gender"*

The application routes the question through a pipeline of five specialized agents to produce a complete, explained answer.

### Agent architecture

| Agent | Role | Temperature |
|-------|------|-------------|
| **QueryUnderstandingAgent** | Classifies intent: visualization vs. data query | 0.1 (deterministic) |
| **CodeGenerationAgent** | Generates Python (pandas / matplotlib) code | 0.2 (precise) |
| **ExecutionAgent** | Runs generated code in a sandboxed namespace | N/A |
| **ReasoningAgent** | Streams a natural-language explanation with chain-of-thought | 0.6 (fluent) |
| **DataInsightAgent** | Generates dataset summary and suggested questions on upload | 0.5 |

### Models used

| Model | Parameters | Best for |
|-------|-----------|----------|
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | 253B | Complex multi-step reasoning |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | 49B | Lower latency, production deployments |

Both models support NVIDIA's extended thinking mode (`<think>...</think>` tokens), which is surfaced in the UI as a collapsible "Model Thinking" panel.

### UI layout

- **Left panel (30%)** — Model selector, CSV uploader, dataset preview, LLM-generated dataset insights
- **Right panel (70%)** — Chat interface with message history, inline plots, collapsible thinking blocks, and generated code


### Key concepts covered

- Multi-agent orchestration and intent routing
- LLM-generated code execution in sandboxed environments
- Chain-of-thought / extended thinking with Nemotron models
- Temperature tuning per agent role
- Conversation context management across turns
- Streaming responses with real-time token display

## Lab 3 — AI Podcast Assistant (Audio & Image Processing)

**Location:** [examples/ai-podcast-assistant/](examples/ai-podcast-assistant/)

An end-to-end multimodal application that processes podcast audio files and images using the **Phi‑4 Multimodal LLM** (5.6B parameters) through NVIDIA NIM Microservices. The model handles text, audio, and images natively within a single API — no separate transcription service needed.

### What it does

#### Audio pipeline
Users provide any MP3 podcast or audio recording. The application:

1. **Chunks the audio** — Long files are split into 30-second segments to stay within API payload limits
2. **Transcribes each chunk** — Each segment is base64-encoded and sent to Phi‑4 for accurate transcription
3. **Refines into structured notes** — Raw transcription is reformatted into bullet-pointed, well-organized notes
4. **Summarizes** — A concise summary is generated from the detailed notes
5. **Translates** — Both notes and summary are translated into any target language while preserving structure and formatting
6. **Exports** — Results are saved as `.txt` files for sharing and reference

#### Image pipeline (bonus use cases)
The same Phi‑4 model processes images via three demonstrated use cases:

| Use Case | Description | Output |
|----------|-------------|--------|
| **Image Description** | Detailed natural-language caption of any image | `image_description.txt` |
| **Visual Q&A** | Ask targeted questions about image content | Printed Q/A pairs |
| **Text Extraction (OCR)** | Read and transcribe text embedded in images, preserving layout | `image_extracted_text.txt` |

### Model used

| Model | Parameters | Inputs | Context |
|-------|-----------|--------|---------|
| `microsoft/phi-4-multimodal-instruct` via NVIDIA NIM | 5.6B | Text, Audio, Image | 128K tokens |

### Key concepts covered

- Multimodal inference with a single unified model (text + audio + image)
- Long-audio chunking and reassembly strategies
- Base64 encoding of binary media for REST API transmission
- Prompt engineering for transcription, summarization, and translation tasks
- Vision API format for image understanding (OpenAI-compatible content arrays)
- Python venv isolation for Jupyter notebook reproducibility

---

## Why No Local GPU Is Required

All three labs offload GPU-intensive inference to **NVIDIA's cloud APIs**. Your local machine only runs lightweight Python — making HTTP requests, processing text, and rendering the UI.

| Lab | Local workload | Remote (NVIDIA Cloud) |
|-----|---------------|----------------------|
| Lab 1 — RAG | FAISS vector search (CPU), Streamlit UI | Llama 3 70B (LLM) + nv-embedqa-e5-v5 (embeddings) |
| Lab 2 — Multi-Agent | Agent orchestration logic, Streamlit UI | Nemotron 253B / 49B (reasoning + code generation) |
| Lab 3 — Podcast Assistant | Audio chunking, base64 encoding, Jupyter UI | Phi‑4 Multimodal (transcription, summarization, image analysis) |

The pattern is the same across all three:

```
Your laptop                    NVIDIA Cloud (API Catalog / NIM)
─────────────                  ────────────────────────────────
Python + CPU only  ──HTTPS──►  GPU cluster runs the model
                   ◄──────────  Returns text response
```

The only local requirements are **Python**, a few pip packages, and an **NVIDIA API key**. Models like Nemotron 253B or Phi‑4 Multimodal would require multiple H200 GPUs to run locally — the API Catalog makes them accessible from any laptop.

---

## NVIDIA NIMs = Deployment Blueprint

**Location:** [nims_blueprints/NVIDIA_NIMS_H200x2_V1.0.pdf](nims_blueprints/NVIDIA_NIMS_H200x2_V1.0.pdf)

Hardware and architecture reference for deploying NVIDIA NIM (Neural Inference Microservices) on-premises using dual H200 GPU configurations. This document covers:

- Recommended server hardware specifications
- NIM microservices architecture for production inference
- Sizing guidance for energy-sector LLM workloads
- Integration patterns for enterprise environments

This is the deployment path for organizations that need to run inference within their own data centers due to data sovereignty, latency, or compliance requirements — common in the energy sector.

---
