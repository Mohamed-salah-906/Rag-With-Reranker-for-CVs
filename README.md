# CV RAG + Re-ranking (SmolLM3 + Chroma + FlashRank)

A lightweight, local-first Retrieval-Augmented Generation (RAG) app for screening **CVs**.  
It ingests PDFs from a folder, indexes them in a vector store, retrieves relevant chunks for a job/skills query, **re-ranks** the results with a cross-encoder, and then:
- **Ranks entire CV files** by how well they match the query.
- Provides an **Ask (RAG)** mode to answer questions grounded only in CV content.

Built with **Streamlit**, **LangChain**, **ChromaDB**, **Ollama** (LLM + embeddings), and **FlashRank** (cross-encoder reranker).

---

## Features
- 📂 Read all CV PDFs from `CVs/`
- 🔎 Multi-query retrieval (synonyms/abbreviations/role variants)
- 🥇 Cross-encoder **re-ranking** for better precision
- 🧮 **CV-level scoring** (composite of best + average chunk scores)
- 💬 RAG answer mode grounded in retrieved text
- 🧰 CPU-first configuration for low-VRAM machines

---

## Quickstart

> **Windows tip:** if your C: drive is small, move caches to D: (see _Disk & Cache_ below).

### 1) Install dependencies
```bash
pip install --upgrade streamlit langchain langchain-community chromadb flashrank pypdf
```

### 2) Pull local models for Ollama
```bash
ollama pull smollm3          # chat LLM
ollama pull nomic-embed-text # embeddings
```

### 3) (Recommended) Run on CPU
Avoid GPU VRAM errors:
```powershell
# PowerShell (temporary for this window)
$env:OLLAMA_NO_GPU="1"

# Or set permanently
setx OLLAMA_NO_GPU "1"
```

### 4) Launch
Put your PDFs under `CVs/` then run:
```bash
streamlit run cv_rag_rerank_app.py
```

---

## Project Structure
```
.
├─ CVs/                       # your CV PDFs
├─ chroma_db/                 # Chroma persistence (configurable)
├─ cv_rag_rerank_app.py       # Streamlit app
├─ README.md                  # this file
└─ docs/
   └─ RAG_Deep_Dive.md        # detailed documentation
```

---

## Configuration (edit in `cv_rag_rerank_app.py`)
```python
DATA_PATH   = "CVs"
PERSIST_DIR = "chroma_db"                # move off C: if needed
EMBED_MODEL = "nomic-embed-text"         # Ollama embedding model
LLM_MODEL   = "smollm3"                  # no fallback
RERANK_MODEL = "ms-marco-MultiBERT-L-12" # FlashRank model (multilingual)

TOPK_INITIAL = 20   # retrieved before re-rank (smaller = faster, less recall)
TOPK_RERANK  = 10   # chunks kept after re-rank
TOPK_CVS     = 10   # CV files shown in the ranking
```
If you’re on a tiny machine, keep `TOPK_*` small.

---

## Usage

### Rank CVs
- Select **“Rank CVs”**
- Enter a job title and/or skills (e.g., “Senior Data Scientist, NLP, LangChain, Kubernetes, German”)
- The app shows top CVs with evidence snippets

### Ask (RAG)
- Select **“Ask (RAG)”**
- Ask focused questions: “Which candidate has Kubernetes + MLOps?”, “Who has B2 German?”

---

## Disk & Cache (Windows)

If `C:` is low on space:
```powershell
# Move Ollama models to D:
setx OLLAMA_MODELS "D:\Ollama\models"

# Move HuggingFace cache (FlashRank uses this) to D:
setx HF_HOME "D:\hf_cache"
```
Restart Ollama/terminal after setting these, then re-run the app.

---

## Troubleshooting

### 1) `TypeError: MultiQueryRetriever.from_llm() got an unexpected keyword argument 'vectorstore'`
Use `retriever=vdb.as_retriever(...)` instead of `vectorstore=...`. This app already does that.  
If you still see this, upgrade LangChain:
```bash
pip install -U langchain langchain-community
```

### 2) `cudaMalloc failed: out of memory`
You’re trying to run on GPU without enough VRAM. Force CPU:
```powershell
setx OLLAMA_NO_GPU "1"
```
(Or in code via `model_kwargs={"num_gpu": 0}` for `ChatOllama` and `OllamaEmbeddings`.)

### 3) FlashRank 404 when downloading a reranker
Use a **supported model**, e.g.:
- `ms-marco-MultiBERT-L-12` (multilingual, ~150MB)
- `ms-marco-MiniLM-L-12-v2` (English-focused, very small/fast)

Upgrade FlashRank:
```bash
pip install -U flashrank
```

### 4) Index won’t persist / wrong location
Change `PERSIST_DIR` to a drive with space, delete old `chroma_db`, and rerun:
```bash
rmdir /S /Q chroma_db  # Windows
```

---

## How it Works (high level)
1. **Ingest & chunk** CV PDFs (chunk_size ~800, overlap ~120).
2. **Embed** each chunk with `nomic-embed-text` and store vectors in **Chroma**.
3. **Multi-query retrieve:** expand your query into several variants and fetch top-`k` chunks for each; merge results.
4. **Re-rank** candidates with a cross-encoder (`ms-marco-MultiBERT-L-12`) to score true semantic relevance.
5. **Score CV files:** aggregate chunk scores per file and compute a composite score (see details in `docs/RAG_Deep_Dive.md`).
6. **RAG answer:** format the top chunks into a prompt and ask **SmolLM3** to answer concisely from context only.

For a deeper explanation (similarity math, reranker theory, and tuning), read `docs/RAG_Deep_Dive.md`.
