# LlamaIndex Professional Tutorials

> Comprehensive hands-on tutorials covering LlamaIndex from fundamentals to advanced production patterns
> **Refreshed November 2026** — llama-index-core 0.14.x, current-generation models (GPT-6.1 family, Claude Sonnet 5.5), and modern evaluation APIs.

## Overview

This tutorial series is designed for **ML engineers familiar with embeddings** who want to master LlamaIndex for building production-ready RAG (Retrieval-Augmented Generation) systems. The tutorials use the latest LlamaIndex modular architecture and cover everything from basic setup to advanced hybrid search and reranking techniques.

## What's New (November 2026 Refresh)

- **Models**: `gpt-4o-mini` → **`gpt-6.1-terra`** (workhorse) / **`gpt-6.1-sol`** (evaluation/quality) / `gpt-6-luna` (budget). `gpt-4o-mini` shuts down 2027-01-14; `gpt-3.5-turbo` and `gpt-4` were retired 2026-10-23.
- **Cohere rerank**: `rerank-english-v3.0` → **`rerank-v3.5`** (Rerank 4/5 also available).
- **AWS Bedrock tests**: Claude Sonnet 4.5 → **Claude Sonnet 5.5** (`global.anthropic.claude-sonnet-5-5`).
- **Evaluation**: migrated to `llama_index.core.evaluation` (`FaithfulnessEvaluator`/`RelevancyEvaluator` with `llm=`, `RetrieverEvaluator` with `HitRate`/`MRR`); `ServiceContext` fully removed (it now raises in core 0.14).
- **Custom query engines**: `BaseQueryEngine` now requires `_get_prompt_modules()` — the Notebook 5 `ProductionQueryEngine` has been fixed accordingly.
- **Dependencies**: minimum version pins added; `fastembed`, `cohere`, `boto3`, `nest-asyncio` now declared; new `pyproject.toml`.
- **Repo cleanup**: stale notebook duplicates removed; `.gitignore` fixed (no longer ignores README/extras).

## Target Audience

- **Prerequisites**:
  - Python 3.12+ proficiency
  - Understanding of embeddings and vector similarity search
  - Familiarity with ML/NLP concepts (transformers, semantic search)

- **Assumed Knowledge**:
  - What embeddings are and how they work
  - Vector similarity concepts (cosine similarity, dot product)
  - Basic understanding of transformers and LLMs

- **What You'll Learn**:
  - LlamaIndex-specific APIs and best practices
  - Production-ready RAG system implementation
  - Performance tuning and optimization strategies
  - Cost optimization and scalability considerations

## Learning Path

The tutorials are organized into 5 progressive notebooks (Notebooks 1–3 live in the repo root; Notebooks 4–5 are in `extras/`):

### 📘 Notebook 1: `01_setup_and_basics.ipynb`
**Difficulty**: Beginner | **Time**: 45-60 minutes

- LlamaIndex architecture and modular ecosystem
- Installation with latest modular packages
- Global Settings configuration
- Create your first VectorStoreIndex
- Execute basic queries and analyze responses

**Practical Example**: Simple Document QA System

### 📗 Notebook 2: `02_documents_and_chunking.ipynb`
**Difficulty**: Beginner-Intermediate | **Time**: 90-120 minutes

- Loading documents from multiple sources (PDFs, web)
- Document preprocessing and cleaning
- Chunking strategies: sentence, token, semantic
- Metadata management (custom metadata, filtering)
- Node relationships and hierarchies
- Batch embedding optimization

**Practical Example**: Research Paper Processing Pipeline

### 📙 Notebook 3: `03_indexing_and_simple_queries.ipynb`
**Difficulty**: Intermediate | **Time**: 90-120 minutes

- Vector store integration (Qdrant, Chroma)
- Embedding model comparison (OpenAI vs HuggingFace)
- Index persistence and loading
- Query engine configuration and modes
- Response synthesis strategies
- VectorIndexAutoRetriever for smart filtering

**Practical Example**: Scalable Document QA with Qdrant

### 📕 Notebook 4: `extras/04_advanced_retrieval.ipynb`
**Difficulty**: Intermediate-Advanced | **Time**: 120-150 minutes

- RecursiveRetriever for hierarchical retrieval
- QueryFusionRetriever (combining multiple retrievers)
- Custom retriever implementation
- Query transformation techniques (HyDE, rewriting)
- Complex metadata filtering (AND/OR/NOT logic)
- Retrieval metrics and evaluation (manual + built-in `RetrieverEvaluator`)

**Practical Example**: Multi-Strategy Research Paper Search

### 📔 Notebook 5: `extras/05_hybrid_search_and_reranking.ipynb`
**Difficulty**: Advanced | **Time**: 150-180 minutes

- Dense vs sparse vectors (BM25)
- Hybrid search implementation with Qdrant
- Alpha parameter tuning for score fusion
- Reranking models (Cohere `rerank-v3.5`, SentenceTransformer cross-encoders)
- Custom query engine implementation (production-ready, with `_get_prompt_modules`)
- Production patterns (error handling, monitoring, caching)
- Performance benchmarking and optimization

**Practical Example**: Production Document QA System

### 📓 Bonus Notebook: `Evaluating_the_Ideal_Chunk_Size_for_a_RAG_System_using_LlamaIndex.ipynb`

An end-to-end walkthrough of picking the optimal chunk size using response evaluation
(Faithfulness + Relevancy) across chunk sizes 128–2048. Modernized for llama-index 0.14
(`Settings`-based, no `ServiceContext`, current model IDs).

## Quick Start

### 1. Clone or Download

```bash
cd /Users/sourangshupal/Downloads/llama-index-tutorials
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment (Python 3.12+)
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or, using the pyproject (editable-free install of the dependency set):

```bash
pip install ".[notebooks]"
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required for OpenAI integrations
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For Anthropic LLM integration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: For Cohere reranking (used in Notebook 5)
COHERE_API_KEY=your_cohere_api_key_here

# Optional: For Qdrant Cloud (local Qdrant works without this)
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Optional: For AWS Bedrock tests (test_bedrock*.py)
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_DEFAULT_REGION=us-east-1
```

### 5. Prepare Sample Data

Add your PDFs and documents to the data directories:

```
data/
├── sample_docs/          # For Document QA examples
│   └── (add your PDFs here)
└── research_papers/       # For Research Paper examples
    └── (add academic papers here)
```

**Suggested Sample Data**:
- **sample_docs/**: Python documentation, API references, user manuals (2-20 pages)
- **research_papers/**: arXiv papers on ML/NLP topics (PDFs with clear structure)

### 6. Launch Jupyter

```bash
jupyter notebook
```

Start with `01_setup_and_basics.ipynb` in the repo root; advanced notebooks are in `extras/`.

## Project Structure

```
llama-index-tutorials/
├── 01_setup_and_basics.ipynb                     # Notebook 1
├── 02_documents_and_chunking.ipynb               # Notebook 2
├── 03_indexing_and_simple_queries.ipynb          # Notebook 3
├── extras/
│   ├── 04_advanced_retrieval.ipynb               # Notebook 4
│   └── 05_hybrid_search_and_reranking.ipynb      # Notebook 5
├── Evaluating_the_Ideal_Chunk_Size_for_a_RAG_System_using_LlamaIndex.ipynb
├── data/
│   ├── sample_docs/          # Your PDF documents
│   └── research_papers/       # Your academic papers
├── temp/                      # Curriculum/planning docs (internal)
├── storage/                   # Persisted index output (generated, git-ignored)
├── test_bedrock.py            # AWS Bedrock integration test (IAM auth)
├── test_bedrock_api_key.py    # AWS Bedrock test (API-key auth)
├── pyproject.toml             # Project metadata + dependencies
├── requirements.txt           # pip install -r requirements.txt
├── .env                       # API keys (create this yourself)
└── README.md
```

## Key Features

### ✅ Latest LlamaIndex Architecture
- Uses modular `llama-index-core` + integration packages (0.14.x)
- No deprecated features (QueryPipeline, ServiceContext avoided)
- Follows current best practices (Settings object, modern APIs)

### ✅ Production-Ready Patterns
- Error handling and logging
- Performance optimization techniques
- Cost optimization strategies
- Scalability considerations

### ✅ Practical Examples
- Real-world Document QA systems
- Research paper analysis and search
- Complete end-to-end workflows

### ✅ ML Engineering Focus
- Embedding dimension trade-offs
- Retrieval metric calculations
- A/B testing different configurations
- Performance benchmarking

## API Keys Required

| Service | Required For | Get API Key |
|---------|--------------|-------------|
| OpenAI | Embeddings (text-embedding-3-small) & LLM (gpt-6.1-terra) | [OpenAI Platform](https://platform.openai.com/) |
| Anthropic | Alternative LLM (Claude) - Optional | [Anthropic Console](https://console.anthropic.com/) |
| Cohere | Reranking in Notebook 5 - Optional | [Cohere Dashboard](https://dashboard.cohere.com/) |
| Qdrant Cloud | Cloud vector DB - Optional (local works) | [Qdrant Cloud](https://cloud.qdrant.io/) |
| AWS Bedrock | `test_bedrock*.py` scripts - Optional | [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/) |

**Note**: You can complete most tutorials with just the OpenAI API key. Other services are optional or have free local alternatives (e.g., local Qdrant, HuggingFace embeddings).

**Model choice (November 2026)**: Notebooks default to `gpt-6.1-terra` (fast workhorse, ~$1/$5 per 1M tokens). Swap in `gpt-6-luna` to cut costs further or `gpt-6.1-sol`/`gpt-6-astra` for higher quality. Avoid `gpt-4o-mini` (deprecated, shutdown 2027-01-14) and never use `gpt-3.5-turbo`/`gpt-4` (retired 2026-10-23).

## Troubleshooting

### Import Errors

```python
ModuleNotFoundError: No module named 'llama_index'
```

**Solution**: Ensure you're using the modular architecture:
```bash
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai
```

### API Key Errors

```python
AuthenticationError: Incorrect API key provided
```

**Solution**: Verify your `.env` file exists and contains valid keys:
```bash
cat .env  # Check if keys are present
```

### Vector Store Connection Issues

```python
ConnectionError: Could not connect to Qdrant
```

**Solution**: For local Qdrant, ensure it's running:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use in-memory storage for testing:
```python
# No external service needed
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
```

### Memory Issues with Large Documents

**Solution**: Adjust chunk size and batch processing:
```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)  # Smaller chunks
```

## Additional Resources

- **Official LlamaIndex Docs**: https://docs.llamaindex.ai/
- **LlamaIndex GitHub**: https://github.com/run-llama/llama_index
- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **OpenAI Embeddings Guide**: https://platform.openai.com/docs/guides/embeddings
- **OpenAI Model Deprecations**: https://developers.openai.com/api/docs/deprecations

## Version Information

- **Python**: 3.12+
- **LlamaIndex Core**: 0.14.x (latest: 0.14.25)
- **Default LLM**: `gpt-6.1-terra` (OpenAI)
- **Default embeddings**: `text-embedding-3-small` (OpenAI)
- **Architecture**: Modular (llama-index-core + integrations)
- **Last Updated**: 2026-11-15

## Contributing

Found an issue or have suggestions? This is a tutorial project for professional ML engineers learning LlamaIndex. Feel free to:
- Report bugs or errors in notebooks
- Suggest improvements to explanations
- Share additional practical examples
- Contribute performance optimization tips

## License

Educational content for professional training purposes.

---

**Ready to start?** Open `01_setup_and_basics.ipynb` and begin your LlamaIndex journey!
