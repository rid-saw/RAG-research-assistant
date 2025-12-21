# RAG Research Assistant

A production-ready RAG system for querying academic papers with **22% MRR improvement** through fine-tuned embeddings, hybrid retrieval, and neural reranking.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Live Demo

Try the deployed application: [HuggingFace Spaces Demo](https://huggingface.co/spaces/rid-saw/rag-research-assistant)

## Highlights

- **22% MRR improvement** over base model (0.75 → 0.91)
- **87% Hits@1** - correct answer ranked first 87% of the time
- **Hybrid retrieval** combining semantic search + BM25 keyword matching
- **CrossEncoder reranking** for precision on retrieved candidates
- **Rigorous evaluation** with 5-fold CV and 100 negative samples per query

## Architecture

```
Query
  |
  v
+-------------------------------+
|  Hybrid Search                |
|  (Semantic 60% + BM25 40%)    |
+-------------------------------+
  |
  v
+-------------------------------+
|  CrossEncoder Reranker        |
|  (ms-marco-MiniLM-L-6-v2)     |
+-------------------------------+
  |
  v
+-------------------------------+
|  LLM Generation               |
|  (Gemini 2.5 Flash)           |
+-------------------------------+
  |
  v
Answer + Source Citations
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/rid-saw/RAG-research-assistant.git
cd RAG-research-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "GOOGLE_API_KEY=your_gemini_api_key" > .env

# Run the application
python src/app.py
```

Open http://localhost:7860 in your browser.

## Results

Fine-tuned embeddings evaluated with rigorous methodology (5-fold CV, 100 negative samples):

| Metric | Base Model | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| **MRR** | 0.7493 | **0.9149** | **+22.1%** |
| **Hits@1** | 0.6545 | **0.8691** | **+32.8%** |
| Hits@5 | 0.8618 | 0.9709 | +12.7% |
| Hits@10 | 0.9018 | 0.9855 | +9.3% |
| NDCG@10 | 0.7836 | 0.9317 | +18.9% |

**What this means:** The system achieves performance comparable to production academic search engines like Google Scholar (MRR ~0.85) and Semantic Scholar (MRR ~0.80).

## Project Structure

```
RAG-research-assistant/
├── src/
│   ├── app.py                    # Gradio web interface
│   ├── retriever.py              # Hybrid retriever + reranking
│   ├── ingest.py                 # PDF loading and chunking
│   ├── fine_tune_embeddings.py   # Embedding fine-tuning
│   ├── generate_training_data.py # LLM-powered Q&A generation
│   └── download_arxiv.py         # arXiv PDF fetching
├── evaluation/
│   ├── evaluate.py               # Rigorous evaluation script
│   └── training_pairs.json       # 275 Q&A training pairs
├── models/
│   └── fine-tuned-embeddings/    # Fine-tuned model weights (88MB)
├── arxiv_papers.json             # Manifest of 25 arXiv papers
├── data/                         # Research papers (PDFs, runtime-fetched)
└── chroma_db/                    # Vector database (built at runtime)
```

## Key Features

### 1. Hybrid Retrieval
Combines semantic search (fine-tuned embeddings) with BM25 keyword matching for robust retrieval that handles both meaning and exact terms.

### 2. CrossEncoder Reranking
After initial retrieval, a CrossEncoder model re-scores candidates by seeing query and passage together - significantly more accurate than bi-encoder similarity alone.

### 3. Fine-tuned Embeddings
Domain-adapted `all-MiniLM-L6-v2` trained on 275 LLM-generated Q&A pairs using `MultipleNegativesRankingLoss`.

### 4. Evaluation
- 5-fold cross-validation for stable metrics
- 100 negative distractors per query (realistic retrieval simulation)
- Multiple metrics: MRR, Hits@k, NDCG@10


## Tech Stack

- **Embeddings:** Sentence Transformers (fine-tuned all-MiniLM-L6-v2)
- **Reranking:** CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Vector Store:** ChromaDB
- **Keyword Search:** BM25 (rank-bm25)
- **LLM:** Google Gemini 2.5 Flash
- **Framework:** LangChain
- **Interface:** Gradio

## Usage

### Running the Web Interface
```bash
python src/app.py
# Opens at http://localhost:7860
```

### Fine-tuning Embeddings
```bash
# Generate training data (requires Gemini API)
python src/generate_training_data.py

# Fine-tune the model
python src/fine_tune_embeddings.py
```

### Running Evaluation
```bash
python evaluation/evaluate.py
# Compares base vs fine-tuned model
# Results saved to evaluation/evaluation_results.json
```

### Adding New Papers
```bash
# 1. Add PDFs to data/
cp new_paper.pdf data/

# 2. Clear vector store and restart
rm -rf chroma_db/
python src/app.py
```

## Adapting for Your Own Use

This repository is designed as an educational reference. You can adapt it for your own documents:

### Step 1: Prepare Your Documents
Replace the contents of `data/` with your own PDFs (research papers, documentation, books, etc.).

### Step 2: Generate Training Data
```bash
# Modify src/generate_training_data.py to suit your domain
python src/generate_training_data.py
```

This creates Q&A pairs from your documents for fine-tuning.

### Step 3: Fine-tune Embeddings
```bash
python src/fine_tune_embeddings.py
```

This adapts the embedding model to your domain vocabulary and concepts.

### Step 4: Evaluate
```bash
python evaluation/evaluate.py
```

Verify improvement over the base model.

### Step 5: Run Your RAG System
```bash
python src/app.py
```

**Note:** The included fine-tuned model (`models/fine-tuned-embeddings/`) is optimized for ML/AI research papers. For best results on different domains, you should re-run the fine-tuning pipeline on your own documents.

## Copyright & Fair Use

To respect intellectual property and comply with arXiv's terms of use, this project uses a **runtime-fetch and cleanup** architecture.

**Included in this repo:**
- ✅ `models/fine-tuned-embeddings/` - Fine-tuned embedding model (88MB)
- ✅ `arxiv_papers.json` - Manifest of 25 arXiv IDs for the knowledge base
- ✅ Source code and evaluation scripts

**NOT included (rebuilt at runtime):**
- ❌ Original PDF files (~250MB) - Downloaded from arXiv during setup
- ❌ `chroma_db/` - Vector database rebuilt from PDFs
- ❌ `data/` folder - Populated at runtime

### How it works:
1. **Download:** On startup, the system downloads PDFs from arXiv using IDs in `arxiv_papers.json`
2. **Process:** PDFs are parsed, chunked (1000 chars, 200 overlap), and embedded using the fine-tuned model
3. **Index:** ChromaDB vector store is built with embeddings + original text chunks
4. **Cleanup:** PDFs are automatically deleted after processing (copyright compliance)
5. **Serve:** Only embeddings and text chunks remain for retrieval

**Key Point:** Users cannot download original PDFs through this interface. Only small text excerpts are shown as source citations. For full papers, visit [arXiv.org](https://arxiv.org) directly.

This complies with arXiv's Terms of Use and operates under fair use for educational/research purposes.

### Manual Setup
If you prefer to pre-download the source material:
```bash
python src/download_arxiv.py          # Download all papers
python src/download_arxiv.py --clean  # Remove PDFs after embedding
```

## License

MIT License - feel free to use this project for learning and building.

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [LangChain](https://langchain.com/) for RAG orchestration
- [Gradio](https://gradio.app/) for the web interface
- Research papers from [arXiv.org](https://arxiv.org)
