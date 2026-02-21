# Private Credit & Financial Stability RAG System

> A production-grade, fully local Retrieval-Augmented Generation system for analyzing academic research papers on private credit markets and BDC operations — served via a FastAPI REST interface.

## Project Overview

This project implements an advanced RAG pipeline that combines semantic search with large language model inference to create an intelligent document analysis system. Built entirely for local execution on consumer hardware (8GB VRAM), it eliminates dependency on cloud APIs while maintaining high-quality responses. The system is served as a REST API using FastAPI, enabling programmatic access to the RAG pipeline from any HTTP client.

**Target Document**: "Private Credit and Financial Stability" by Sergey Chernenko & David Scharfstein (56-page academic research paper)

### Key Capabilities

- Real-time semantic search across 143 document chunks with sub-100ms latency
- REST API interface via FastAPI with interactive Swagger UI documentation
- Context-aware question answering with strict document boundary enforcement
- Financial domain specialization through rule-based structured prompting
- Verbatim citation of financial metrics and definitions from source text
- Explicit signalling when retrieved context is insufficient, rather than hallucinating
- Zero-shot inference on complex financial terminology and concepts
- Complete data privacy - no external API calls

---

## RAG Architecture

### Pipeline Components

```
PDF Document → Document Loader → Text Splitter → Embedding Model → Vector Store
                                                                          ↓
HTTP Request → FastAPI (/ask) → Embedding Model → Similarity Search → Context Retrieval → LLM → JSON Response
```

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Parsing** | PyMuPDF | Efficient PDF text extraction with metadata |
| **Text Chunking** | RecursiveCharacterTextSplitter | Semantic-aware document segmentation |
| **Embeddings** | Sentence-BERT (all-MiniLM-L6-v2) | 384-dimensional dense vectors |
| **Vector Database** | FAISS | Approximate nearest neighbor search |
| **LLM** | Qwen 2.5 1.5B Instruct | Instruction-tuned causal language model |
| **Framework** | LangChain | RAG orchestration and prompt management |
| **API Server** | FastAPI | Async REST API with Pydantic validation |

---

## System Parameters

### Document Processing Configuration

```python
# Text Splitting Strategy
chunk_size = 1200          # Characters per chunk
chunk_overlap = 200        # Overlap for context continuity
separators = ['\n\n', '\n', '.', ' ']  # Hierarchical splitting
```

**Rationale**: 1200-character chunks with 200-char overlap balance context preservation and retrieval precision. Hierarchical separators ensure semantic boundaries are respected.

**Results**: 56-page PDF → 143 chunks (avg ~800 chars/chunk)

### Embedding Model Specifications

```python
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_dimension = 384
model_size = 90MB
inference_speed = ~1000 sentences/sec (GPU)
```

**Performance Characteristics**:
- Lightweight architecture suitable for real-time inference
- Strong semantic understanding for technical/financial text
- Normalized embeddings for cosine similarity search

### Vector Store Configuration

```python
index_type = 'FAISS (Flat L2)'
retrieval_count = 5        # Top-k documents per query
distance_metric = 'L2'     # Euclidean distance
index_size = ~15MB         # For 143 chunks
```

**Search Strategy**: Exhaustive search ensures highest recall for the relatively small document corpus. Average retrieval time: <100ms.

### LLM Configuration

```python
model = 'Qwen/Qwen2.5-1.5B-Instruct'
dtype = torch.bfloat16              # 16-bit brain float
device_map = 'auto'                 # Automatic GPU allocation
low_cpu_mem_usage = True            # Optimized loading

# Generation Parameters
temperature = 0.2                    # Low temp for factual responses
max_new_tokens = 512                # Response length limit
repetition_penalty = 1.2            # Reduces redundant phrasing
no_repeat_ngram_size = 3            # Prevents 3-gram repetition
do_sample = True                    # Enables temperature sampling
```

**Model Characteristics**:
- **Parameters**: 1.5 billion (efficient for consumer GPUs)
- **Context Window**: 32,768 tokens
- **Architecture**: Transformer decoder with GQA (Grouped Query Attention)
- **VRAM Usage**: ~3.5GB (bfloat16 precision)
- **Inference Speed**: ~30 tokens/sec on 8GB VRAM GPU

**Generation Strategy**: Low temperature (0.2) prioritizes factual accuracy over creativity. Repetition penalties ensure diverse, natural-sounding responses without redundancy.

---

## RAG Implementation Details

### Retrieval Process

```python
def format_prompt(question):
    # 1. Query Encoding
    query_embedding = embeddings.embed_query(question)
    
    # 2. Similarity Search (Top-5)
    search_results = vector_db.similarity_search(question, k=5)
    
    # 3. Context Aggregation
    context = '\n---\n'.join([doc.page_content for doc in search_results])
    
    # 4. Prompt Construction with Structured Rules
    prompt = f'''You are a Financial Analyst. Answer the question only from the context provided.
    RULES:
    1. You must be factually Accurate
    2. If the information is not sufficient then mention 'Not enough information is provided in the document'
    3. You must not answer beyond the context provided
    4. Cite the financial metrics and definitions exactly as present in the context

context:
{context}

question: {question}
Answer: '''
    
    # 5. LLM Inference
    response = pipe(prompt, return_full_text=False)
    return response[0]['generated_text']
```

### Prompt Engineering Strategy

**System Role**: Financial Analyst persona establishes domain expertise and grounds the response style in professional financial analysis.

**Structured Rule Enforcement**: The updated prompt moves beyond a single vague directive to four explicit, numbered constraints that directly shape model behavior:

- **Rule 1 – Factual Accuracy**: Instructs the model to treat factual correctness as non-negotiable, reducing hallucination risk on domain-specific financial data.
- **Rule 2 – Graceful Degradation**: When retrieved context is insufficient, the model is required to explicitly state *"Not enough information is provided in the document"* rather than fabricating an answer from parametric memory. This makes knowledge gaps transparent to the user.
- **Rule 3 – Context Boundary Enforcement**: Strictly prohibits the model from reasoning beyond the retrieved chunks, preventing the blending of internal pretraining knowledge with document-specific facts — a common failure mode in financial RAG systems.
- **Rule 4 – Verbatim Metric Citation**: Requires financial metrics, ratios, and definitions to be cited exactly as they appear in the source text, ensuring numerical precision and eliminating paraphrasing errors on critical data.

**Context Injection**: Retrieved chunks are inserted directly into the prompt with clear `---` delimiters to help the model distinguish between source segments.

**Output Control**: `return_full_text=False` strips the prompt from the output, returning only the generated answer.

#### Impact on Response Quality

The structured rule-based prompt produces measurably different responses compared to the original single-line instruction:

| Dimension | Original Prompt | Updated Prompt |
|-----------|----------------|----------------|
| **Out-of-scope queries** | May draw on parametric knowledge, generating plausible but ungrounded answers | Explicitly refuses and signals insufficient context |
| **Metric citation** | Paraphrases or approximates figures | Reproduces financial metrics verbatim from the document |
| **Hallucination risk** | Higher — model defaults to general financial knowledge when context is thin | Lower — rules create a hard boundary around retrieved content |
| **Transparency** | Silent on retrieval gaps | Actively communicates when the document lacks relevant information |

### Chunk Retrieval Strategy

**Top-K Selection (k=5)**: Empirically optimized to balance:
- **Context Coverage**: Sufficient information for complex queries
- **Noise Reduction**: Avoids diluting relevant context with marginally related chunks
- **Token Budget**: Keeps total prompt under context window limits

**Expected Performance**:
- Simple factual queries: Typically answered from top 1-2 chunks
- Complex analytical queries: Leverage all 5 chunks for comprehensive responses

---

## FastAPI Service

The RAG pipeline is exposed as a production-ready REST API via `main.py`, enabling any HTTP client (curl, Postman, browser, frontend app) to query the system programmatically.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/root` | Welcome message and API status |
| `GET` | `/health` | Model readiness check — returns loading state |
| `POST` | `/ask` | Submit a question; returns structured JSON response |

### Startup & Shutdown — Lifespan Management

Models are loaded once at server startup using FastAPI's `@asynccontextmanager` lifespan pattern, avoiding per-request reloading overhead. On shutdown, all globals are cleared to release GPU memory cleanly.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load embeddings → FAISS index → LLM pipeline
    embeddings = HuggingFaceEmbeddings(...)
    vector_db = FAISS.load_local('RAG INDEX', ...)
    pipe = pipeline(...)
    yield
    # Shutdown: release all resources
    vector_db = pipe = embeddings = None
```

### Request & Response Schema

**Request** (`POST /ask`):
```json
{
  "question": "What is a BDC and what are its leverage requirements?"
}
```

**Response**:
```json
{
  "question": "What is a BDC and what are its leverage requirements?",
  "response": "A Business Development Company (BDC) is..."
}
```

Both schemas are enforced via Pydantic `BaseModel`, providing automatic request validation and clear error messages for malformed inputs.

### Health Check

Before sending queries, clients can poll `/health` to confirm models are ready:

```json
{ "status": "Ready", "Models_Loaded": true }
```

During startup (model loading takes 25–30s), the endpoint returns:

```json
{ "status": "loading", "Models_Loaded": false }
```

### Running the API

```bash
# Install FastAPI and server
pip install fastapi uvicorn

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger UI is available at `http://localhost:8000/docs` once the server is running.

### Example Usage

**curl**:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is DRIP and what role does it play in BDC deleveraging?"}'
```

**Python (`requests`)**:
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What are the leverage reduction strategies for BDCs?"}
)
print(response.json()["response"])
```

---

## Hardware Requirements & Performance

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB VRAM (tested: RTX 3070, RTX 4060 Ti)
- **RAM**: 16GB system memory
- **Storage**: 5GB free space (models + index)
- **CUDA**: Version 11.8 or higher

### Performance Benchmarks (8GB VRAM GPU)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Model Loading | 25-30s | One-time per session |
| Embedding Query | 50-80ms | Sentence-BERT inference |
| Vector Search | 50-100ms | FAISS L2 search (143 vectors) |
| LLM Generation | 1.5-3s | 512 tokens @ ~30 tok/sec |
| **End-to-End Query** | **2-5s** | Total user-facing latency |

### Memory Footprint

- **Qwen 2.5 1.5B (bfloat16)**: ~3.5GB VRAM
- **Embedding Model**: ~400MB VRAM
- **FAISS Index**: ~15MB RAM
- **PyTorch Overhead**: ~2GB VRAM
- **Total Peak Usage**: ~6.5GB VRAM

---

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv2
source venv2/bin/activate  # Windows: venv2\Scripts\activate

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install langchain-community langchain-huggingface transformers \
            langchain-text-splitters faiss-cpu pymupdf sentence-transformers \
            fastapi uvicorn pydantic
```

### Build Vector Index

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load and chunk document
loader = PyMuPDFLoader('data/Private_Cred_Fin_Stab_Paper.pdf')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=['\n\n', '\n', '.', ' ']
)
chunks = text_splitter.split_documents(data)

# Create FAISS index
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local('RAG_INDEX')
```

### Initialize LLM Pipeline

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    temperature=0.2,
    do_sample=True,
    max_new_tokens=512,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)
```

### Start the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server loads all models at startup (~25–30s). Once `/health` returns `"Models_Loaded": true`, the API is ready to accept requests.

### Query via API

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is a Business Development Company and how does it differ from traditional private equity?"}
)
print(response.json()["response"])
```

Or visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Example Queries & Performance

### Query Types Supported

| Query Type | Example | Avg. Response Time |
|------------|---------|-------------------|
| **Definition** | "What is a BDC?" | 2.1s |
| **Comparison** | "How do BDCs differ from RICs?" | 2.8s |
| **Data Analysis** | "What does Table A2 show about default rates?" | 3.2s |
| **Mechanism** | "Explain the DRIP mechanism" | 2.9s |
| **Summarization** | "Summarize the key findings" | 4.5s |

### Sample Output Quality

**Query**: "What are BDC assets?"

**Response (Updated Prompt)**:
```
BDC assets consist of two primary components: (1) principal amounts invested in various 
forms like loans, cash reserves, etc., along with any other relevant securities held 
within them. Loans represent around 87% of most BDC portfolios' composition; Equity 
represents approximately 5.9%; Other components may vary based on specific details per 
individual BDC.
```

**Why this matters**: The updated prompt enforces verbatim metric citation (Rule 4), so figures like "87%" and "5.9%" are drawn directly from the document rather than approximated. Under the original prompt, the response paraphrased loosely — referencing "senior secured debt" and "mezzanine positions" which were not explicitly present in the retrieved context for this query.

---

## Advanced Configuration

### Hyperparameter Tuning Guidelines

**Chunk Size Optimization**:
- **Smaller chunks (600-800)**: Higher precision, may lose context
- **Larger chunks (1500-2000)**: More context, potential noise
- **Current setting (1200)**: Optimal balance for academic papers

**Retrieval Count (k) Selection**:
- **k=3**: Fast, suitable for simple queries
- **k=5**: Current setting, handles complex questions
- **k=7-10**: Useful for multi-faceted questions, watch token limits

**Temperature Tuning**:
- **0.1-0.3**: Factual, deterministic (recommended for finance)
- **0.5-0.7**: Balanced creativity and accuracy
- **0.8-1.0**: Creative, diverse (not recommended for this use case)

### Model Alternatives (Local Deployment)

| Model | Parameters | VRAM | Performance |
|-------|-----------|------|-------------|
| **Qwen 2.5 1.5B** | 1.5B | 3.5GB | Current choice |
| Qwen 2.5 3B | 3B | 6.5GB | +20% quality, slower |
| Llama 3.2 3B | 3B | 6GB | Comparable quality |
| Phi-3 Mini | 3.8B | 8GB | Max for 8GB VRAM |

---

## Project Structure

```
private-credit-rag/
├── data/
│   └── Private_Cred_Fin_Stab_Paper.pdf    # Source document (56 pages)
├── RAG INDEX/                              # FAISS vector store (git-ignored)
│   ├── index.faiss                         # Vector index file
│   └── index.pkl                           # Metadata & docstore
├── __pycache__/                            # Python cache (git-ignored)
├── main.py                                 # FastAPI application entry point
├── Private_cred_finstab_rag.ipynb         # Index-building & exploration notebook
├── .gitignore                              # Excludes RAG INDEX/ and __pycache__/
└── README.md
```

> **Note**: The FAISS vector index (`RAG INDEX/`) is excluded from version control via `.gitignore` as it is a derived artifact that can be regenerated from the source PDF using the notebook. Run the notebook cells top-to-bottom to rebuild the index before starting the API server.

---

## Domain Coverage

### Financial Concepts Indexed

- **Business Development Companies (BDCs)**: Structure, regulation, operations
- **Registered Investment Companies (RICs)**: Tax treatment, distribution requirements
- **Private Credit Markets**: Lending dynamics, market structure
- **Financial Leverage**: Asset-to-capital ratios, deleveraging patterns
- **Default Risk**: Industry-specific betas, aggregate default correlations
- **Portfolio Management**: Holdings analysis, valuation methods
- **Dividend Policy**: DRIP mechanics, payout ratios
- **Regulatory Framework**: RIC status requirements, SEC compliance

### Statistical Data Available

- Summary statistics (2024 Q4)
- Industry default beta coefficients
- Portfolio holdings by sector
- Leverage ratio distributions
- Performance metrics across deciles

---

## Limitations & Future Work

### Current Limitations

- **Single Document**: System designed for one paper; multi-document retrieval not implemented
- **No Citation Tracking**: Responses don't include page numbers (roadmap item)
- **English Only**: No multilingual support
- **Static Index**: Requires rebuild for document updates

### Roadmap

- [ ] **Multi-Document RAG**: Cross-paper analysis and comparison
- [ ] **Quantization**: 4-bit quantized models for 4GB VRAM GPUs
- [ ] **Hybrid Search**: Combine dense (FAISS) + sparse (BM25) retrieval
- [ ] **Re-ranking**: Cross-encoder model for retrieval refinement
- [ ] **Streaming Responses**: Token-by-token SSE output for lower perceived latency
- [x] **REST API**: FastAPI service with `/ask`, `/health`, and `/root` endpoints ✅
- [ ] **Web Interface**: Gradio/Streamlit frontend connecting to the FastAPI backend
- [ ] **Evaluation Suite**: RAGAS metrics for retrieval & generation quality
- [ ] **Authentication**: API key middleware for multi-user deployments

---

## Technical References

### Key Papers & Resources

- **RAG**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **FAISS**: [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- **Sentence-BERT**: [Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- **Qwen 2.5**: [Technical Report](https://arxiv.org/abs/2407.10671)

### Source Document

**Title**: Private Credit and Financial Stability  
**Authors**: Sergey Chernenko, David Scharfstein  
**Institution**: Harvard Business School, MIT Sloan  
**Year**: 2025  
**Pages**: 56  

---


## Acknowledgments

- **Alibaba Cloud** for open-sourcing Qwen models
- **Hugging Face** for model hosting and transformers library
- **Facebook AI Research** for FAISS
- **LangChain** contributors for the RAG framework
- **Sebastián Ramírez** for FastAPI
- **Paper Authors** for the foundational research

---

**Built for researchers, by researchers. Fully local, fully private, fully capable.**