# PDF Reader

A local, AI-powered PDF assistant that allows users to ask questions about a document and receive answers grounded in its content.

The project implements a **Retrieval-Augmented Generation (RAG)** pipeline using semantic embeddings for document retrieval and a locally hosted LLM for response generation.

## Architecture

```text
                  ┌─────────────────┐
                  │     PDF File    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  pdfplumber     │
                  │ Text Extraction │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Sentence-Based  │
                  │    Chunking     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Sentence        │
                  │ Transformer     │
                  │   Embeddings    │
                  └────────┬────────┘
                           │
                           ▼
User Query ──────► Semantic Retrieval
      │                  │
      │                  ▼
      │           Relevant Chunks
      │                  │
      │                  ▼
      └──────────► Ollama / Llama 3.2
                         │
                         ▼
                       Answer
```

## How It Works

### 1. PDF Processing

The application extracts text from each page of the PDF using `pdfplumber`.

The extracted text is then split into sentence-based chunks with overlapping context to improve retrieval across sentence boundaries.

### 2. Embedding Generation

Each document chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The user's query is embedded using the same model.

Cosine similarity is then used to determine which document chunks are most relevant to the query.

### 3. Query Classification

The application contains lightweight semantic detection for **Meta questions** and **Follow-up questions**

For these questions, the system prioritizes chunks from the beginning and end of the document.

When a follow-up is detected, the system can reuse the previously retrieved chunks rather than performing an entirely new retrieval.

## Tech Stack

* **Python**
* **Ollama** - Local LLM inference
* **Llama 3.2** - Language model
* **Sentence Transformers** - Semantic embeddings
* **NumPy** - Vector operations and similarity ranking
* **pdfplumber** - PDF text extraction
* **TextBlob** - Query spelling correction

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kmbruss/pdf-reader
cd PDF-note-reader
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama for your operating system, then download the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before starting the application.

### 5. Add your PDF

Place your PDF in the project directory and name it:

```text
essay.pdf
```

Alternatively, modify:

```python
PDF_PATH = "essay.pdf"
```

to point to another PDF.

## Usage

Run the application with:

```bash
python main.py
```

## Future Improvements

Potential improvements include:

* Support for multiple PDFs
* vector databases instead of recomputing embeddings
* Reranking retrieved passages
* Source citations with page numbers
* Support for tables and images within PDFs
* Evaluation framework for retrieval and answer quality