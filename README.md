# RAG Document Q&A — Streamlit & Desktop App

Two RAG (Retrieval-Augmented Generation) applications for querying PDF documents: a simple Streamlit web app and a full-featured PyQt5 desktop application with multi-provider support and database logging.

---

## What's Inside

### `app.py` — Streamlit RAG Web App
Upload PDFs to a local folder, embed them into a FAISS vector store with one click, then ask questions about the documents. Uses Groq's LLaMA model for fast inference and OpenAI embeddings for semantic search. Retrieved source chunks are shown in an expandable section below each answer.

### `advancedapp.py` — PyQt5 Desktop RAG Application
A native desktop app (v2) with a full GUI built in PyQt5. Features include:
- **Multi-provider LLM switching** — OpenAI (GPT-4o), Groq (LLaMA 3), or local Ollama
- **Persistent index management** — create and save FAISS indexes, load them back later without re-embedding
- **Dual-panel layout** — chat history on the left, retrieved context chunks on the right
- **Database logging** — logs every Q&A interaction (question, answer, provider, response time) to SQLite or SQL Server
- **Background threading** — indexing and querying run in worker threads so the UI stays responsive

---

## Project Structure

```
├── app.py                # Streamlit RAG web app
├── advancedapp.py        # PyQt5 desktop RAG application
├── research_papers/      # Drop your PDF files here
├── index.faiss           # Pre-built FAISS vector index
├── index.pkl             # FAISS index metadata
└── requirements.txt      # Python dependencies
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/neobaul/project-5-ragdocumentqa.git
cd project-5-ragdocumentqa
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install streamlit langchain-groq langchain-openai langchain-ollama faiss-cpu pypdf pyqt5 pyodbc
```

### 4. Set up API keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### 5. Add your PDFs
Place your PDF files in the `research_papers/` folder.

---

## Running the Apps

### Streamlit web app
```bash
streamlit run app.py
```
1. Click **Document Embedding** to index your PDFs
2. Type a question and get an answer with source references

### PyQt5 desktop app
```bash
python advancedapp.py
```
1. Select your LLM provider and database type from the left panel
2. Choose your PDF folder and click **Create & Save New Index** (or load an existing one)
3. Ask questions in the chat panel — retrieved context appears below the conversation

---

## How It Works

```
PDF files in folder
    ↓
PyPDFDirectoryLoader → RecursiveCharacterTextSplitter (1000 tokens, 200 overlap)
    ↓
OpenAI Embeddings → FAISS Vector Store (saved to disk)
    ↓
User question → Retriever → Top-K relevant chunks
    ↓
Stuffed into prompt → LLM (Groq / OpenAI / Ollama)
    ↓
Answer + source chunks displayed
    ↓
Interaction logged to SQLite / SQL Server
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web App | Streamlit |
| Desktop App | PyQt5 |
| LLM (web) | Groq LLaMA 3.1 8B Instant |
| LLM (desktop) | OpenAI GPT-4o / Groq LLaMA 3 / Ollama |
| Embeddings | OpenAI `text-embedding-ada-002` / Ollama `nomic-embed-text` |
| Vector Store | FAISS (persisted to disk) |
| PDF Loading | LangChain `PyPDFDirectoryLoader` |
| Database Logging | SQLite / SQL Server (pyodbc) |
| Background Tasks | PyQt5 `QThread` worker pattern |
| Language | Python 3.11 |
