# Daedalus- Agentic Workflow for Building Pydantic AI Agents

An intelligent documentation crawler and RAG (Retrieval-Augmented Generation) system built using Pydantic AI, LangGraph, and Supabase that is capable of building other Pydantic AI agents. The system crawls the Pydantic AI documentation, stores content in a vector database, and provides Pydantic AI agent code by retrieving and analyzing relevant documentation chunks.

This project works locally using Ollama, reasoner model includes gemma:2b(since it's lightweight) and for the primary(coding) model we use llama3.1:8b.
#You can use any primary model as long as it is tool friendly

## Features

- Multi-agent workflow using LangGraph
- Specialized agents for reasoning, routing, and coding
- Pydantic AI documentation crawling and chunking
- Vector database storage with Supabase
- RAG-based question answering
- Support for code block preservation
- Streamlit UI for interactive querying

## Techstack

- Python 3.11+
- PydanticAI
- Supabase database
- OpenAI/OpenRouter API key or Ollama for local LLMs
- Streamlit (for web interface)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/coleam00/archon.git
cd archon/iterations/v2-agentic-workflow
```

2. Install dependencies (recommended to use a Python virtual environment):
```bash
python -m venv venv
source ./venv/Scripts/activate  # bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Edit `.env` with your API keys and preferences:
   ```env
   BASE_URL=https://api.openai.com/v1 for OpenAI, https://api.openrouter.ai/v1 for OpenRouter, or your Ollama URL
   LLM_API_KEY=your_openai_or_openrouter_api_key
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   PRIMARY_MODEL=gemma:2b  # or something else
   REASONER_MODEL=llama3.1:8b     # or something else
   EMBEDDING MODEL:nomic-embed-text:v1.5
   ```

## Usage

### Database Setup

Execute the SQL commands in `ollama_site_pages.sql` by going to Supabse -> "SQL Editor" tab and pasting in the SQL into the editor there. Then click "Run".

### Crawl Documentation

To crawl and store documentation in the vector database:

```bash
python crawl_pydantic_ai_docs.py
```

This will:
1. Fetch URLs from the documentation sitemap
2. Crawl each page and split into chunks
3. Generate embeddings and store in Supabase

### Chunk Config
 `crawl_pydantic_ai_docs.py`:
```python
chunk_size = 5000  # Characters per chunk
```

The chunker intelligently preserves:
- Code blocks
- Paragraph boundaries
- Sentence boundaries

### Streamlit Web Interface
```bash
streamlit run streamlit_ui.py
```

The interface will be available at `http://localhost:8501`

## Configuration

### Database Schema

The Supabase database uses the following schema:
```sql
CREATE TABLE site_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT,
    chunk_number INTEGER,
    title TEXT,
    summary TEXT,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);
```

## Project Structure

- `daedalus_graph.py`: LangGraph agentic workflow
- `pydantic_ai_coder.py`: RAG implementation
- `crawl_pydantic_ai_docs.py`: Documentation crawler and processor
- `streamlit_ui.py`: Web interface with streaming support
- `ollama_site_pages.sql`: Database setup commands
- `requirements.txt`: Project dependencies

  [MIT License](LICENSE)
