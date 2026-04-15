# Instructions

## Prerequisites

- Python 3.11+ or Docker  
- OpenAI API key  
- **ChromaDB** (Docker): `docker compose up -d chromadb`

## Environment

Create `.env` in the project root (see `.env.example`):

```env
OPENAI_API_KEY=sk-...
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

## Run locally

```bash
docker compose up -d chromadb
pip install -r requirements.txt
python -m streamlit run app/ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Run with Docker (app + Chroma)

```bash
docker compose up --build
```

Put `OPENAI_API_KEY` in `.env` in the project root so Compose can pass it to the `app` service.

## Usage

1. Upload up to 3 PDF or DOCX files.  
2. Click **Process Documents**.  
3. Ask in Arabic or English.  
4. Grounded answers show **Sources**; otherwise a fixed no-information message.
