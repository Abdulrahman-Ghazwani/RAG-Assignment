# Instructions

## Prerequisites

- Python 3.11+ **or** Docker
- OpenAI API key

## Environment

Create `.env` in the project root:

```env
OPENAI_API_KEY=sk-...
```

## Run locally

```bash
python -m pip install streamlit
pip install -r requirements.txt
python -m streamlit run app/ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

## Run with Docker

```bash
docker build -t rag-assignment .
docker run --rm -p 8501:8501 --env-file .env rag-assignment
```

Open [http://localhost:8501](http://localhost:8501)

## Usage

1. Upload up to 3 PDF or DOCX files.
2. Click **Process Documents**.
3. Ask in Arabic or English.
4. Grounded answers show **Sources**.

If the documents do not support an answer, the app returns a fixed no-information message and hides sources.
