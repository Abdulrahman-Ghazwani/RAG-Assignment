# Instructions

## 1) Prerequisites

- Python 3.11+ (for local run), or Docker (for container run)
- OpenAI API key

## 2) Environment setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 3) Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python -m streamlit run app/ui/streamlit_app.py
```

Open:

- [http://localhost:8501](http://localhost:8501)

## 4) Run with Docker

Build image:

```bash
docker build -t rag-assignment .
```

Run container:

```bash
docker run --rm -p 8501:8501 --env-file .env rag-assignment
```

Open:

- [http://localhost:8501](http://localhost:8501)

## 5) Usage

1. Upload up to 3 files (`.pdf` or `.docx`).
2. Click **Process Documents**.
3. Ask questions in Arabic or English.
4. Review answer and sources.

## 6) Notes

- If a question is not grounded in uploaded docs, the app returns a no-information message.
- Sources are shown only for grounded answers.
