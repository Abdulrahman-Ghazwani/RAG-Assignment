# RAG Assignment

Simple **RAG** demo: upload up to **3** **PDF/DOCX** files, index them, then ask questions in **Arabic or English** with **streaming** answers and **sources** when the answer is grounded in the text.

| File | Purpose |
|------|---------|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | Local or Docker run steps |
| [REPORT.md](REPORT.md) | Short solution overview |

```bash
python -m streamlit run app/ui/streamlit_app.py
```

Requires `OPENAI_API_KEY` in `.env`.
