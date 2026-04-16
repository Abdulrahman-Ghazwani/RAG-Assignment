# RAG Assignment

**Standalone run guide for submission:** see **[INSTRUCTIONS.md](INSTRUCTIONS.md)** (English, step-by-step).

Upload up to **3** PDF or DOCX files, index them, then ask questions in **Arabic or English**. Answers **stream** from the model; when the reply is grounded in your files, **Sources** are listed.

| Layer | What |
|-------|------|
| **UI** | Angular — `frontend/` |
| **API** | FastAPI — `app/api/main.py` |
| **Vectors** | ChromaDB — Docker service |

---

## Run with Docker (recommended)

You only need **Docker** (e.g. Docker Desktop) and an **OpenAI API key**. You do **not** need to run `pip install -r requirements.txt` on your machine: Python dependencies are installed **inside** the `app` image when Docker builds it.

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd RAG-Assignment
```

Copy **`.env.example`** to **`.env`** in the project root (e.g. `cp .env.example .env`, or copy the file in Explorer). Then set:

```env
OPENAI_API_KEY=sk-...
```

(`CHROMA_HOST` / `CHROMA_PORT` are set inside `docker-compose.yml` for the containers; you normally only need the API key here.)

### 2. Start everything

```bash
docker compose up --build
```

First run builds the **API** and **frontend** images (this can take a few minutes).

### 3. Open the app

| What | URL |
|------|-----|
| **Web UI** (Angular + nginx) | [http://localhost:4200](http://localhost:4200) |
| **API health** | [http://localhost:8080/api/health](http://localhost:8080/api/health) |
| **Chroma** (for the app, not a browser UI) | `http://localhost:8000` |

### 4. Use the UI

1. Upload up to **3** PDF or DOCX files (drag-and-drop or browse).  
2. Click **Process documents** and wait until indexing finishes.  
3. Ask questions; grounded answers show **Sources** when applicable.

### Useful Docker commands

| Command | Purpose |
|---------|---------|
| `docker compose up --build` | Build images and run (foreground logs) |
| `docker compose up --build -d` | Same, detached (background) |
| `docker compose down` | Stop and remove containers |
| `docker compose logs -f app` | Follow API logs |

---

## How the pieces connect

- The **browser** loads the Angular app from port **4200**.  
- Nginx in the **frontend** container serves the static UI and **proxies** `/api/` to the **app** service (FastAPI on **8080**), so the UI and API share one origin and you avoid CORS issues.  
- The UI sends **`X-Session-Id`** (a UUID stored in `localStorage`) so each browser session gets its own Chroma collection on the server.  
- **ChromaDB** runs as another container; data persists in the **`chroma_data`** volume.

---

## Optional: run without Docker (developers)

Use this if you want hot-reload for Python or Angular without rebuilding images. Then you **do** install dependencies on your machine.

**Requirements:** Python 3.11+, Node 20+, Chroma running (e.g. `docker compose up -d chromadb` only).

```bash
# Python API (from repo root)
pip install -r requirements.txt
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8080
```

In another terminal:

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:4200](http://localhost:4200). The dev server proxies `/api` to `http://localhost:8080` via `frontend/proxy.conf.json`. Put `CHROMA_HOST=localhost` in `.env` when the API runs on the host and Chroma is exposed on port 8000.

`requirements.txt` is used for **local** Python runs and by the **Dockerfile** during `docker build`; it is **not** required on the host if you only use `docker compose`.

---

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| API errors / no answers | `OPENAI_API_KEY` in `.env`, rebuild: `docker compose up --build` |
| `413 Request Entity Too Large` | Large PDFs: nginx allows 50MB in `frontend/nginx.conf`; rebuild **frontend** after changes |
| Port already in use | Stop other services on **4200**, **8080**, or **8000**, or change ports in `docker-compose.yml` |
| Slow indexing | Defaults use larger embedding batches + parallel requests (same model). If OpenAI rate-limits, set `EMBEDDING_MAX_PARALLEL=1` or lower `EMBEDDING_BATCH_SIZE` in `.env` and rebuild the `app` image / restart Compose. |

---

## Repo layout

| Path | Role |
|------|------|
| `app/` | FastAPI app and RAG pipeline services |
| `frontend/` | Angular source and Docker nginx config |
| `docker-compose.yml` | Chroma, API, frontend services |
| `Dockerfile` | Python API image |
| `requirements.txt` | Python deps (Docker build + optional local run) |
| `.env.example` | Copy to `.env` and set `OPENAI_API_KEY` |
