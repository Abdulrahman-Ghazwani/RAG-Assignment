# How to Run the RAG Project

This document is a **standalone guide** for reviewers and new users. It explains what you need, how to start the system with Docker, and how to verify that everything works.

---

## What This Project Does

- You upload **PDF** or **DOCX** files (up to three distinct documents per browser session).
- The backend **extracts text**, **splits** it into chunks, **embeds** them into vectors, and stores them in **ChromaDB**.
- You ask **questions in Arabic or English**; the app **retrieves** relevant chunks and streams an **answer** from the language model.
- The web UI is **Angular**; the API is **FastAPI**; **Chroma** runs as its own container.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Docker** | Docker Desktop (Windows/macOS) or Docker Engine + Compose (Linux). |
| **OpenAI API key** | Required for embeddings and chat. Create a key in your OpenAI account. |
| **Internet** | Needed the first time Docker pulls images and when calling OpenAI. |

You do **not** need Python or Node installed on your machine if you only use Docker.

---

## Step 1 — Get the Code

```bash
git clone <your-repository-url>
cd RAG-Advanced
```

(Use the folder name you actually have if it differs.)

---

## Step 2 — Configure Environment

1. Copy **`.env.example`** to **`.env`** in the **project root** (same folder as `docker-compose.yml`).
2. Open **`.env`** and set:

```env
OPENAI_API_KEY=sk-your-key-here
```

3. Save the file.

Other variables (Chroma host/port, CORS) are normally set inside **`docker-compose.yml`** for containers. You rarely need to change them for a default local run.

---

## Step 3 — Start All Services

From the project root:

```bash
docker compose up --build
```

- The **first** run downloads base images and builds the API and frontend images; this can take several minutes.
- Leave this terminal open to see logs. Press **Ctrl+C** to stop.

To run in the background:

```bash
docker compose up --build -d
```

---

## Step 4 — Open the Application

| Service | URL | Purpose |
|---------|-----|---------|
| **Web UI** | [http://localhost:4200](http://localhost:4200) | Main application (Angular behind nginx). |
| **API health** | [http://localhost:8080/api/health](http://localhost:8080/api/health) | Should return `{"status":"ok"}`. |
| **Chroma** | `http://localhost:8000` | Used internally by the API; not a full browser app. |

The UI talks to the API through **nginx** (same site, `/api/...`), so you usually do not configure CORS manually in the browser.

---

## Step 5 — Use the UI (Quick Test)

1. Open **http://localhost:4200**.
2. Upload one or more **PDF** or **DOCX** files (drag-and-drop or click the drop zone).
3. Click **Index documents** (or equivalent) and wait until indexing finishes.
4. Type a question in the chat box and send. Answers **stream** token by token when the model responds.
5. If the answer is grounded in your files, **Sources** may list file names and pages.

---

## Stopping and Cleaning Up

| Command | Effect |
|---------|--------|
| `docker compose down` | Stops and removes containers (named volumes like Chroma data are kept unless you remove them explicitly). |
| `docker compose logs -f app` | Follow logs from the API container (useful for debugging). |

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| **API errors or empty answers** | Confirm `OPENAI_API_KEY` in `.env`, then `docker compose up --build` again. |
| **Upload too large (HTTP 413)** | Reduce PDF size or adjust nginx body size in `frontend/nginx.conf` and rebuild the **frontend** image. |
| **Port already in use** | Stop other apps using ports **4200**, **8080**, or **8000**, or change port mappings in `docker-compose.yml`. |
| **Slow indexing or rate limits** | In `.env`, try `EMBEDDING_MAX_PARALLEL=1` or a smaller `EMBEDDING_BATCH_SIZE`, then rebuild the **app** service. |

---

## Optional: Run Without Full Docker (Developers)

If you want to run Python or Angular on the host with hot reload, install **Python 3.11+**, **Node 20+**, and keep **Chroma** running (for example `docker compose up -d chromadb` only). Full commands and proxy details are in **`README.md`**.

---

## Project Layout (Short)

| Path | Role |
|------|------|
| `app/` | Python: FastAPI app, RAG pipeline, loaders, chunker, vector store client |
| `app/api/main.py` | HTTP routes and session handling |
| `frontend/` | Angular source and Docker nginx config |
| `docker-compose.yml` | Defines Chroma, API, and frontend services |
| `Dockerfile` | Builds the Python API image |
| `requirements.txt` | Python dependencies (used by Docker build and optional local runs) |

---

## Submission Checklist

- [ ] `.env` is **not** committed; only `.env.example` is in the repo.
- [ ] `OPENAI_API_KEY` is set locally when running.
- [ ] `docker compose up --build` starts all services without errors.
- [ ] Health check and UI work as described above.

For more architecture notes and the same content in a slightly different shape, see **`README.md`**.
