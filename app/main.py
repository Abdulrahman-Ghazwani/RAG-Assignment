"""
Run the HTTP API (same app Docker uses).

From the project root:

    python -m app.main

For development with auto-reload, prefer:

    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8080
"""

import uvicorn


def main() -> None:
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
