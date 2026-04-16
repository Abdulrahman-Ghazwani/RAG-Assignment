import { HttpClient, HttpErrorResponse, HttpEventType, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { firstValueFrom } from 'rxjs';

/** Base URL for API calls. Empty string = same origin (dev proxy or Docker nginx). */
const API_BASE = '';

/** SHA-256 of file bytes (must match server-side deduplication). */
export async function sha256Hex(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const hash = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export interface ProcessApiResult {
  ok: boolean;
  message: string;
  indexed: { filename: string; size: number; sha256: string }[];
  skipped_already_indexed: { filename: string; sha256: string }[];
  skipped_duplicate_in_request: { filename: string; sha256: string; reason: string }[];
}

export type ProcessProgress = {
  percent: number;
  label: string;
};

export type ChatStreamEvent =
  | { type: 'token'; text: string }
  | { type: 'done'; sources: string[] }
  | { type: 'error'; message: string };

export interface ChatHistoryTurn {
  question: string;
  answer: string;
  sources: string[];
}

export interface IndexedDocument {
  filename: string;
  sha256: string | null;
  size: number | null;
}

export interface CorpusResponse {
  indexed: boolean;
  documents: IndexedDocument[];
  distinct_count?: number;
}

@Injectable({ providedIn: 'root' })
export class RagService {
  private readonly sessionId: string;

  constructor(private readonly http: HttpClient) {
    let id = localStorage.getItem('rag_session_id');
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem('rag_session_id', id);
    }
    this.sessionId = id;
  }

  getSessionId(): string {
    return this.sessionId;
  }

  async fetchChatHistory(): Promise<ChatHistoryTurn[]> {
    const res = await firstValueFrom(
      this.http.get<{ turns: ChatHistoryTurn[] }>(`${API_BASE}/api/chat/history`, {
        headers: new HttpHeaders({ 'X-Session-Id': this.sessionId }),
      }),
    );
    return res.turns ?? [];
  }

  async fetchCorpus(): Promise<CorpusResponse> {
    return await firstValueFrom(
      this.http.get<CorpusResponse>(`${API_BASE}/api/corpus`, {
        headers: new HttpHeaders({ 'X-Session-Id': this.sessionId }),
      }),
    );
  }

  async removeCorpusDocument(params: { sha256?: string | null; filename: string }): Promise<{ ok: boolean; removed: string }> {
    const body: { sha256?: string; filename: string } = { filename: params.filename };
    if (params.sha256) {
      body.sha256 = params.sha256;
    }
    return await firstValueFrom(
      this.http.post<{ ok: boolean; removed: string }>(`${API_BASE}/api/corpus/remove`, body, {
        headers: new HttpHeaders({
          'X-Session-Id': this.sessionId,
          'Content-Type': 'application/json',
        }),
      }),
    );
  }

  async clearServerCorpus(): Promise<{ ok: boolean; message: string }> {
    return await firstValueFrom(
      this.http.post<{ ok: boolean; message: string }>(`${API_BASE}/api/corpus/clear`, {}, {
        headers: new HttpHeaders({
          'X-Session-Id': this.sessionId,
          'Content-Type': 'application/json',
        }),
      }),
    );
  }

  /**
   * Uploads files to `/api/process` and reports progress.
   * The API does not stream indexing %, so we **simulate** phases: 0–45% from upload bytes,
   * then 45–96% on a timer until the HTTP response arrives (then 100%).
   */
  async processDocuments(
    files: File[],
    onProgress?: (p: ProcessProgress) => void,
  ): Promise<ProcessApiResult> {
    const form = new FormData();
    for (const f of files) {
      form.append('files', f);
    }
    const totalBytes = Math.max(1, files.reduce((s, f) => s + f.size, 0));

    // Timers only drive the fake progress bar; the real work is the single POST request.
    let indexTimer: ReturnType<typeof setInterval> | null = null;
    let indexPct = 45;
    let indexStarted = false;
    let fallbackTimer: ReturnType<typeof setTimeout> | null = null;

    const stopIndexTimer = (): void => {
      if (indexTimer) {
        clearInterval(indexTimer);
        indexTimer = null;
      }
    };

    const clearFallback = (): void => {
      if (fallbackTimer) {
        clearTimeout(fallbackTimer);
        fallbackTimer = null;
      }
    };

    const startIndexPhase = (): void => {
      if (indexStarted) {
        return;
      }
      indexStarted = true;
      clearFallback();
      indexPct = 45;
      onProgress?.({ percent: 45, label: 'Indexing & embedding…' });
      indexTimer = setInterval(() => {
        indexPct = Math.min(indexPct + 0.65, 96);
        onProgress?.({ percent: Math.round(indexPct), label: 'Indexing & embedding…' });
      }, 220);
    };

    // If upload events are slow to fire, still switch to "indexing" UI after 600ms.
    fallbackTimer = setTimeout(() => {
      if (!indexStarted) {
        startIndexPhase();
      }
    }, 600);

    return new Promise((resolve, reject) => {
      this.http
        .post<ProcessApiResult>(`${API_BASE}/api/process`, form, {
          headers: new HttpHeaders({ 'X-Session-Id': this.sessionId }),
          reportProgress: true,
          observe: 'events',
        })
        .subscribe({
          next: (event) => {
            if (event.type === HttpEventType.UploadProgress) {
              const ev = event;
              const total = ev.total && ev.total > 0 ? ev.total : totalBytes;
              const loaded = ev.loaded;
              // Map byte progress to 0–45% of the bar; remaining % reserved for "indexing" phase.
              const uploadPct = Math.min(45, Math.round((45 * loaded) / total));
              onProgress?.({ percent: uploadPct, label: 'Uploading…' });
              if (loaded >= total) {
                startIndexPhase();
              }
            }
            if (event.type === HttpEventType.Response) {
              clearFallback();
              stopIndexTimer();
              onProgress?.({ percent: 100, label: 'Done' });
              const body = event.body;
              if (!body) {
                reject(new Error('Empty response'));
                return;
              }
              resolve(body);
            }
          },
          error: (err: HttpErrorResponse) => {
            clearFallback();
            stopIndexTimer();
            if (err.status === 413) {
              reject(
                new Error(
                  'Upload too large for the server limit. Try smaller PDFs or fewer pages, or ask your admin to raise the upload size.',
                ),
              );
              return;
            }
            let detail = err.message;
            const e = err.error as { detail?: unknown } | undefined;
            if (e?.detail) {
              detail = typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail);
            }
            reject(new Error(detail || `HTTP ${err.status}`));
          },
        });
    });
  }

  /**
   * Reads the server's SSE (`text/event-stream`): JSON lines `data: {...}` with `token` chunks
   * until `done: true` plus optional `sources`. Incomplete chunks stay in `buffer` across reads.
   */
  async *chatStream(question: string): AsyncGenerator<ChatStreamEvent> {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-Id': this.sessionId,
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = (await res.json()) as { detail?: unknown };
        if (j?.detail) {
          detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
        }
      } catch {
        /* ignore */
      }
      yield { type: 'error', message: detail || `HTTP ${res.status}` };
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      yield { type: 'error', message: 'No response body' };
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by blank lines; last fragment may be incomplete.
      const parts = buffer.split('\n\n');
      buffer = parts.pop() ?? '';

      for (const block of parts) {
        const line = block.trim().split('\n').find((l) => l.startsWith('data: '));
        if (!line) {
          continue;
        }
        try {
          const data = JSON.parse(line.slice(6)) as Record<string, unknown>;
          if (typeof data['token'] === 'string' && data['token'].length > 0) {
            yield { type: 'token', text: data['token'] as string };
          }
          if (data['done'] === true) {
            const sources = Array.isArray(data['sources'])
              ? (data['sources'] as string[])
              : [];
            yield { type: 'done', sources };
          }
        } catch {
          yield { type: 'error', message: 'Invalid server stream' };
        }
      }
    }
  }
}
