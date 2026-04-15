import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnInit, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RagService, sha256Hex } from './rag.service';

export interface ChatTurn {
  question: string;
  answer: string;
  sources: string[];
}

const MAX_FILES = 3;
const ACCEPT_EXT = ['.pdf', '.docx'];

function isAllowedFile(f: File): boolean {
  const n = f.name.toLowerCase();
  return ACCEPT_EXT.some((ext) => n.endsWith(ext));
}

@Component({
  selector: 'ng-root',
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  readonly title = 'Document RAG';
  readonly tagline = 'Grounded answers from your PDFs and Word files — any language; sources when grounded.';
  readonly maxFiles = MAX_FILES;

  readonly processing = signal(false);
  readonly processPercent = signal(0);
  readonly processLabel = signal('');
  readonly chatting = signal(false);
  readonly error = signal<string | null>(null);
  readonly info = signal<string | null>(null);

  readonly history = signal<ChatTurn[]>([]);
  readonly draftQuestion = signal('');

  readonly selectedFiles = signal<File[]>([]);
  readonly dragActive = signal(false);

  private readonly fileInput = viewChild<ElementRef<HTMLInputElement>>('fileInput');

  constructor(public readonly rag: RagService) {}

  ngOnInit(): void {
    void this.loadChatHistoryFromServer();
  }

  private async loadChatHistoryFromServer(): Promise<void> {
    try {
      const turns = await this.rag.fetchChatHistory();
      if (turns.length > 0) {
        this.history.set(
          turns.map((t) => ({
            question: t.question,
            answer: t.answer,
            sources: t.sources ?? [],
          })),
        );
      }
    } catch {
      /* offline or cold start */
    }
  }

  fileCount(): number {
    return this.selectedFiles().length;
  }

  private clearFileInput(): void {
    const el = this.fileInput()?.nativeElement;
    if (el) {
      el.value = '';
    }
  }

  addFilesFromList(files: File[] | FileList | null): void {
    if (!files || files.length === 0) {
      return;
    }
    const incoming = Array.from(files);
    const valid = incoming.filter((f) => isAllowedFile(f));
    if (incoming.length !== valid.length) {
      this.error.set('Only PDF and DOCX files are allowed.');
    } else {
      this.error.set(null);
    }

    const current = this.selectedFiles();
    const merged: File[] = [...current];
    for (const f of valid) {
      if (merged.some((x) => x.name === f.name && x.size === f.size)) {
        continue;
      }
      if (merged.length >= MAX_FILES) {
        this.error.set(`At most ${MAX_FILES} files. Remove one to add another.`);
        this.clearFileInput();
        return;
      }
      merged.push(f);
    }

    this.selectedFiles.set(merged);
    this.info.set(merged.length > 0 ? `${merged.length}/${MAX_FILES} ready to index.` : null);
    this.clearFileInput();
  }

  onFileInputChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.addFilesFromList(input.files);
  }

  removeFile(index: number): void {
    this.selectedFiles.update((list) => list.filter((_, i) => i !== index));
    const n = this.selectedFiles().length;
    this.info.set(n > 0 ? `${n}/${MAX_FILES} ready to index.` : null);
    this.clearFileInput();
  }

  clearAllFiles(): void {
    this.selectedFiles.set([]);
    this.info.set(null);
    this.error.set(null);
    this.clearFileInput();
  }

  onDragOver(event: DragEvent): void {
    if (this.processing()) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.dragActive.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.dragActive.set(false);
  }

  onDrop(event: DragEvent): void {
    if (this.processing()) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.dragActive.set(false);
    const dt = event.dataTransfer;
    if (!dt?.files?.length) {
      return;
    }
    this.addFilesFromList(dt.files);
  }

  openFilePicker(): void {
    this.fileInput()?.nativeElement.click();
  }

  async processDocuments(): Promise<void> {
    const files = this.selectedFiles();
    if (files.length === 0) {
      this.error.set(`Add 1–${MAX_FILES} PDF or DOCX files first.`);
      return;
    }
    if (files.length > MAX_FILES) {
      this.error.set(`At most ${MAX_FILES} files allowed.`);
      return;
    }
    this.processing.set(true);
    this.processPercent.set(0);
    this.processLabel.set('Starting…');
    this.error.set(null);
    this.info.set(null);
    try {
      const result = await this.rag.processDocuments(files, (p) => {
        this.processPercent.set(p.percent);
        this.processLabel.set(p.label);
      });
      this.history.set([]);

      const indexedSet = new Set(result.indexed.map((x) => x.sha256));
      const hashes = await Promise.all(files.map((f) => sha256Hex(f)));
      this.selectedFiles.update((list) =>
        list.filter((f) => {
          const i = files.indexOf(f);
          if (i === -1) {
            return true;
          }
          return !indexedSet.has(hashes[i]);
        }),
      );

      const n = this.selectedFiles().length;
      const tail =
        n > 0
          ? ` — ${n} file(s) still in the queue (not indexed this round).`
          : ' You can ask a question below.';
      this.info.set(`${result.message}${tail}`);
    } catch (e) {
      this.error.set(e instanceof Error ? e.message : String(e));
    } finally {
      if (this.processPercent() >= 99) {
        await new Promise((r) => setTimeout(r, 450));
      }
      this.processing.set(false);
      this.processPercent.set(0);
      this.processLabel.set('');
    }
  }

  async sendQuestion(): Promise<void> {
    const q = this.draftQuestion().trim();
    if (!q) {
      return;
    }
    this.draftQuestion.set('');
    this.chatting.set(true);
    this.error.set(null);

    const turn: ChatTurn = { question: q, answer: '', sources: [] };
    this.history.update((h) => [...h, turn]);
    const idx = this.history().length - 1;

    try {
      for await (const ev of this.rag.chatStream(q)) {
        if (ev.type === 'token') {
          this.history.update((h) => {
            const next = [...h];
            const cur = { ...next[idx] };
            cur.answer += ev.text;
            next[idx] = cur;
            return next;
          });
        } else if (ev.type === 'done') {
          this.history.update((h) => {
            const next = [...h];
            const cur = { ...next[idx] };
            cur.sources = ev.sources;
            next[idx] = cur;
            return next;
          });
        } else if (ev.type === 'error') {
          this.error.set(ev.message);
          this.history.update((h) => {
            const next = [...h];
            const cur = { ...next[idx] };
            cur.answer = cur.answer || ev.message;
            next[idx] = cur;
            return next;
          });
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.error.set(msg);
    } finally {
      this.chatting.set(false);
    }
  }
}
