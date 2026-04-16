import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, ElementRef, HostListener, OnInit, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { IndexedDocument, ProcessApiResult, RagService } from './rag.service';

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

/** One uploaded file in the list (id is for Angular @for track). */
export interface SessionUploadRow {
  id: string;
  file: File;
  /** Set when restored from sessionStorage (placeholder File has size 0). */
  displaySize?: number;
}

@Component({
  selector: 'ng-root',
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  readonly title = 'Document RAG';
  readonly tagline = 'Ask questions grounded in your PDFs and Word files.';

  readonly processing = signal(false);
  readonly processPercent = signal(0);
  readonly processLabel = signal('');
  readonly chatting = signal(false);
  readonly error = signal<string | null>(null);
  readonly info = signal<string | null>(null);

  readonly history = signal<ChatTurn[]>([]);
  readonly draftQuestion = signal('');

  readonly uploadRows = signal<SessionUploadRow[]>([]);
  readonly indexedDocuments = signal<IndexedDocument[]>([]);
  readonly indexedDistinctCount = signal(0);
  readonly removingIndexed = signal(false);
  readonly clearingServer = signal(false);
  readonly clearServerIndexDialogOpen = signal(false);
  readonly dragActive = signal(false);

  private readonly fileInput = viewChild<ElementRef<HTMLInputElement>>('fileInput');

  constructor(public readonly rag: RagService) {}

  ngOnInit(): void {
    this.restoreUploadRowsFromStorage();
    void this.loadChatHistoryFromServer();
    void this.loadCorpusFromServer();
  }

  private uploadRowsStorageKey(): string {
    return `rag_corpus_upload_rows_v1_${this.rag.getSessionId()}`;
  }

  private persistUploadRowsToStorage(): void {
    try {
      const rows = this.uploadRows();
      if (rows.length === 0) {
        sessionStorage.removeItem(this.uploadRowsStorageKey());
        return;
      }
      const payload = rows.map((r) => ({
        id: r.id,
        name: r.file.name,
        size: this.logicalByteSize(r),
        lastModified: r.file.lastModified,
      }));
      sessionStorage.setItem(this.uploadRowsStorageKey(), JSON.stringify(payload));
    } catch {
      /* quota / private mode */
    }
  }

  private restoreUploadRowsFromStorage(): void {
    try {
      const raw = sessionStorage.getItem(this.uploadRowsStorageKey());
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw) as { id: string; name: string; size: number; lastModified: number }[];
      if (!Array.isArray(parsed) || parsed.length === 0) {
        return;
      }
      const rows: SessionUploadRow[] = parsed.slice(0, MAX_FILES).map((p) => ({
        id: p.id,
        displaySize: p.size,
        file: new File([], p.name, {
          lastModified: p.lastModified,
          type: 'application/octet-stream',
        }),
      }));
      this.uploadRows.set(rows);
    } catch {
      /* ignore */
    }
  }

  private setUploadRows(rows: SessionUploadRow[]): void {
    this.uploadRows.set(rows);
    this.persistUploadRowsToStorage();
  }

  logicalByteSize(row: SessionUploadRow): number {
    return row.displaySize ?? row.file.size;
  }

  private async loadCorpusFromServer(): Promise<void> {
    try {
      const res = await this.rag.fetchCorpus();
      const incoming = res.documents ?? [];
      const n = typeof res.distinct_count === 'number' ? res.distinct_count : incoming.length;
      const localDocs = this.indexedDocuments();
      const localN = this.indexedDistinctCount();
      if (
        incoming.length === 0 &&
        n === 0 &&
        (localDocs.length > 0 || localN > 0) &&
        this.uploadRows().length > 0
      ) {
        return;
      }
      this.indexedDocuments.set(incoming);
      this.indexedDistinctCount.set(n);
      if (this.uploadRows().length === 0 && n > 0) {
        this.info.set(
          `${n} document(s) on the server — remove one or clear index to add more.`,
        );
      }
    } catch {
      /* offline */
    }
  }

  private async loadChatHistoryFromServer(): Promise<void> {
    try {
      const turns = await this.rag.fetchChatHistory();
      this.history.set(
        turns.map((t) => ({
          question: t.question,
          answer: t.answer,
          sources: t.sources ?? [],
        })),
      );
    } catch {
      /* offline or cold start */
    }
  }

  private httpErr(e: unknown): string {
    if (e instanceof HttpErrorResponse) {
      const d = e.error as { detail?: unknown } | undefined;
      return typeof d?.detail === 'string' ? d.detail : e.message;
    }
    return e instanceof Error ? e.message : String(e);
  }

  fileCount(): number {
    return this.uploadRows().length;
  }

  private namesLikelySame(serverName: string, fileName: string): boolean {
    const a = serverName.trim();
    const b = fileName.trim();
    if (a === b) {
      return true;
    }
    const al = a.toLowerCase();
    const bl = b.toLowerCase();
    if (al === bl) {
      return true;
    }
    const base = (s: string) => s.replace(/^.*[/\\]/, '');
    return base(al) === base(bl);
  }

  docForUploadedRow(row: SessionUploadRow): IndexedDocument | undefined {
    const f = row.file;
    const sz = this.logicalByteSize(row);
    const docs = this.indexedDocuments();
    return (
      docs.find(
        (d) =>
          this.namesLikelySame(d.filename, f.name) && (d.size == null || d.size === sz),
      ) ?? docs.find((d) => this.namesLikelySame(d.filename, f.name))
    );
  }

  isRowIndexed(row: SessionUploadRow): boolean {
    return this.docForUploadedRow(row) !== undefined;
  }

  private isNewFileIndexedOnServer(f: File): boolean {
    const docs = this.indexedDocuments();
    return (
      docs.find(
        (d) =>
          this.namesLikelySame(d.filename, f.name) && (d.size == null || d.size === f.size),
      ) !== undefined
    );
  }

  pendingQueueCount(): number {
    return this.uploadRows().filter((r) => !this.isRowIndexed(r)).length;
  }

  corpusTotal(): number {
    return this.indexedDistinctCount() + this.pendingQueueCount();
  }

  slotsForQueue(): number {
    return Math.max(0, MAX_FILES - this.indexedDistinctCount() - this.pendingQueueCount());
  }

  async removeIndexedDocument(doc: IndexedDocument): Promise<void> {
    this.error.set(null);
    this.removingIndexed.set(true);
    try {
      await this.rag.removeCorpusDocument({
        sha256: doc.sha256 ?? undefined,
        filename: doc.filename,
      });
      await this.loadCorpusFromServer();
      await this.loadChatHistoryFromServer();
      this.refreshInfoAfterCorpusChange();
    } catch (e) {
      this.error.set(this.httpErr(e));
    } finally {
      this.removingIndexed.set(false);
    }
  }

  openClearServerIndexDialog(): void {
    this.clearServerIndexDialogOpen.set(true);
  }

  closeClearServerIndexDialog(): void {
    this.clearServerIndexDialogOpen.set(false);
  }

  @HostListener('document:keydown.escape', ['$event'])
  onDocumentEscape(event: Event): void {
    if (this.clearServerIndexDialogOpen()) {
      event.preventDefault();
      this.closeClearServerIndexDialog();
    }
  }

  async performClearServerIndex(): Promise<void> {
    this.closeClearServerIndexDialog();
    this.error.set(null);
    this.clearingServer.set(true);
    try {
      await this.rag.clearServerCorpus();
      this.indexedDocuments.set([]);
      this.indexedDistinctCount.set(0);
      await this.loadCorpusFromServer();
      await this.loadChatHistoryFromServer();
      this.info.set('Server index cleared. You can add up to 3 documents again.');
    } catch (e) {
      this.error.set(this.httpErr(e));
    } finally {
      this.clearingServer.set(false);
    }
  }

  private refreshInfoAfterCorpusChange(): void {
    const ix = this.indexedDistinctCount();
    const pending = this.pendingQueueCount();
    const listed = this.uploadRows().length;
    if (listed > 0) {
      this.info.set(
        pending > 0
          ? ix > 0
            ? `${pending} not indexed yet · ${ix} on server (${listed} in your list)`
            : `${pending} ready to index`
          : ix > 0
            ? `${listed} in your list (all indexed on the server)`
            : null,
      );
    } else if (ix > 0) {
      this.info.set(
        `${ix} document(s) on the server — remove one or clear index to add more.`,
      );
    } else {
      this.info.set(null);
    }
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

    const maxPending = MAX_FILES - this.indexedDistinctCount();
    const current = this.uploadRows();
    const merged: SessionUploadRow[] = [...current];
    for (const f of valid) {
      if (merged.some((x) => x.file.name === f.name && x.file.size === f.size)) {
        continue;
      }
      const wouldBePending = !this.isNewFileIndexedOnServer(f);
      const pendingInMerged = merged.filter((x) => !this.isRowIndexed(x)).length;
      if (wouldBePending && pendingInMerged >= maxPending) {
        this.error.set(
          maxPending <= 0
            ? `Corpus full (${MAX_FILES} documents). Remove a file below or clear the server index.`
            : `Queue full for this session (${MAX_FILES} distinct documents max, including server index).`,
        );
        this.clearFileInput();
        return;
      }
      merged.push({ id: crypto.randomUUID(), file: f });
    }

    this.setUploadRows(merged);
    const ix = this.indexedDistinctCount();
    const pend = merged.filter((r) => !this.isRowIndexed(r)).length;
    this.info.set(
      merged.length > 0
        ? pend > 0
          ? ix > 0
            ? `${pend} not indexed yet · ${ix} on server (${merged.length} in your list)`
            : `${pend} ready to index`
          : ix > 0
            ? `${merged.length} in your list (all indexed on the server)`
            : null
        : ix > 0
          ? `${ix} document(s) on the server — remove one or clear index to add more.`
          : null,
    );
    this.clearFileInput();
  }

  onFileInputChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.addFilesFromList(input.files);
  }

  async removeUploadRow(rowId: string): Promise<void> {
    const row = this.uploadRows().find((r) => r.id === rowId);
    if (!row) {
      return;
    }
    const doc = this.docForUploadedRow(row);
    if (doc) {
      this.error.set(null);
      this.removingIndexed.set(true);
      try {
        await this.rag.removeCorpusDocument({
          sha256: doc.sha256 ?? undefined,
          filename: doc.filename,
        });
        await this.loadCorpusFromServer();
        await this.loadChatHistoryFromServer();
      } catch (e) {
        this.error.set(this.httpErr(e));
        return;
      } finally {
        this.removingIndexed.set(false);
      }
    }
    this.uploadRows.update((list) => list.filter((r) => r.id !== rowId));
    this.persistUploadRowsToStorage();
    this.refreshInfoAfterCorpusChange();
    this.clearFileInput();
  }

  clearPendingFiles(): void {
    this.uploadRows.update((list) => list.filter((r) => this.isRowIndexed(r)));
    this.persistUploadRowsToStorage();
    this.refreshInfoAfterCorpusChange();
    this.error.set(null);
    this.clearFileInput();
  }

  private applyProcessResultToCorpusState(result: ProcessApiResult): void {
    const byHash = new Map<string, IndexedDocument>();
    for (const d of this.indexedDocuments()) {
      if (d.sha256) {
        byHash.set(d.sha256, d);
      }
    }
    const put = (m: { filename: string; sha256: string; size: number | null }) => {
      byHash.set(m.sha256, {
        filename: m.filename,
        sha256: m.sha256,
        size: m.size,
      });
    };
    for (const m of result.indexed ?? []) {
      put({ filename: m.filename, sha256: m.sha256, size: m.size });
    }
    for (const m of result.skipped_already_indexed ?? []) {
      if (!byHash.has(m.sha256)) {
        put({ filename: m.filename, sha256: m.sha256, size: null });
      }
    }
    const list = [...byHash.values()].sort((a, b) =>
      String(a.filename).toLowerCase().localeCompare(String(b.filename).toLowerCase()),
    );
    if (list.length === 0) {
      return;
    }
    this.indexedDocuments.set(list);
    this.indexedDistinctCount.set(Math.min(MAX_FILES, list.length));
  }

  onDragOver(event: DragEvent): void {
    if (this.processing() || this.removingIndexed() || this.clearingServer() || this.slotsForQueue() === 0) {
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
    if (this.processing() || this.removingIndexed() || this.clearingServer() || this.slotsForQueue() === 0) {
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
    const pendingRows = this.uploadRows().filter((r) => !this.isRowIndexed(r));
    const rowsWithBytes = pendingRows.filter((r) => r.file.size > 0);
    if (pendingRows.length > 0 && rowsWithBytes.length === 0) {
      this.error.set(
        'These entries were restored after a page reload and no longer contain file data. Remove them and add your PDFs or DOCX files again.',
      );
      return;
    }
    const files = rowsWithBytes.map((r) => r.file);
    if (files.length === 0) {
      this.error.set(
        this.uploadRows().length === 0
          ? `Add 1–${MAX_FILES} PDF or DOCX files first.`
          : 'Nothing new to index — these files are already on the server. Remove one below if you need a slot.',
      );
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

      this.applyProcessResultToCorpusState(result);
      await this.loadCorpusFromServer();

      const nPending = this.pendingQueueCount();
      const tail =
        nPending > 0
          ? ` — ${nPending} file(s) still waiting to be indexed.`
          : ' You can ask a question below.';
      this.info.set(`${result.message}${tail}`);
      this.persistUploadRowsToStorage();
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
