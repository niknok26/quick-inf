// Narrates a deck on this device: when a deck opens, `start` speaks its slides one at a time (the current slide first,
// then the ones after it, then the ones before), caching each slide's audio so reopening the deck costs nothing.
// The speech model makes audio slower than real time, so what matters is that a slide is ready before the presenter
// reaches it: `setCurrent` moves the one they are on to the front. Only text goes anywhere; the audio never leaves
// the device.

import type { RuntimeClient } from '../runtime/client';
import { RuntimeError } from '../runtime/contract';
import { encodeWav } from '../ui/workspaces/speech/wav';
import { getNarration, narrationKey, putNarration } from './narration-cache';
import { splitForSpeech } from './split-text';

export interface NarrationSlide {
  id: string;
  text: string;
}

/** A saved reference voice (see src/store/voices.ts). */
export interface NarrationVoice {
  id: string;
  createdAt: number;
  audio: Float32Array;
  transcript: string;
}

export interface NarratorOptions {
  client: Pick<RuntimeClient, 'load' | 'generateSpeech'>;
  modelId: string;
  voice: NarrationVoice;
  language?: string;
  /** Longest stretch of text spoken in one request (whole sentences). */
  pieceChars?: number;
}

export type SlideState = 'cached' | 'queued' | 'generating' | 'failed';

export interface SlideStatus {
  state: SlideState;
  /** While generating: the piece being spoken, of the slide's pieces. */
  piece?: { index: number; of: number };
  seconds?: number;
  error?: string;
}

const PAUSE_SECONDS = 0.2;

interface Entry {
  slide: NarrationSlide;
  key: string;
  status: SlideStatus;
}

export class Narrator {
  private entries: Entry[] = [];
  private current = 0;
  private active: { entry: Entry; cancel: () => void } | null = null;
  private pumping = false;
  private stopped = false;
  private readonly listeners = new Set<() => void>();

  constructor(private readonly options: NarratorOptions) {}

  /** Starts narrating `slides`, beginning at `current`. Slides already cached are ready at once. */
  async start(slides: readonly NarrationSlide[], current = 0): Promise<void> {
    this.stopped = false;
    this.current = current;
    const { modelId, voice, language = 'english' } = this.options;
    this.entries = await Promise.all(
      slides.map(async (slide) => {
        const key = await narrationKey({ modelId, voiceId: voice.id, voiceCreatedAt: voice.createdAt, language, text: slide.text });
        const cached = await getNarration(key);
        const status: SlideStatus = cached ? { state: 'cached', seconds: cached.seconds } : { state: 'queued' };
        return { slide, key, status };
      }),
    );
    this.notify();
    void this.pump();
  }

  /** The slide the presenter is on: it moves to the front of the queue. */
  setCurrent(index: number): void {
    this.current = index;
    const wanted = this.entries[index];
    // Work on another slide in progress is dropped if the current one isn't ready (what it had made is lost).
    if (this.active && wanted && wanted.status.state === 'queued' && this.active.entry !== wanted) {
      this.active.cancel();
    }
  }

  status(slideId: string): SlideStatus | undefined {
    return this.entries.find((e) => e.slide.id === slideId)?.status;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** The slide's audio (a WAV), waiting for it to be made if need be. */
  async audio(slideId: string): Promise<Blob> {
    const entry = this.entries.find((e) => e.slide.id === slideId);
    if (!entry) throw new Error(`Unknown slide ${slideId}`);
    for (;;) {
      if (entry.status.state === 'cached') {
        const row = await getNarration(entry.key);
        if (row) return row.wav;
        // Evicted from the cache since: make it again.
        entry.status = { state: 'queued' };
        this.notify();
        void this.pump();
      }
      if (entry.status.state === 'failed') throw new Error(entry.status.error ?? 'Narration failed');
      if (this.stopped) throw new RuntimeError('cancelled', 'Narration stopped.');
      await new Promise<void>((resolve) => {
        const off = this.subscribe(() => {
          off();
          resolve();
        });
      });
    }
  }

  /** Tries a failed slide again. */
  retry(slideId: string): void {
    const entry = this.entries.find((e) => e.slide.id === slideId);
    if (entry?.status.state !== 'failed') return;
    entry.status = { state: 'queued' };
    this.notify();
    void this.pump();
  }

  /** Stops generating (what is cached stays). */
  stop(): void {
    this.stopped = true;
    this.active?.cancel();
    this.notify();
  }

  private notify(): void {
    for (const l of this.listeners) l();
  }

  /** The next slide to make: the current one, then those after it, then those before. */
  private next(): Entry | undefined {
    const n = this.entries.length;
    for (let i = 0; i < n; i++) {
      const entry = this.entries[(this.current + i) % n]!;
      if (entry.status.state === 'queued') return entry;
    }
    return undefined;
  }

  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      await this.options.client.load(this.options.modelId);
      while (!this.stopped) {
        const entry = this.next();
        if (!entry) break;
        await this.make(entry);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      for (const e of this.entries) if (e.status.state === 'queued') e.status = { state: 'failed', error: message };
    } finally {
      this.pumping = false;
      this.notify();
    }
  }

  private async make(entry: Entry): Promise<void> {
    const { client, voice, language = 'english', pieceChars } = this.options;
    const pieces = splitForSpeech(entry.slide.text, pieceChars);
    if (pieces.length === 0) {
      entry.status = { state: 'failed', error: 'The slide has no text.' };
      this.notify();
      return;
    }
    const audio: Float32Array[] = [];
    let sampleRate = 24000;
    let cancelHandle = () => {};
    this.active = { entry, cancel: () => cancelHandle() };
    try {
      for (const [index, piece] of pieces.entries()) {
        entry.status = { state: 'generating', piece: { index: index + 1, of: pieces.length } };
        this.notify();
        const handle = client.generateSpeech(
          { text: piece, settings: { voice: voice.id, language }, reference: { audio: voice.audio, text: voice.transcript } },
          (p) => audio.push(p.audio),
        );
        cancelHandle = handle.cancel;
        if (this.stopped) handle.cancel();
        const result = await handle.result;
        sampleRate = result.sampleRate;
        if (index < pieces.length - 1) audio.push(new Float32Array(Math.round(PAUSE_SECONDS * sampleRate)));
      }
      const wav = encodeWav(audio, sampleRate);
      const seconds = audio.reduce((n, a) => n + a.length, 0) / sampleRate;
      await putNarration({ key: entry.key, wav, seconds, createdAt: Date.now() });
      entry.status = { state: 'cached', seconds };
    } catch (err) {
      if (err instanceof RuntimeError && err.code === 'cancelled') {
        // Dropped for another slide, or stopped: back in the queue (or left, if stopped).
        entry.status = { state: 'queued' };
      } else {
        entry.status = { state: 'failed', error: err instanceof Error ? err.message : String(err) };
      }
    } finally {
      this.active = null;
      this.notify();
    }
  }
}
