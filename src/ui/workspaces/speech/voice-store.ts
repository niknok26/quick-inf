// The saved reference voices, loaded from IndexedDB once and shared by every speech tab.

import { useEffect, useSyncExternalStore } from 'react';
import { deleteVoice, listVoices, putVoice, VOICE_SAMPLE_RATE, type VoiceRow } from '../../../store/voices';

let voices: readonly VoiceRow[] = [];
let loading: Promise<void> | null = null;
const listeners = new Set<() => void>();

function set(next: readonly VoiceRow[]): void {
  voices = next;
  for (const l of listeners) l();
}

function load(): Promise<void> {
  loading ??= listVoices().then(
    (rows) => set([...rows].sort((a, b) => a.createdAt - b.createdAt)),
    (err: unknown) => {
      loading = null;
      console.warn('Could not read the saved voices', err);
    },
  );
  return loading;
}

export const voiceStore = {
  /** A voice already loaded (null before the first load or when deleted). */
  get: (id: string): VoiceRow | null => voices.find((v) => v.id === id) ?? null,
  async create(name: string, transcript: string, audio: Float32Array): Promise<VoiceRow> {
    const row: VoiceRow = { id: crypto.randomUUID(), name, transcript, sampleRate: VOICE_SAMPLE_RATE, audio, createdAt: Date.now() };
    await putVoice(row);
    set([...voices, row]);
    return row;
  },
  async remove(id: string): Promise<void> {
    await deleteVoice(id);
    set(voices.filter((v) => v.id !== id));
  },
};

export function useVoices(): readonly VoiceRow[] {
  useEffect(() => void load(), []);
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => voices,
  );
}
