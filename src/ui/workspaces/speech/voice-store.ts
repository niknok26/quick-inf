// The saved reference voices' records, read from the browser's file storage once and shared by every speech tab. The
// clips themselves are loaded only when a voice is used or previewed (see src/store/voice-files.ts).

import { useEffect, useSyncExternalStore } from 'react';
import { deleteVoice, listVoices, loadVoice, saveVoice, sha256Hex, type VoiceClip, type VoiceMeta } from '../../../store/voice-files';
import { decodeVoiceBlob } from './voice-audio';

let voices: readonly VoiceMeta[] = [];
let loading: Promise<void> | null = null;
const listeners = new Set<() => void>();

function set(next: readonly VoiceMeta[]): void {
  voices = next;
  for (const l of listeners) l();
}

function load(): Promise<void> {
  loading ??= listVoices().then(
    (rows) => set(rows),
    (err: unknown) => {
      loading = null;
      console.warn('Could not read the saved voices', err);
    },
  );
  return loading;
}

export const voiceStore = {
  /** A voice's record, if loaded (null before the first load or when deleted). */
  get: (id: string): VoiceMeta | null => voices.find((v) => v.id === id) ?? null,
  /** The voice's clip, checked against its record. */
  load: (id: string): Promise<VoiceClip> => loadVoice(id),
  async create(name: string, transcript: string, audio: Float32Array): Promise<VoiceMeta> {
    const meta = await saveVoice({ name, transcript, audio });
    set([...voices, meta]);
    return meta;
  },
  /**
   * Installs the reference clip the app is given (D27): an MP3 with its transcript, decoded on this device to 24 kHz
   * mono and kept as a protected voice (usable and deletable, never played or exported). `expectedSha256` is the
   * MP3's hash from the manifest, checked before decoding; the stored clip is then checked against its own hash
   * whenever it is loaded (decoders differ between browsers, so the decoded samples have no fixed hash).
   */
  async installProtected(fields: { name: string; transcript: string; mp3: Blob; expectedSha256?: string }): Promise<VoiceMeta> {
    if (fields.expectedSha256 && sha256Hex(new Uint8Array(await fields.mp3.arrayBuffer())) !== fields.expectedSha256) {
      throw new Error('The reference clip failed its integrity check.');
    }
    const meta = await saveVoice({ name: fields.name, transcript: fields.transcript, audio: await decodeVoiceBlob(fields.mp3), protected: true });
    set([...voices, meta]);
    return meta;
  },
  async remove(id: string): Promise<void> {
    await deleteVoice(id);
    set(voices.filter((v) => v.id !== id));
  },
};

export function useVoices(): readonly VoiceMeta[] {
  useEffect(() => void load(), []);
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => voices,
  );
}
