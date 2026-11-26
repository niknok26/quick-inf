// Narration audio made on this device, kept in IndexedDB (`narration`) so a deck's slides are spoken once and then
// replayed for free. The key is a hash of everything that changes the audio: engine version, model, voice, language
// and text. Nothing here leaves the device.

import { dbDelete, dbGet, dbGetAll, dbPut } from '../store/db';

/** Bump when the speech engine changes in a way that changes the audio, so old clips are not reused. */
export const NARRATION_CACHE_VERSION = 1;

export interface NarrationRow {
  key: string;
  wav: Blob;
  seconds: number;
  createdAt: number;
}

export interface NarrationKeyParts {
  modelId: string;
  /** A saved voice's id and creation time (a voice is never edited: a new clip is a new voice). */
  voiceId: string;
  voiceCreatedAt: number;
  language: string;
  text: string;
}

export async function narrationKey(parts: NarrationKeyParts): Promise<string> {
  const material = [NARRATION_CACHE_VERSION, parts.modelId, parts.voiceId, parts.voiceCreatedAt, parts.language, parts.text.trim()].join('\u0000');
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(material));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export const getNarration = (key: string) => dbGet<NarrationRow>('narration', key);
export const putNarration = (row: NarrationRow) => dbPut('narration', row);
export const deleteNarration = (key: string) => dbDelete('narration', key);
export const listNarration = () => dbGetAll<NarrationRow>('narration');
