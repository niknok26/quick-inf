// Narration audio made on this device, kept in IndexedDB (`narration`) so a deck's slides are spoken once and then
// replayed for free. The key is a hash of everything that changes the audio: engine version, model, voice, language
// and text. Nothing here leaves the device. The cache is capped (2 GiB by default): when a new clip would take it
// over, the clips used longest ago are dropped (decision D32). Each clip has a small record (size, last use) in
// `narrationMeta`, so touching a clip on use does not rewrite its audio.

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


/** Default cap on the cache's total size. */
export const NARRATION_CAP_BYTES = 2 * 1024 ** 3;

export interface NarrationMeta {
  key: string;
  size: number;
  /** When the clip was last made or played (ms since the epoch). */
  lastAccessed: number;
}

/** The keys to drop, least recently used first, to bring the total to `cap`; never `keep` (the clip just made). */
export function selectEvictions(metas: readonly NarrationMeta[], cap: number, keep?: string): string[] {
  let total = metas.reduce((n, m) => n + m.size, 0);
  const evict: string[] = [];
  for (const m of [...metas].sort((a, b) => a.lastAccessed - b.lastAccessed)) {
    if (total <= cap) break;
    if (m.key === keep) continue;
    evict.push(m.key);
    total -= m.size;
  }
  return evict;
}

/** A cached clip, marked as just used. */
export async function getNarration(key: string): Promise<NarrationRow | undefined> {
  const row = await dbGet<NarrationRow>('narration', key);
  if (row) await dbPut<NarrationMeta>('narrationMeta', { key, size: row.wav.size, lastAccessed: Date.now() });
  return row;
}

/** Caches a clip, then drops the least recently used ones if the cache is over `cap`. */
export async function putNarration(row: NarrationRow, cap = NARRATION_CAP_BYTES): Promise<void> {
  await dbPut('narration', row);
  await dbPut<NarrationMeta>('narrationMeta', { key: row.key, size: row.wav.size, lastAccessed: Date.now() });
  for (const key of selectEvictions(await dbGetAll<NarrationMeta>('narrationMeta'), cap, row.key)) await deleteNarration(key);
}

export async function deleteNarration(key: string): Promise<void> {
  await dbDelete('narration', key);
  await dbDelete('narrationMeta', key);
}
