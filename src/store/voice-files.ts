// Reference voices for cloning TTS (Qwen3-TTS Base), kept in the browser's private file storage (OPFS), never in
// IndexedDB (decision D27/D32): `voices/<id>.bin` is the clip as raw little-endian float32, mono, 24 kHz, and
// `voices/<id>.json` the small record beside it. The clip is hashed when saved and checked again every time it is
// loaded, so a damaged or swapped file is refused. Decks read the same files.

import { Sha256 } from './sha256';

export const VOICE_SAMPLE_RATE = 24000;
const VOICES_DIR = 'voices';

export interface VoiceMeta {
  id: string;
  name: string;
  /** What the clip says, word for word (the model aligns it with the clip). */
  transcript: string;
  sampleRate: typeof VOICE_SAMPLE_RATE;
  createdAt: number;
  /** Samples in the clip. */
  samples: number;
  /** SHA-256 (hex) of the stored `.bin`. */
  sha256: string;
  /** A clip the app was given rather than one the user made: it can be used and deleted, never played or exported. */
  protected: boolean;
}

export interface VoiceClip extends VoiceMeta {
  audio: Float32Array;
}

/** The clip as the bytes stored in a `.bin` (little-endian float32). */
export function clipBytes(audio: Float32Array): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(audio.byteLength);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < audio.length; i++) view.setFloat32(i * 4, audio[i]!, true);
  return bytes;
}

/** The samples in a `.bin`'s bytes. */
export function clipSamples(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const audio = new Float32Array(Math.floor(bytes.byteLength / 4));
  for (let i = 0; i < audio.length; i++) audio[i] = view.getFloat32(i * 4, true);
  return audio;
}

export function sha256Hex(bytes: Uint8Array): string {
  return new Sha256().update(bytes).hex();
}

/** Throws if `bytes` are not the clip the record describes. */
export function verifyClip(meta: Pick<VoiceMeta, 'sha256' | 'samples' | 'name'>, bytes: Uint8Array): void {
  if (bytes.byteLength !== meta.samples * 4 || sha256Hex(bytes) !== meta.sha256) {
    throw new Error(`The saved voice “${meta.name}” failed its integrity check; delete it and add it again.`);
  }
}

async function dir(create: boolean): Promise<FileSystemDirectoryHandle | null> {
  const root = await navigator.storage.getDirectory();
  try {
    return await root.getDirectoryHandle(VOICES_DIR, { create });
  } catch (err) {
    if (!create && err instanceof DOMException && err.name === 'NotFoundError') return null;
    throw err;
  }
}

async function write(directory: FileSystemDirectoryHandle, name: string, data: Uint8Array<ArrayBuffer> | string): Promise<void> {
  const writable = await (await directory.getFileHandle(name, { create: true })).createWritable();
  try {
    await writable.write(data);
  } finally {
    await writable.close();
  }
}

/** The saved voices' records (no audio), oldest first. */
export async function listVoices(): Promise<VoiceMeta[]> {
  const directory = await dir(false);
  if (!directory) return [];
  const metas: VoiceMeta[] = [];
  for await (const [name, handle] of (directory as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
    if (!name.endsWith('.json') || handle.kind !== 'file') continue;
    try {
      metas.push(JSON.parse(await (await (handle as FileSystemFileHandle).getFile()).text()) as VoiceMeta);
    } catch {
      // A half-written record: ignored (its clip is cleaned up when the voice is saved again or deleted).
    }
  }
  return metas.sort((a, b) => a.createdAt - b.createdAt);
}

/** Loads a voice's clip and checks it against its record. */
export async function loadVoice(id: string): Promise<VoiceClip> {
  const directory = await dir(false);
  if (!directory) throw new Error('That voice is no longer saved on this device.');
  let meta: VoiceMeta;
  let bytes: Uint8Array;
  try {
    meta = JSON.parse(await (await (await directory.getFileHandle(`${id}.json`)).getFile()).text()) as VoiceMeta;
    bytes = new Uint8Array(await (await (await directory.getFileHandle(`${id}.bin`)).getFile()).arrayBuffer());
  } catch {
    throw new Error('That voice is no longer saved on this device.');
  }
  verifyClip(meta, bytes);
  return { ...meta, audio: clipSamples(bytes) };
}

/** Saves a voice (the clip first, then its record, so a record always has its clip). */
export async function saveVoice(fields: { name: string; transcript: string; audio: Float32Array; protected?: boolean }): Promise<VoiceMeta> {
  const directory = (await dir(true))!;
  const id = crypto.randomUUID();
  const bytes = clipBytes(fields.audio);
  const meta: VoiceMeta = {
    id,
    name: fields.name,
    transcript: fields.transcript,
    sampleRate: VOICE_SAMPLE_RATE,
    createdAt: Date.now(),
    samples: fields.audio.length,
    sha256: sha256Hex(bytes),
    protected: fields.protected ?? false,
  };
  await write(directory, `${id}.bin`, bytes);
  await write(directory, `${id}.json`, JSON.stringify(meta));
  return meta;
}

export async function deleteVoice(id: string): Promise<void> {
  const directory = await dir(false);
  if (!directory) return;
  // The record first: without it the voice is gone from every list even if removing the clip fails.
  for (const name of [`${id}.json`, `${id}.bin`]) await directory.removeEntry(name).catch(() => {});
}
