// Reference voices for cloning TTS (Qwen3-TTS Base), kept on the device in IndexedDB (`voices`, keyed by id).
// A record is the clip and what it says: decks read the same rows. Generated audio is never stored here.

import { dbDelete, dbGet, dbGetAll, dbPut } from './db';

export const VOICE_SAMPLE_RATE = 24000;

export interface VoiceRow {
  id: string;
  name: string;
  /** What the clip says, word for word (the model aligns it with the clip). */
  transcript: string;
  sampleRate: typeof VOICE_SAMPLE_RATE;
  /** The clip: mono, float32, at `sampleRate`. */
  audio: Float32Array;
  createdAt: number;
}

export const getVoice = (id: string) => dbGet<VoiceRow>('voices', id);
export const putVoice = (row: VoiceRow) => dbPut('voices', row);
export const listVoices = () => dbGetAll<VoiceRow>('voices');
export const deleteVoice = (id: string) => dbDelete('voices', id);
