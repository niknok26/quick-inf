// Development voice-cloning smoke for Qwen3-TTS Base: loads the model through the app's runtime client and worker
// (files from the dev server's local model cache), speaks a line in the voice of a reference clip (the clip and its
// transcript sent as `reference` on the request) and reports the time and real-time factor, with the clip as a base64
// WAV. The reference lives in the dev model cache at refs/local/<name>.wav with its transcript in <name>.txt.
//
//   /clone-smoke.html?ref=<name>[&text=<sentence>][&lang=english]
// Results are exposed as `window.__bench`.

import { getRuntimeClient } from '../runtime/client';
import { encodeWav } from '../ui/workspaces/speech/wav';

const params = new URLSearchParams(location.search);
const MODEL_ID = params.get('model') ?? 'qwen3-tts-1.7b-base';
const SAMPLE_RATE = 24000;
const refName = params.get('ref') ?? 'ref20';
const text = params.get('text') ?? 'The quick brown fox jumps over the lazy dog near the riverbank.';

const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

async function base64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** The clip as mono float32 at the model's rate (the browser decodes and resamples). */
async function referenceAudio(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const context = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await context.decodeAudioData(await response.arrayBuffer());
  return decoded.getChannelData(0).slice();
}

async function run(): Promise<void> {
  const base = `/__models/refs/local/${refName}`;
  const [audio, transcript] = await Promise.all([referenceAudio(`${base}.wav`), fetch(`${base}.txt`).then((r) => r.text())]);
  const refText = transcript.trim();
  log(`reference ${refName}: ${(audio.length / SAMPLE_RATE).toFixed(1)} s, "${refText.slice(0, 60)}…"`);
  const client = getRuntimeClient();
  const loadStart = performance.now();
  const loaded = await client.load(MODEL_ID);
  log(`loaded on the ${loaded.processor} in ${Math.round(performance.now() - loadStart)} ms`);
  const pieces: Float32Array[] = [];
  const handle = client.generateSpeech(
    { text, settings: { voice: 'clone', language: params.get('lang') ?? 'english' }, reference: { audio, text: refText } },
    (p) => pieces.push(p.audio),
  );
  const r = await handle.result;
  const seconds = r.samples / r.sampleRate;
  log(`${seconds.toFixed(2)} s of audio in ${Math.round(r.timings.totalMs)} ms (rtf ${(r.timings.totalMs / 1000 / seconds).toFixed(2)})`);
  log(`stages (ms): ${JSON.stringify(Object.fromEntries(Object.entries(r.timings.stages ?? {}).map(([k, v]) => [k, Math.round(v)])))}`);
  state.results.clip = { text, refName, audioS: +seconds.toFixed(2), totalMs: Math.round(r.timings.totalMs), wav: await base64(encodeWav(pieces, r.sampleRate)) };
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
