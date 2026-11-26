// Development: do the Qwen3-TTS lookup and codec graphs compute the same on this device's GPU as on another? Runs the
// text and codec embedding lookups on fixed ids, and the speaker and codec encoders on a reference clip, and reports
// checksums, so two devices' pages can be compared. Results in `window.__bench`.
//
//   /graph-check.html?model=Qwen__Qwen3-TTS-12Hz-0.6B-Base&ref=ref13

import { getGpuDevice } from '../device/gpu';
import { bytesSource } from '../runtime/onnx/external-weights';
import { createGraphSession } from '../runtime/onnx/onnx-adapter';
import { loadOrt } from '../runtime/onnx/ort-env';

const params = new URLSearchParams(location.search);
const repo = params.get('model') ?? 'Qwen__Qwen3-TTS-12Hz-0.6B-Base';
const ref = params.get('ref') ?? 'ref13';
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};
const fetchBytes = async (url: string) => new Uint8Array(await (await fetch(url)).arrayBuffer());
const model = (path: string) => fetchBytes(`/__models/${repo}/local/${path}`);
const summary = (a: ArrayLike<number>) => {
  let sum = 0;
  let abs = 0;
  let finite = true;
  for (let i = 0; i < a.length; i++) {
    const v = Number(a[i]);
    sum += v;
    abs += Math.abs(v);
    if (!Number.isFinite(v)) finite = false;
  }
  return `n ${a.length} sum ${sum.toFixed(3)} abs ${abs.toFixed(3)} finite ${finite} first ${Array.from({ length: 4 }, (_, i) => Number(a[i]).toFixed(4)).join(',')}`;
};

async function run(): Promise<void> {
  const ort = await loadOrt();
  const device = await getGpuDevice();
  const make = async (path: string) => createGraphSession(bytesSource(await model(path)), device, {}, []);
  const textEmbed = await make('text_embed_q4.onnx');
  const ids = BigInt64Array.of(100n, 5000n, 100000n, 151644n);
  const t = await textEmbed.run({ ids: new ort.Tensor('int64', ids, [1, 4]) });
  const te = t.embeds!.data as Float32Array;
  const w = te.length / 4;
  for (let i = 0; i < 4; i++) log(`text row ${ids[i]}: ${summary(te.subarray(i * w, (i + 1) * w))}`);
  const codecEmbed = await make('codec_embed_q4.onnx');
  const cids = BigInt64Array.of(5n, 2100n, 3100n, 20000n, 33000n);
  const c = await codecEmbed.run({ ids: new ort.Tensor('int64', cids, [1, 5]) });
  const ce = c.embeds!.data as Float32Array;
  const cw = ce.length / 5;
  for (let i = 0; i < 5; i++) log(`codec row ${cids[i]}: ${summary(ce.subarray(i * cw, (i + 1) * cw))}`);
  // The reference clip: decoded to 24 kHz mono, as the voice panel does.
  const wav = await fetchBytes(`/__models/refs/local/${ref}.wav`);
  const decoded = await new OfflineAudioContext(1, 1, 24000).decodeAudioData(wav.buffer.slice(wav.byteOffset, wav.byteOffset + wav.byteLength) as ArrayBuffer);
  const clip = decoded.getChannelData(0).slice();
  log(`clip ${clip.length} samples: ${summary(clip)}`);
  const speaker = await make('speaker_q4.onnx');
  const sp = await speaker.run({ audio: new ort.Tensor('float32', clip, [1, clip.length]) });
  log(`x-vector: ${summary(sp.xvector!.data as Float32Array)}`);
  const encoder = await make('tok_encoder_q4.onnx');
  const frames = Math.ceil(clip.length / 1920);
  const padded = new Float32Array(frames * 1920);
  padded.set(clip);
  const e = await encoder.run({ audio: new ort.Tensor('float32', padded, [1, padded.length]) });
  const codes = Array.from(e.codes!.data as BigInt64Array, Number);
  log(`codes ${frames} frames: group0 first 16: ${codes.slice(0, 16).join(',')}; group7 first 8: ${codes.slice(7 * frames, 7 * frames + 8).join(',')}; sum ${codes.reduce((a, b) => a + b, 0)}`);
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error.slice(0, 400)}`);
  })
  .finally(() => {
    state.done = true;
  });
