// Development: runs the Qwen3-TTS codec decoder alone on random codes, to find out whether (and how fast) it runs on a
// device's WebGPU. `opt` sets ONNX Runtime's graph optimization level. Results in `window.__bench`.
//
//   /decoder-probe.html?model=Qwen__Qwen3-TTS-12Hz-0.6B-Base&frames=40&opt=all|basic|disabled

import { getGpuDevice } from '../device/gpu';
import { bytesSource } from '../runtime/onnx/external-weights';
import { createGraphSession } from '../runtime/onnx/onnx-adapter';
import { loadOrt } from '../runtime/onnx/ort-env';

const params = new URLSearchParams(location.search);
const repo = params.get('model') ?? 'Qwen__Qwen3-TTS-12Hz-0.6B-Base';
const frames = Number(params.get('frames') ?? 40);
const optimization = (params.get('opt') ?? 'all') as 'all' | 'basic' | 'disabled';
const cpu = params.get('cpu') === '1';

const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

async function run(): Promise<void> {
  const ort = await loadOrt();
  const device = cpu ? null : await getGpuDevice();
  const response = await fetch(`/__models/${repo}/local/tok_decoder_q4c8.onnx`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  log(`decoder ${(bytes.length / 1e6).toFixed(0)} MB, optimization ${optimization}, ${cpu ? 'CPU' : 'GPU'}`);
  const session = await createGraphSession(bytesSource(bytes), device, {}, [], { optimization });
  log('session created');
  const codes = new BigInt64Array(16 * frames);
  for (let i = 0; i < codes.length; i++) codes[i] = BigInt((i * 7919) % 2048);
  for (let i = 0; i < 3; i++) {
    const t = performance.now();
    const res = await session.run({ codes: new ort.Tensor('int64', codes, [1, 16, frames]) });
    const audio = res.audio!.data as Float32Array;
    log(`run ${i}: ${Math.round(performance.now() - t)} ms, ${audio.length} samples (${(audio.length / 24000).toFixed(2)} s), first ${audio[1000]!.toFixed(4)}`);
    res.audio!.dispose();
  }
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error.slice(0, 600)}`);
  })
  .finally(() => {
    state.done = true;
  });
