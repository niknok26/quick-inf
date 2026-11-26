// Development latency probe for merging the Qwen3-TTS code predictor's 15 steps into one graph (see
// voice-lab/ONDEVICE-TTS-HANDOVER.md section 11). Compares, on the same inputs, one frame as the engine runs it today
// (15 separate runs of the builder graph, KV cache on the GPU, logits and hidden state read back each step) with one
// run of the same graph chained 15 times (tools: qwen-web/chain_cp.py). Teacher-forced: no sampling.
//
//   /cp-probe.html?model=Qwen__Qwen3-TTS-12Hz-0.6B-Base&frames=20
// Results in `window.__bench`.

import { getGpuDevice } from '../device/gpu';
import { bytesSource } from '../runtime/onnx/external-weights';
import { createGraphSession } from '../runtime/onnx/onnx-adapter';
import { loadOrt } from '../runtime/onnx/ort-env';

const params = new URLSearchParams(location.search);
const repo = params.get('model') ?? 'Qwen__Qwen3-TTS-12Hz-0.6B-Base';
const frames = Number(params.get('frames') ?? 20);
const base = `/__models/${repo}/local`;
const LAYERS = 5;
const KV_HEADS = 8;
const HEAD_DIM = 128;
const GROUP = 2048;
const STEPS = 15;
const WIDTH = 1024;

const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

const fetchBytes = async (url: string) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
};
const kv = (prefix: string) => Array.from({ length: LAYERS }, (_, i) => [`${prefix}.${i}.key`, `${prefix}.${i}.value`]).flat();
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;

async function run(): Promise<void> {
  const ort = await loadOrt();
  const device = await getGpuDevice();
  const [graph, data, probeGraph, noPastGraph] = await Promise.all([
    fetchBytes(`${base}/code_predictor/model.onnx`),
    fetchBytes(`${base}/code_predictor/model.onnx.data`),
    fetchBytes(`${base}/code_predictor_probe/model.onnx`),
    fetchBytes(`${base}/code_predictor_probe/model_nopast.onnx`),
  ]);
  const external = { 'model.onnx.data': bytesSource(data) };
  const baseline = await createGraphSession(bytesSource(graph), device, external, kv('present'));
  const chained = await createGraphSession(bytesSource(probeGraph), device, external, []);
  log(`sessions ready (baseline inputs ${baseline.inputNames.length}, chained inputs ${chained.inputNames.length})`);
  // The chained graph without its empty step-0 past inputs, its inputs and output on the GPU, with graph capture.
  const captured = await createGraphSession(bytesSource(noPastGraph), device, external, ['logits_all'], { graphCapture: true });
  const gpuUsage = 0x80 | 0x04 | 0x08; // STORAGE | COPY_SRC | COPY_DST
  const x0Buffer = device.createBuffer({ size: 2 * WIDTH * 4, usage: gpuUsage });
  const embBuffer = device.createBuffer({ size: (STEPS - 1) * WIDTH * 4, usage: gpuUsage });
  const x0Tensor = ort.Tensor.fromGpuBuffer(x0Buffer, { dataType: 'float32', dims: [1, 2, WIDTH] });
  const embTensor = ort.Tensor.fromGpuBuffer(embBuffer, { dataType: 'float32', dims: [1, STEPS - 1, WIDTH] });

  // Fixed pseudo-random rows (the same for both, every frame).
  let seed = 12345;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5) * 0.5;
  const x0 = Float32Array.from({ length: 2 * WIDTH }, random);
  const emb = Float32Array.from({ length: (STEPS - 1) * WIDTH }, random);
  const empty = () => new ort.Tensor('float32', new Float32Array(0), [1, KV_HEADS, 0, HEAD_DIM]);

  /** One frame the way the engine does it today; returns the 15 head slices. */
  async function frameBaseline(): Promise<Float32Array> {
    let past: Record<string, InstanceType<typeof ort.Tensor>> = Object.fromEntries(kv('past_key_values').map((n) => [n, empty()]));
    let length = 0;
    const result = new Float32Array(STEPS * GROUP);
    for (let j = 0; j < STEPS; j++) {
      const rows = j === 0 ? x0 : emb.subarray((j - 1) * WIDTH, j * WIDTH);
      const n = j === 0 ? 2 : 1;
      const feeds = {
        inputs_embeds: new ort.Tensor('float32', Float32Array.from(rows), [1, n, WIDTH]),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(length + n).fill(1n), [1, length + n]),
        ...past,
      };
      const res = await baseline.run(feeds);
      for (const t of Object.values(past)) t.dispose();
      past = Object.fromEntries(kv('present').map((name, i) => [kv('past_key_values')[i]!, res[name]!]));
      length += n;
      const logits = res.logits!.data as Float32Array;
      const row = logits.subarray(logits.length - 15 * GROUP - 0, logits.length); // last position's 30720
      result.set(row.subarray(j * GROUP, (j + 1) * GROUP), j * GROUP);
      void (res.hidden_states!.data as Float32Array); // the engine reads it back too
      res.logits!.dispose();
      res.hidden_states!.dispose();
    }
    for (const t of Object.values(past)) t.dispose();
    return result;
  }

  async function frameChained(): Promise<Float32Array> {
    const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {
      x0: new ort.Tensor('float32', x0, [1, 2, WIDTH]),
      emb: new ort.Tensor('float32', emb, [1, STEPS - 1, WIDTH]),
      ...Object.fromEntries(kv('past_key_values').map((n) => [n, empty()])),
    };
    const res = await chained.run(feeds);
    const logits = (res.logits_all!.data as Float32Array).slice();
    res.logits_all!.dispose();
    return logits;
  }

  async function frameCaptured(): Promise<Float32Array> {
    device.queue.writeBuffer(x0Buffer, 0, x0);
    device.queue.writeBuffer(embBuffer, 0, emb);
    const res = await captured.run({ x0: x0Tensor, emb: embTensor });
    const logits = ((await res.logits_all!.getData()) as Float32Array).slice();
    return logits;
  }

  // Correctness first, then timing (warm-up frames included).
  const a = await frameBaseline();
  const b = await frameChained();
  let maxDiff = 0;
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(a[i]! - b[i]!));
    maxAbs = Math.max(maxAbs, Math.abs(a[i]!));
  }
  log(`logits agree: max |diff| ${maxDiff.toExponential(2)} (max |logit| ${maxAbs.toFixed(2)})`);
  let captureOk = true;
  try {
    const c = await frameCaptured();
    let d = 0;
    for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i]! - c[i]!));
    log(`captured graph agrees: max |diff| ${d.toExponential(2)}`);
    await frameCaptured();
  } catch (err) {
    captureOk = false;
    log(`graph capture failed: ${err instanceof Error ? err.message.slice(200, 900) : String(err)}`);
  }
  for (let i = 0; i < 4; i++) {
    await frameBaseline();
    await frameChained();
  }
  const times: { baseline: number[]; chained: number[]; captured: number[] } = { baseline: [], chained: [], captured: [] };
  for (let i = 0; i < frames; i++) {
    let t = performance.now();
    await frameBaseline();
    times.baseline.push(performance.now() - t);
    t = performance.now();
    await frameChained();
    times.chained.push(performance.now() - t);
    if (captureOk) {
      t = performance.now();
      await frameCaptured();
      times.captured.push(performance.now() - t);
    }
  }
  const summary = {
    baselineMedianMs: +median(times.baseline).toFixed(1),
    chainedMedianMs: +median(times.chained).toFixed(1),
    baselineMinMs: +Math.min(...times.baseline).toFixed(1),
    chainedMinMs: +Math.min(...times.chained).toFixed(1),
  };
  state.results = { ...summary, maxDiff, maxAbs, frames, capturedMedianMs: captureOk ? +median(times.captured).toFixed(1) : null };
  if (captureOk) log(`captured graph (1 replay): median ${median(times.captured).toFixed(1)} ms, best ${Math.min(...times.captured).toFixed(1)} ms`);
  log(`per frame, median of ${frames}: baseline (15 runs) ${summary.baselineMedianMs} ms, chained (1 run) ${summary.chainedMedianMs} ms; best ${summary.baselineMinMs} vs ${summary.chainedMinMs} ms`);
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
