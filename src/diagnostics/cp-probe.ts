// Development check and timing of the merged Qwen3-TTS code-predictor graph (tools/export/merge_qwen3_tts_cp.py):
// the 5-layer predictor chained 15 times with the codec-embedding lookups and the top-k / temperature sampling inside
// one graph, against the engine's classic way (15 separate runs of the builder graph, a lookup between steps, sampling
// in JS). Same weights, same noise: the codes and the embedding sum must match; then the time per frame.
//
//   /cp-probe.html?model=Qwen__Qwen3-TTS-12Hz-0.6B-Base&frames=20&graph=frame.onnx
// Results in `window.__bench`.

import { getGpuDevice } from '../device/gpu';
import { bytesSource } from '../runtime/onnx/external-weights';
import { createGraphSession } from '../runtime/onnx/onnx-adapter';
import { loadOrt } from '../runtime/onnx/ort-env';

const params = new URLSearchParams(location.search);
const repo = params.get('model') ?? 'Qwen__Qwen3-TTS-12Hz-0.6B-Base';
const frames = Number(params.get('frames') ?? 20);
/** Size of the random hidden state (the talker's real ones are around 10-15 per value). */
const hiddenScale = Number(params.get('scale') ?? 2);
const mergedFile = params.get('graph') ?? 'frame.onnx';
/** Another build of the predictor (a directory beside code_predictor/ holding model.onnx, model.onnx.data, frame.onnx). */
const cpDir = params.get('cpdir') ?? 'code_predictor';
const base = `/__models/${repo}/local`;
const LAYERS = 5;
const KV_HEADS = 8;
const HEAD_DIM = 128;
const GROUP = 2048;
const STEPS = 15;
const WIDTH = 1024;
const VOCAB = 3072;
const TOPK = 50;
const TEMPERATURE = 0.9;

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
/** The graph's sampling noise: TEMPERATURE * Gumbel(0,1) from uniform noise. */
const gumbelOf = (u: Float32Array) => Float32Array.from(u, (x) => -TEMPERATURE * Math.log(-Math.log(x)));
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;

const loggedSteps = new Set<number>();
let seed = 987654321;
const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;

// ONNX Runtime lists node placements at verbose log level; the op types left on the CPU are counted (as tts-check.ts does).
const placement: Record<string, number> = {};
let onCpu = false;
if (params.has('placement')) {
  (globalThis as { __quickInfOrtVerbose?: boolean }).__quickInfOrtVerbose = true;
  for (const level of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    const original = console[level].bind(console);
    console[level] = (...args: unknown[]) => {
      const text = args.map(String).join(' ');
      if (text.includes('VerifyEachNodeIsAssignedToAnEp')) {
        const header = /Node\(s\) placed on \[(\w+)\]/.exec(text);
        const node = /AnEp\]\s{3}(\w+) \(/.exec(text);
        if (header) onCpu = header[1] === 'CPUExecutionProvider';
        else if (node && onCpu) placement[node[1]!] = (placement[node[1]!] ?? 0) + 1;
        return;
      }
      original(...args);
    };
  }
}

async function run(): Promise<void> {
  const ort = await loadOrt();
  const device = await getGpuDevice();
  const [graph, data, mergedGraph, embedGraph] = await Promise.all([
    fetchBytes(`${base}/${cpDir}/model.onnx`),
    fetchBytes(`${base}/${cpDir}/model.onnx.data`),
    fetchBytes(params.has('cpdir') ? `${base}/${cpDir}/frame.onnx` : `${base}/code_predictor_probe/${mergedFile}`),
    fetchBytes(`${base}/codec_embed_q4.onnx`),
  ]);
  const external = { 'model.onnx.data': bytesSource(data) };
  const baseline = await createGraphSession(bytesSource(graph), device, external, kv('present'));
  const merged = await createGraphSession(bytesSource(mergedGraph), device, external, []);
  const embedder = await createGraphSession(bytesSource(embedGraph), device, {}, []);
  if (params.has('placement')) log(`ops left on the CPU (all sessions): ${JSON.stringify(placement)}`);
  const withPast = merged.inputNames.includes('past_key_values.0.key');
  log(`sessions ready (merged inputs: ${merged.inputNames.length}, past inputs: ${withPast})`);
  const empty = () => new ort.Tensor('float32', new Float32Array(0), [1, KV_HEADS, 0, HEAD_DIM]);

  const lookup = async (id: number): Promise<Float32Array> => {
    const res = await embedder.run({ ids: new ort.Tensor('int64', BigInt64Array.of(BigInt(id)), [1, 1]) });
    const row = (res.embeds!.data as Float32Array).slice();
    res.embeds!.dispose();
    return row;
  };

  /** The engine's classic way: 15 runs, a lookup between steps, top-k / temperature sampling in JS (Gumbel-max with the given noise). */
  async function classic(hidden: Float32Array, c0: number, noise: Float32Array): Promise<{ codes: number[]; sum: Float32Array }> {
    let past: Record<string, InstanceType<typeof ort.Tensor>> = Object.fromEntries(kv('past_key_values').map((n) => [n, empty()]));
    let length = 0;
    const e0 = await lookup(c0);
    const sum = Float32Array.from(e0);
    let rows: Float32Array[] = [hidden, e0];
    const codes: number[] = [];
    for (let j = 0; j < STEPS; j++) {
      const n = rows.length;
      const embeds = new Float32Array(n * WIDTH);
      rows.forEach((r, i) => embeds.set(r, i * WIDTH));
      const res = await baseline.run({
        inputs_embeds: new ort.Tensor('float32', embeds, [1, n, WIDTH]),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(length + n).fill(1n), [1, length + n]),
        ...past,
      });
      for (const t of Object.values(past)) t.dispose();
      past = Object.fromEntries(kv('present').map((name, i) => [kv('past_key_values')[i]!, res[name]!]));
      length += n;
      const logits = res.logits!.data as Float32Array;
      const head = logits.subarray(logits.length - 15 * GROUP + j * GROUP, logits.length - 15 * GROUP + (j + 1) * GROUP);
      void (res.hidden_states!.data as Float32Array);
      res.logits!.dispose();
      res.hidden_states!.dispose();
      if (j < 2 && !loggedSteps.has(j)) {
        loggedSteps.add(j);
        const top = Array.from(head.keys()).sort((a, b) => head[b]! - head[a]!).slice(0, 4);
        log(`classic step ${j} (hidden scale ${hiddenScale}): top ${top.map((t) => `${t}:${head[t]!.toFixed(3)}`).join(' ')}`);
      }
      // Top-k as the graph does it: the logits at or above the k-th largest, Gumbel noise per token.
      const kth = [...head].sort((a, b) => b - a)[TOPK - 1]!;
      let best = -Infinity;
      let pick = 0;
      for (let token = 0; token < GROUP; token++) {
        if (head[token]! < kth) continue;
        const score = head[token]! / TEMPERATURE - Math.log(-Math.log(noise[j * GROUP + token]!));
        if (score > best) {
          best = score;
          pick = token;
        }
      }
      codes.push(pick);
      const row = await lookup(VOCAB + j * GROUP + pick);
      for (let i = 0; i < WIDTH; i++) sum[i]! += row[i]!;
      rows = [row];
    }
    for (const t of Object.values(past)) t.dispose();
    return { codes, sum };
  }

  async function mergedFrame(hidden: Float32Array, c0: number, noise: Float32Array): Promise<{ codes: number[]; sum: Float32Array }> {
    const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {
      hidden: new ort.Tensor('float32', hidden, [1, 1, WIDTH]),
      c0: new ort.Tensor('int64', BigInt64Array.of(BigInt(c0)), [1, 1]),
      gumbel: new ort.Tensor('float32', gumbelOf(noise), [STEPS, 1, GROUP]),
    };
    if (withPast) for (const n of kv('past_key_values')) feeds[n] = empty();
    const res = await merged.run(feeds);
    const codes = Array.from(res.codes!.data as Float32Array, Math.round);
    const sum = (res.embed_sum!.data as Float32Array).slice();
    res.codes!.dispose();
    res.embed_sum!.dispose();
    return { codes, sum };
  }

  const trial = () => ({
    hidden: Float32Array.from({ length: WIDTH }, () => (random() - 0.5) * hiddenScale),
    c0: Math.floor(random() * 2048),
    noise: Float32Array.from({ length: STEPS * GROUP }, () => 1e-6 + random() * (1 - 2e-6)),
  });

  // Correctness: the same codes and the same embedding sum.
  let mismatches = 0;
  let maxSumDiff = 0;
  const trials = 8;
  for (let t = 0; t < trials; t++) {
    const { hidden, c0, noise } = trial();
    const a = await classic(hidden, c0, noise);
    const b = await mergedFrame(hidden, c0, noise);
    const same = a.codes.every((c, i) => c === b.codes[i]);
    if (t < 3) log(`trial ${t}: merged codes ${b.codes.join(',')} sum[0..3] ${[...b.sum.subarray(0, 3)].map((v) => v.toFixed(4)).join(',')}`);
    if (!same) {
      mismatches++;
      log(`trial ${t}: codes differ\n  classic ${a.codes.join(',')}\n  merged  ${b.codes.join(',')}`);
    }
    for (let i = 0; i < WIDTH; i++) maxSumDiff = Math.max(maxSumDiff, Math.abs(a.sum[i]! - b.sum[i]!));
  }
  log(`${trials} trials: ${trials - mismatches} with identical codes; max |sum difference| ${maxSumDiff.toExponential(2)}`);

  // Timing.
  for (let i = 0; i < 3; i++) {
    const { hidden, c0, noise } = trial();
    await classic(hidden, c0, noise);
    await mergedFrame(hidden, c0, noise);
  }
  const times = { classic: [] as number[], merged: [] as number[] };
  for (let i = 0; i < frames; i++) {
    const { hidden, c0, noise } = trial();
    let t = performance.now();
    await classic(hidden, c0, noise);
    times.classic.push(performance.now() - t);
    t = performance.now();
    await mergedFrame(hidden, c0, noise);
    times.merged.push(performance.now() - t);
  }
  const summary = {
    classicMedianMs: +median(times.classic).toFixed(1),
    mergedMedianMs: +median(times.merged).toFixed(1),
    classicBestMs: +Math.min(...times.classic).toFixed(1),
    mergedBestMs: +Math.min(...times.merged).toFixed(1),
  };
  state.results = { ...summary, trials, mismatches, maxSumDiff };
  log(`per frame, median of ${frames}: classic (15 runs + 15 lookups + JS sampling) ${summary.classicMedianMs} ms, merged (1 run) ${summary.mergedMedianMs} ms; best ${summary.classicBestMs} vs ${summary.mergedBestMs} ms`);
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
