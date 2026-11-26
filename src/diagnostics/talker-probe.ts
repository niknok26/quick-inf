// Development: does the Qwen3-TTS main model (talker) compute the same on this device's GPU as on another? Runs a
// prefill of fixed pseudo-random rows and a few single-row steps, and reports the top logits and a checksum of each
// step, so two devices' pages can be compared. Results in `window.__bench`.
//
//   /talker-probe.html?model=Qwen__Qwen3-TTS-12Hz-0.6B-Base&rows=16

import { getGpuDevice } from '../device/gpu';
import { bytesSource } from '../runtime/onnx/external-weights';
import { createGraphSession } from '../runtime/onnx/onnx-adapter';
import { loadOrt } from '../runtime/onnx/ort-env';

const params = new URLSearchParams(location.search);
const repo = params.get('model') ?? 'Qwen__Qwen3-TTS-12Hz-0.6B-Base';
const rows = Number(params.get('rows') ?? 16);
const WIDTH = Number(params.get('width') ?? 1024);
const LAYERS = 28;
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};
const names = (prefix: string) => Array.from({ length: LAYERS }, (_, i) => [`${prefix}.${i}.key`, `${prefix}.${i}.value`]).flat();
const fetchBytes = async (path: string) => new Uint8Array(await (await fetch(`/__models/${repo}/local/${path}`)).arrayBuffer());

async function run(): Promise<void> {
  const ort = await loadOrt();
  const device = await getGpuDevice();
  const session = await createGraphSession(bytesSource(await fetchBytes('talker/model.onnx')), device, { 'model.onnx.data': bytesSource(await fetchBytes('talker/model.onnx.data')) }, names('present'));
  log('talker ready');
  let seed = 424242;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5) * 0.4;
  const pastNames = names('past_key_values');
  const presentNames = names('present');
  let past: Record<string, InstanceType<typeof ort.Tensor>> = Object.fromEntries(pastNames.map((n) => [n, new ort.Tensor('float32', new Float32Array(0), [1, 8, 0, 128])]));
  let length = 0;
  const step = async (n: number, label: string) => {
    const embeds = Float32Array.from({ length: n * WIDTH }, random);
    const res = await session.run({
      inputs_embeds: new ort.Tensor('float32', embeds, [1, n, WIDTH]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(length + n).fill(1n), [1, length + n]),
      ...past,
    });
    for (const t of Object.values(past)) t.dispose();
    past = Object.fromEntries(presentNames.map((name, i) => [pastNames[i]!, res[name]!]));
    length += n;
    const logits = res.logits!.data as Float32Array;
    const last = logits.subarray(logits.length - 3072);
    const top = Array.from(last.keys()).sort((a, b) => last[b]! - last[a]!).slice(0, 5);
    let sum = 0;
    let finite = true;
    for (const v of last) {
      sum += v;
      if (!Number.isFinite(v)) finite = false;
    }
    const hidden = res.hidden_states!.data as Float32Array;
    log(`${label}: top ${top.map((t) => `${t}:${last[t]!.toFixed(3)}`).join(' ')} sum ${sum.toFixed(2)} finite ${finite} eos(2150) ${last[2150]!.toFixed(3)} hidden[0..3] ${[...hidden.subarray(hidden.length - WIDTH, hidden.length - WIDTH + 3)].map((v) => v.toFixed(4)).join(',')}`);
    res.logits!.dispose();
    res.hidden_states!.dispose();
  };
  await step(rows, `prefill ${rows} rows`);
  for (let i = 1; i <= 3; i++) await step(1, `step ${i}`);
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error.slice(0, 400)}`);
  })
  .finally(() => {
    state.done = true;
  });
