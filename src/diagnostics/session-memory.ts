// Development: creates the Qwen3-TTS graphs one at a time on the GPU, pausing after each, so a script can read the
// device's memory between steps (on a phone: `adb shell dumpsys meminfo`, the GPU process is
// com.android.chrome:privileged_process0). The weights are read whole into memory here, which the engine avoids.
//
//   /session-memory.html?model=Qwen__Qwen3-TTS-12Hz-0.6B-Base&pause=6000
// `window.__bench.log` gets a line "created <name>" after each; the page waits `pause` ms before the next.

import { getGpuDevice } from '../device/gpu';
import { bytesSource } from '../runtime/onnx/external-weights';
import { createGraphSession } from '../runtime/onnx/onnx-adapter';

const params = new URLSearchParams(location.search);
const repo = params.get('model') ?? 'Qwen__Qwen3-TTS-12Hz-0.6B-Base';
const pause = Number(params.get('pause') ?? 6000);
const base = `/__models/${repo}/local`;
const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};
const fetchBytes = async (path: string) => new Uint8Array(await (await fetch(`${base}/${path}`)).arrayBuffer());
const kv = (layers: number) => Array.from({ length: layers }, (_, i) => [`present.${i}.key`, `present.${i}.value`]).flat();

const GRAPHS: { name: string; graph: string; data?: string; gpuOutputs?: string[] }[] = [
  { name: 'speaker', graph: 'speaker_q4.onnx' },
  { name: 'codecEmbed', graph: 'codec_embed_q4.onnx' },
  { name: 'textEmbed', graph: 'text_embed_q4.onnx' },
  { name: 'encoder', graph: 'tok_encoder_q4.onnx' },
  { name: 'decoder', graph: 'tok_decoder_q4c8.onnx' },
  { name: 'cpFrame', graph: 'code_predictor/frame.onnx', data: 'code_predictor/model.onnx.data' },
  { name: 'talker', graph: 'talker/model.onnx', data: 'talker/model.onnx.data', gpuOutputs: kv(28) },
];

async function run(): Promise<void> {
  const device = await getGpuDevice();
  log('device ready');
  await new Promise((r) => setTimeout(r, pause));
  for (const g of GRAPHS) {
    const graph = await fetchBytes(g.graph);
    const data: Record<string, ReturnType<typeof bytesSource>> = {};
    if (g.data) data['model.onnx.data'] = bytesSource(await fetchBytes(g.data));
    await createGraphSession(bytesSource(graph), device, data, g.gpuOutputs ?? []);
    log(`created ${g.name}`);
    await new Promise((r) => setTimeout(r, pause));
  }
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error.slice(0, 400)}`);
  })
  .finally(() => {
    state.done = true;
  });
