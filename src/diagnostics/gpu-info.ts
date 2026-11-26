// Development: what this browser's WebGPU offers (adapter, features, limits), and whether a buffer as large as the
// biggest Qwen3-TTS weight file can be created. Results in `window.__bench` and on the page.
//
//   /gpu-info.html

const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

async function run(): Promise<void> {
  log(`secure context: ${window.isSecureContext}; userAgent: ${navigator.userAgent}`);
  log(`deviceMemory: ${(navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 'n/a'} GB; cores: ${navigator.hardwareConcurrency}`);
  if (!navigator.gpu) throw new Error('navigator.gpu is not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const info = adapter.info;
  log(`adapter: ${info.vendor} / ${info.architecture} / ${info.device} / ${info.description}`);
  log(`features: ${[...adapter.features].sort().join(', ')}`);
  const wanted = ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxStorageBuffersPerShaderStage', 'maxBindGroups'] as const;
  const limits = adapter.limits as unknown as Record<string, number>;
  log(`limits: ${wanted.map((k) => `${k}=${limits[k]}`).join(', ')}`);
  state.results = { adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description }, limits: Object.fromEntries(wanted.map((k) => [k, limits[k]])), features: [...adapter.features] };
  // Can a buffer the size of the largest weight file (516 MB talker) be made and written?
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: limits.maxBufferSize!, maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize! } });
  for (const mb of [256, 516, 1024]) {
    try {
      device.pushErrorScope('out-of-memory');
      const buffer = device.createBuffer({ size: mb * 1024 * 1024, usage: 0x80 | 0x08 });
      const error = await device.popErrorScope();
      log(`${mb} MB storage buffer: ${error ? 'FAILED ' + error.message : 'ok'}`);
      buffer.destroy();
    } catch (err) {
      log(`${mb} MB storage buffer: threw ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
