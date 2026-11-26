// Development harness for deck narration (src/narration): opens a small deck, narrates it in the first saved
// reference voice (made in a speech tab's Settings) and shows each slide's state as it goes. Reload to see the cache
// make every slide ready at once. `window.__narrator` is the Narrator (try `setCurrent(3)`); `window.__bench` has the log.
//
//   /deck-narration.html?model=qwen3-tts-0.6b-base[&current=0][&slides=<JSON [{"id","text"}]>]

import { getRuntimeClient } from '../runtime/client';
import { Narrator, type NarrationSlide } from '../narration/narrator';
import { listVoices } from '../store/voices';

const params = new URLSearchParams(location.search);
const modelId = params.get('model') ?? 'qwen3-tts-0.6b-base';
const SLIDES: NarrationSlide[] = params.has('slides')
  ? (JSON.parse(params.get('slides')!) as NarrationSlide[])
  : [
      { id: 's1', text: 'Welcome to this walkthrough. Today we are going to deploy a serverless application on AWS, step by step.' },
      { id: 's2', text: 'AWS Lambda lets you run code without provisioning or managing servers, and you only pay for the compute time you use.' },
      { id: 's3', text: 'First, we create the function. Then we attach a role that lets it read from the bucket.' },
      { id: 's4', text: 'That is all it takes. Thanks for watching.' },
    ];

const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
const w = window as unknown as { __bench: typeof state; __narrator?: Narrator };
w.__bench = state;
const out = document.getElementById('out')!;
const t0 = performance.now();
const log = (line: string) => {
  const text = `${((performance.now() - t0) / 1000).toFixed(1)} s  ${line}`;
  state.log.push(text);
};

async function run(): Promise<void> {
  const voices = await listVoices();
  const voice = voices.sort((a, b) => a.createdAt - b.createdAt)[0];
  if (!voice) throw new Error('No saved reference voice: add one in a speech tab\u2019s Settings first.');
  log(`voice "${voice.name}", ${SLIDES.length} slides, model ${modelId}`);
  const narrator = new Narrator({ client: getRuntimeClient(), modelId, voice });
  w.__narrator = narrator;
  const last = new Map<string, string>();
  const render = () => {
    const lines = SLIDES.map((s) => {
      const st = narrator.status(s.id);
      const text = st ? `${st.state}${st.piece ? ` (${st.piece.index}/${st.piece.of})` : ''}${st.seconds ? ` ${st.seconds.toFixed(1)} s` : ''}${st.error ? ` ${st.error}` : ''}` : '…';
      if (last.get(s.id) !== text) {
        last.set(s.id, text);
        log(`${s.id}: ${text}`);
      }
      return `${s.id}  ${text}`;
    });
    out.textContent = `${lines.join('\n')}\n\n${state.log.join('\n')}`;
  };
  narrator.subscribe(render);
  await narrator.start(SLIDES, Number(params.get('current') ?? 0));
  render();
  for (const s of SLIDES) await narrator.audio(s.id);
  state.results.statuses = Object.fromEntries(SLIDES.map((s) => [s.id, narrator.status(s.id)]));
  log('all slides ready');
  render();
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
    out.textContent = state.log.join('\n');
  })
  .finally(() => {
    state.done = true;
  });
