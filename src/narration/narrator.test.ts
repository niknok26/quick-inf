import { describe, expect, it } from 'vitest';
import { RuntimeError } from '../runtime/contract';
import type { NarrationRow } from './narration-cache';
import { Narrator, type NarrationStore, type NarratorOptions } from './narrator';

const voice = { id: 'v', createdAt: 1, audio: new Float32Array(10), transcript: 'ref' };

function setup(behaviour: (call: number, text: string) => 'ok' | 'fail') {
  const rows = new Map<string, NarrationRow>();
  const cache: NarrationStore = { get: async (k) => rows.get(k), put: async (r) => void rows.set(r.key, r) };
  const calls: string[] = [];
  const client = {
    load: async () => ({}) as never,
    generateSpeech(request: { text: string }, onProgress: (p: { audio: Float32Array; chunk: number; chunks: number }) => void) {
      calls.push(request.text);
      const call = calls.length;
      const outcome = behaviour(call, request.text);
      // Each call makes audio of a different length, so takes can be told apart.
      const audio = new Float32Array(100 * call);
      const result =
        outcome === 'ok'
          ? Promise.resolve().then(() => {
              onProgress({ audio, chunk: 1, chunks: 1 });
              return { sampleRate: 24000, samples: audio.length, timings: { firstAudioMs: 0, totalMs: 0 } };
            })
          : Promise.reject(new RuntimeError('generation-failed', 'boom'));
      return { result, cancel: () => {} };
    },
  } as unknown as NarratorOptions['client'];
  const narrator = new Narrator({ client, modelId: 'm', voice, cache });
  return { narrator, calls, rows };
}

const settled = (narrator: Narrator, id: string, state: string) =>
  new Promise<void>((resolve) => {
    const check = () => narrator.status(id)?.state === state && (off(), resolve());
    const off = narrator.subscribe(check);
    check();
  });

describe('Narrator', () => {
  it('speaks the slide the presenter is on first, then the rest, and caches each', async () => {
    const { narrator, calls, rows } = setup(() => 'ok');
    await narrator.start([{ id: 'a', text: 'One.' }, { id: 'b', text: 'Two.' }, { id: 'c', text: 'Three.' }], 1);
    await settled(narrator, 'a', 'cached');
    expect(calls).toEqual(['Two.', 'Three.', 'One.']);
    expect(rows.size).toBe(3);
  });

  it('regenerate makes a new take, keeping the old clip playable until the new one replaces it', async () => {
    const { narrator, calls } = setup(() => 'ok');
    await narrator.start([{ id: 'a', text: 'One.' }], 0);
    const first = await narrator.audio('a');
    narrator.regenerate('a');
    expect(narrator.status('a')?.state).toBe('queued');
    expect((await narrator.audio('a')).size).toBe(first.size); // still the old take while the new one is made
    await settled(narrator, 'a', 'cached');
    expect(calls).toEqual(['One.', 'One.']);
    expect((await narrator.audio('a')).size).toBeGreaterThan(first.size); // the second call made longer audio
  });

  it('keeps the old take when a new one fails, and says why', async () => {
    const { narrator } = setup((call) => (call === 1 ? 'ok' : 'fail'));
    await narrator.start([{ id: 'a', text: 'One.' }], 0);
    const first = await narrator.audio('a');
    narrator.regenerate('a');
    await new Promise<void>((resolve) => {
      const off = narrator.subscribe(() => narrator.status('a')?.error && (off(), resolve()));
    });
    expect(narrator.status('a')).toMatchObject({ state: 'cached', error: 'boom' });
    expect((await narrator.audio('a')).size).toBe(first.size);
  });

  it('ignores regenerate for a slide being made right now', async () => {
    const { narrator, calls } = setup(() => 'ok');
    await narrator.start([{ id: 'a', text: 'One.' }], 0);
    narrator.regenerate('a'); // state is queued/generating: nothing cached yet, nothing to replace
    await settled(narrator, 'a', 'cached');
    expect(calls).toEqual(['One.']);
  });
});
