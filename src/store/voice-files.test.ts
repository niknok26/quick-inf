import { describe, expect, it } from 'vitest';
import { clipBytes, clipSamples, sha256Hex, verifyClip } from './voice-files';

const meta = (audio: Float32Array) => ({ name: 'Test', samples: audio.length, sha256: sha256Hex(clipBytes(audio)) });

describe('voice clip files', () => {
  it('stores samples as little-endian float32 and reads them back exactly', () => {
    const audio = Float32Array.of(0, 0.5, -0.25, 1, -1, 1e-7);
    const bytes = clipBytes(audio);
    expect(bytes.byteLength).toBe(24);
    expect([...bytes.subarray(4, 8)]).toEqual([0, 0, 0, 0x3f]); // 0.5
    expect([...clipSamples(bytes)]).toEqual([...audio]);
  });
  it('accepts the clip the record describes', () => {
    const audio = Float32Array.of(0.1, 0.2, 0.3);
    expect(() => verifyClip(meta(audio), clipBytes(audio))).not.toThrow();
  });
  it('refuses a clip that was changed, truncated or swapped', () => {
    const audio = Float32Array.of(0.1, 0.2, 0.3);
    const m = meta(audio);
    const changed = clipBytes(audio);
    changed[5] = changed[5]! ^ 1;
    expect(() => verifyClip(m, changed)).toThrow(/integrity/);
    expect(() => verifyClip(m, clipBytes(audio).subarray(0, 8))).toThrow(/integrity/);
    expect(() => verifyClip(m, clipBytes(Float32Array.of(0.1, 0.2, 0.4)))).toThrow(/integrity/);
  });
  it('handles an empty clip', () => {
    expect(clipBytes(new Float32Array(0)).byteLength).toBe(0);
  });
});
