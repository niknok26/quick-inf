import { describe, expect, it } from 'vitest';
import { mixToMono } from './voice-audio';

describe('mixToMono', () => {
  it('averages the channels', () => {
    expect([...mixToMono([Float32Array.of(1, -1, 0.5), Float32Array.of(0, 1, 0.5)])]).toEqual([0.5, 0, 0.5]);
  });
  it('copies a single channel instead of aliasing it', () => {
    const channel = Float32Array.of(0.25, 0.5);
    const mono = mixToMono([channel]);
    mono[0] = 9;
    expect(channel[0]).toBe(0.25);
  });
});
