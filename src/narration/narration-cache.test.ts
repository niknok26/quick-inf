import { describe, expect, it } from 'vitest';
import { selectEvictions, type NarrationMeta } from './narration-cache';

const meta = (key: string, size: number, lastAccessed: number): NarrationMeta => ({ key, size, lastAccessed });

describe('selectEvictions', () => {
  it('keeps everything while under the cap', () => {
    expect(selectEvictions([meta('a', 10, 1), meta('b', 10, 2)], 20)).toEqual([]);
  });
  it('drops the least recently used first until the cache fits', () => {
    expect(selectEvictions([meta('a', 10, 3), meta('b', 10, 1), meta('c', 10, 2)], 15)).toEqual(['b', 'c']);
  });
  it('never drops the clip just made, even if it is the oldest or alone over the cap', () => {
    expect(selectEvictions([meta('new', 30, 1), meta('old', 10, 2)], 20, 'new')).toEqual(['old']);
    expect(selectEvictions([meta('new', 30, 1)], 20, 'new')).toEqual([]);
  });
});
