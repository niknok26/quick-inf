import { describe, expect, it } from 'vitest';
import { validateVoice, isRecommendedDuration } from './voice-validation';

describe('validateVoice', () => {
  it('returns errors when name and transcript are empty', () => {
    const errors = validateVoice('', '', null);
    expect(errors.name).toBe('Name is required');
    expect(errors.transcript).toBe('Transcript is required');
    expect(errors.audioDuration).toBe('No audio clip selected');
  });

  it('returns errors for too-short clip', () => {
    const errors = validateVoice('Test Voice', 'Hello world', 2);
    expect(errors.audioDuration).toMatch(/at least 3 seconds/);
  });

  it('returns errors for too-long clip', () => {
    const errors = validateVoice('Test Voice', 'Hello world', 35);
    expect(errors.audioDuration).toMatch(/at most 30 seconds/);
  });

  it('returns no errors for valid input', () => {
    const errors = validateVoice('My Voice', 'This is a test sentence.', 10);
    expect(errors).toEqual({});
  });

  it('accepts boundary values (3s and 30s)', () => {
    expect(validateVoice('V', 'T', 3)).toEqual({});
    expect(validateVoice('V', 'T', 30)).toEqual({});
  });

  it('rejects values just outside boundaries', () => {
    expect(validateVoice('V', 'T', 2.9).audioDuration).toBeDefined();
    expect(validateVoice('V', 'T', 30.1).audioDuration).toBeDefined();
  });

  it('trims name but not transcript for validation', () => {
    // Name with only whitespace is invalid
    const errors = validateVoice('   ', 'Some text', 10);
    expect(errors.name).toBe('Name is required');
  });
});

describe('isRecommendedDuration', () => {
  it('returns true for 8–15 seconds inclusive', () => {
    expect(isRecommendedDuration(8)).toBe(true);
    expect(isRecommendedDuration(10)).toBe(true);
    expect(isRecommendedDuration(15)).toBe(true);
  });

  it('returns false outside recommended range', () => {
    expect(isRecommendedDuration(7)).toBe(false);
    expect(isRecommendedDuration(16)).toBe(false);
    expect(isRecommendedDuration(1)).toBe(false);
  });
});
