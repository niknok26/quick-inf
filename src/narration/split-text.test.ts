import { describe, expect, it } from 'vitest';
import { splitForSpeech } from './split-text';

describe('splitForSpeech', () => {
  it('keeps short text in one piece', () => {
    expect(splitForSpeech('Hello there. How are you?')).toEqual(['Hello there. How are you?']);
  });
  it('splits between whole sentences at the limit', () => {
    expect(splitForSpeech('One two three. Four five six. Seven eight nine.', 30)).toEqual(['One two three. Four five six.', 'Seven eight nine.']);
  });
  it('keeps an over-long single sentence whole', () => {
    const long = `${'word '.repeat(30).trim()}.`;
    expect(splitForSpeech(long, 20)).toEqual([long]);
  });
  it('keeps text without final punctuation', () => {
    expect(splitForSpeech('No punctuation at all')).toEqual(['No punctuation at all']);
  });
  it('returns nothing for blank text', () => {
    expect(splitForSpeech('   ')).toEqual([]);
  });
});
