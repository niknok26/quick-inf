// Long narration is spoken in pieces: one request per stretch of whole sentences, so a piece never grows past what the
// model handles well and progress shows between pieces.

/** Splits `text` into pieces of whole sentences, each at most `limit` characters (a longer single sentence stays whole). */
export function splitForSpeech(text: string, limit = 350): string[] {
  const sentences = text.trim().match(/[^.!?…]+(?:[.!?…]+["')\]]*|$)\s*/g)?.map((s) => s.trim()).filter(Boolean) ?? [];
  const pieces: string[] = [];
  let current = '';
  for (const sentence of sentences) {
    if (current && current.length + 1 + sentence.length > limit) {
      pieces.push(current);
      current = sentence;
    } else {
      current = current ? `${current} ${sentence}` : sentence;
    }
  }
  if (current) pieces.push(current);
  return pieces;
}
