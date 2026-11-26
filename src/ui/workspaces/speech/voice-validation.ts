// Voice clip validation helpers. Pure functions, easy to test.

export interface ValidationErrors {
  name?: string;
  transcript?: string;
  audioDuration?: string;
}

/** Validate a new voice record before saving. Returns error map (empty = valid). */
export function validateVoice(
  name: string,
  transcript: string,
  durationSeconds: number | null,
): ValidationErrors {
  const errors: ValidationErrors = {};
  if (!name.trim()) errors.name = 'Name is required';
  if (!transcript.trim()) errors.transcript = 'Transcript is required';
  if (durationSeconds === null) errors.audioDuration = 'No audio clip selected';
  else if (durationSeconds < 3) errors.audioDuration = `Clip must be at least 3 seconds (${durationSeconds.toFixed(1)} s)`;
  else if (durationSeconds > 30) errors.audioDuration = `Clip must be at most 30 seconds (${durationSeconds.toFixed(1)} s)`;
  return errors;
}

/** Check if a duration is in the recommended range (8–15 s). */
export function isRecommendedDuration(seconds: number): boolean {
  return seconds >= 8 && seconds <= 15;
}
