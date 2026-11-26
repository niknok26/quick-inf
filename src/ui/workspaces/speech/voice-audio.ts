// Turning a recorded or chosen audio file into a reference clip: mono float32 at the cloning model's 24 kHz.

import { VOICE_SAMPLE_RATE } from '../../../store/voice-files';

/** Averages the channels of a clip into one. */
export function mixToMono(channels: Float32Array[]): Float32Array {
  const first = channels[0];
  if (!first) return new Float32Array(0);
  if (channels.length === 1) return first.slice();
  const mono = new Float32Array(first.length);
  for (const channel of channels) for (let i = 0; i < mono.length; i++) mono[i]! += channel[i]!;
  for (let i = 0; i < mono.length; i++) mono[i]! /= channels.length;
  return mono;
}

/** Decodes any audio the browser can play; the decoder resamples to the context's rate. */
export async function decodeVoiceBlob(blob: Blob): Promise<Float32Array> {
  const decoded = await new OfflineAudioContext(1, 1, VOICE_SAMPLE_RATE).decodeAudioData(await blob.arrayBuffer());
  return mixToMono(Array.from({ length: decoded.numberOfChannels }, (_, c) => decoded.getChannelData(c)));
}
