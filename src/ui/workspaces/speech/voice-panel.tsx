// Reference voices for cloning models (Qwen3-TTS Base): pick a saved voice, or make one from a recording or an audio
// file plus what it says. Voices stay on the device (see src/store/voice-files.ts). A protected voice (the clip the app
// was given) can be selected and deleted, but has no player and its clip is never shown or offered for download.

import { useEffect, useRef, useState } from 'react';
import { VOICE_SAMPLE_RATE, type VoiceMeta } from '../../../store/voice-files';
import { decodeVoiceBlob } from './voice-audio';
import { voiceStore } from './voice-store';
import { isRecommendedDuration, validateVoice } from './voice-validation';
import { encodeWav } from './wav';

interface VoicePanelProps {
  voices: readonly VoiceMeta[];
  selectedId: string | undefined;
  onSelect: (id: string) => void;
}

export function VoicePanel({ voices, selectedId, onSelect }: VoicePanelProps) {
  const [name, setName] = useState('');
  const [transcript, setTranscript] = useState('');
  const [clip, setClip] = useState<Float32Array | null>(null);
  const [recordedSeconds, setRecordedSeconds] = useState<number | null>(null);
  const [notice, setNotice] = useState('');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const selected = voices.find((v) => v.id === selectedId);

  const seconds = clip ? clip.length / VOICE_SAMPLE_RATE : null;
  const problems = Object.values(validateVoice(name, transcript, seconds)).filter(Boolean);

  // The selected voice's clip as a playable URL (loaded from the device's file storage; never for a protected voice).
  const previewId = selected && !selected.protected ? selected.id : null;
  useEffect(() => {
    if (!previewId) return;
    let alive = true;
    let url: string | null = null;
    voiceStore
      .load(previewId)
      .then((voice) => {
        if (!alive) return;
        url = URL.createObjectURL(encodeWav([voice.audio], voice.sampleRate));
        setPreviewUrl(url);
      })
      .catch((err: unknown) => alive && setNotice(err instanceof Error ? err.message : String(err)));
    return () => {
      alive = false;
      setPreviewUrl(null);
      if (url) URL.revokeObjectURL(url);
    };
  }, [previewId]);
  // Leaving mid-recording releases the microphone.
  useEffect(
    () => () => {
      if (timer.current) clearInterval(timer.current);
      if (recorder.current?.state === 'recording') recorder.current.stop();
    },
    [],
  );

  async function useBlob(blob: Blob, what: string): Promise<void> {
    try {
      setClip(await decodeVoiceBlob(blob));
      setNotice('');
    } catch {
      setClip(null);
      setNotice(`Could not read ${what}.`);
    }
  }

  async function startRecording(): Promise<void> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data);
      rec.onstop = () => {
        for (const track of stream.getTracks()) track.stop();
        void useBlob(new Blob(chunks, { type: rec.mimeType }), 'the recording');
      };
      rec.start();
      recorder.current = rec;
      setClip(null);
      setNotice('');
      setRecordedSeconds(0);
      timer.current = setInterval(() => setRecordedSeconds((s) => (s ?? 0) + 1), 1000);
    } catch {
      setNotice('The microphone is not available.');
    }
  }

  function stopRecording(): void {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    setRecordedSeconds(null);
    if (recorder.current?.state === 'recording') recorder.current.stop();
  }

  async function save(): Promise<void> {
    if (!clip || problems.length > 0) return;
    try {
      const meta = await voiceStore.create(name.trim(), transcript.trim(), clip);
      setName('');
      setTranscript('');
      setClip(null);
      setNotice('');
      onSelect(meta.id);
    } catch {
      setNotice('Could not save the voice on this device.');
    }
  }

  async function remove(voice: VoiceMeta): Promise<void> {
    if (!confirm(`Delete the voice “${voice.name}”?`)) return;
    await voiceStore.remove(voice.id);
    const next = voices.find((v) => v.id !== voice.id);
    onSelect(next?.id ?? '');
  }

  return (
    <div className="voice-panel">
      <h3 className="voice-panel__title">Reference voice</h3>
      {voices.length > 0 && (
        <label className="image-setting image-setting--select">
          <span className="image-setting__label">Saved voice</span>
          <select value={selected?.id ?? ''} onChange={(e) => onSelect(e.target.value)}>
            {voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {selected && (
        <div className="voice-panel__preview">
          {previewUrl && <audio controls controlsList="nodownload" src={previewUrl} />}
          {!selected.protected && <span className="muted">“{selected.transcript}”</span>}
          {selected.protected && <span className="muted">The reference clip provided with the app. It can be used for narration or deleted.</span>}
          <button type="button" className="button button--ghost button--small voice-panel__delete" onClick={() => void remove(selected)}>
            Delete this voice
          </button>
        </div>
      )}

      <fieldset className="voice-panel__form">
        <legend>New voice</legend>
        <label className="image-setting image-setting--text">
          <span className="image-setting__label">Name</span>
          <input type="text" value={name} maxLength={50} placeholder="e.g. My voice" onChange={(e) => setName(e.target.value)} />
        </label>
        <div className="voice-panel__audio-source">
          <label className="button button--ghost button--small">
            Choose an audio file…
            <input
              type="file"
              accept="audio/*"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void useBlob(file, file.name);
                e.target.value = '';
              }}
            />
          </label>
          {recordedSeconds === null ? (
            <button type="button" className="button button--ghost button--small" onClick={() => void startRecording()}>
              Record
            </button>
          ) : (
            <button type="button" className="button button--ghost button--small" onClick={stopRecording}>
              Stop ({recordedSeconds} s)
            </button>
          )}
        </div>
        {seconds !== null && (
          <p className={isRecommendedDuration(seconds) ? 'muted' : 'voice-panel__warn'}>
            Clip: {seconds.toFixed(1)} s{isRecommendedDuration(seconds) ? '' : ' (8–15 s works best; 3–30 s allowed)'}
          </p>
        )}
        <label className="image-setting image-setting--text">
          <span className="image-setting__label">What the clip says, word for word</span>
          <textarea rows={3} value={transcript} placeholder="Type the exact words spoken in the clip" onChange={(e) => setTranscript(e.target.value)} />
        </label>
        {notice && <p className="voice-panel__warn">{notice}</p>}
        {clip && problems.length > 0 && <p className="voice-panel__warn">{problems[0]}</p>}
        <button type="button" className="button button--small" disabled={!clip || problems.length > 0} onClick={() => void save()}>
          Save voice
        </button>
      </fieldset>
    </div>
  );
}
