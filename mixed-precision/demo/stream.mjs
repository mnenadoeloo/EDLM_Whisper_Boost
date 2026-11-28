// The microphone and timed replay use the same 16 kHz PCM frames and scheduler.
export const RATE = 16000;
export const FRAME = 1600;
const UPDATE = 1.5 * RATE, MAX_SEGMENT = 24 * RATE, QUIET = 0.8 * RATE;

export function wavBytes(frames) {
  const count = frames.reduce((n, frame) => n + frame.length, 0);
  const bytes = new Uint8Array(44 + 2 * count), view = new DataView(bytes.buffer);
  for (const [offset, tag] of [[0, 'RIFF'], [8, 'WAVEfmt '], [36, 'data']]) {
    bytes.set(new TextEncoder().encode(tag), offset);
  }
  view.setUint32(4, bytes.length - 8, true);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, RATE, true);
  view.setUint32(28, RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  view.setUint32(40, count * 2, true);
  let offset = 44;
  for (const frame of frames) for (const sample of frame) {
    view.setInt16(offset, sample, true);
    offset += 2;
  }
  return bytes;
}

export async function infer(base, options, frames) {
  const params = new URLSearchParams(options);
  const response = await fetch(base + '/api/transcribe?' + params, {
    method: 'POST', headers: {'Content-Type': 'audio/wav'}, body: wavBytes(frames),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Ошибка распознавания');
  return data;
}

export async function replayPcm(pcm, session, stopped = () => false) {
  const start = performance.now();
  for (let p = 0; p < pcm.length && !stopped(); p += FRAME) {
    const end = Math.min(p + FRAME, pcm.length);
    await new Promise(resolve => setTimeout(resolve, Math.max(0, start + 1000 * end / RATE - performance.now())));
    if (stopped() || session.closed) break;
    session.push(pcm.subarray(p, end));
  }
}

const normalize = word => word.toLocaleLowerCase().replace(/[^\p{L}\p{N}]/gu, '');

// Align the immutable prefix to a newly decoded hypothesis. A punctuation change,
// insertion or deletion before the boundary must not duplicate a confirmed word.
function suffixAfter(stable, words) {
  const n = stable.length;
  if (!n) return words;
  const limit = Math.min(words.length, 2 * n);
  let row = Array.from({length: limit + 1}, (_, i) => i);
  for (let i = 0; i < n; i++) {
    const next = [i + 1];
    for (let j = 1; j <= limit; j++) {
      next[j] = Math.min(row[j] + 1, next[j - 1] + 1,
        row[j - 1] + Number(normalize(stable[i]) !== normalize(words[j - 1])));
    }
    row = next;
  }
  let end = 0;
  for (let j = 1; j <= limit; j++) {
    if (row[j] < row[end] || (row[j] === row[end] && Math.abs(j - n) < Math.abs(end - n))) end = j;
  }
  return words.slice(end);
}

export class Agreement {
  stable = [];
  draft = [];
  update(text, final) {
    const words = text.trim() ? text.trim().split(/\s+/) : [];
    const fresh = suffixAfter(this.stable, words);
    let matched = 0;
    while (matched < Math.min(this.draft.length, fresh.length)
      && normalize(this.draft[matched]) === normalize(fresh[matched])) matched++;
    if (final) matched = fresh.length;
    this.stable.push(...fresh.slice(0, matched));
    this.draft = fresh.slice(matched);
  }
}

function segment() {
  return {frames: [], count: 0, sent: 0, quiet: 0, sealed: false, agreement: new Agreement()};
}

export class LiveSession {
  // One decoder at a time. Completed utterances are preserved. Only the newest
  // prefix of the open utterance is scheduled after a slow request.
  constructor(decode, onUpdate, onError) {
    this.decode = decode;
    this.onUpdate = onUpdate;
    this.onError = onError;
    this.current = segment();
    this.pending = [];
    this.history = [];
    this.preroll = [];
    this.recentRms = 0;
    this.recorded = 0;
    this.running = null;
    this.closed = false;
  }

  push(frame) {
    if (this.closed || !frame.length) return;
    this.recorded += frame.length;
    const rms = Math.sqrt(frame.reduce((sum, v) => sum + v * v, 0) / frame.length);
    this.recentRms = Math.max(rms, this.recentRms * 0.97);
    const voiced = rms > Math.max(8, this.recentRms * 0.025);
    const s = this.current;
    if (!s.count && !voiced) {
      this.preroll.push(frame);
      if (this.preroll.length > 2) this.preroll.shift();
      return;
    }
    if (!s.count) {
      s.frames.push(...this.preroll);
      s.count = this.preroll.reduce((n, f) => n + f.length, 0);
      this.preroll = [];
    }
    s.frames.push(frame);
    s.count += frame.length;
    s.quiet = voiced ? 0 : s.quiet + frame.length;
    if (s.quiet >= QUIET || s.count >= MAX_SEGMENT) this.seal();
    if (this.pending.reduce((n, item) => n + item.count, this.current.count) > 60 * RATE) {
      this.closed = true;
      this.seal();
      this.onError(new Error('CPU отстаёт более чем на минуту. Запись остановлена, накопленный звук обрабатывается.'));
    }
    this.schedule();
  }

  seal() {
    if (!this.current.count) return;
    this.current.sealed = true;
    this.pending.push(this.current);
    this.current = segment();
  }

  schedule() {
    if (this.running || this.failure) return;
    const s = this.pending[0] ?? this.current;
    if (!s.sealed && s.count - s.sent < UPDATE) return;
    this.running = this.run(s).catch(error => {
      this.closed = true;
      this.failure = error;
      this.onError(error);
    }).finally(() => {
      this.running = null;
      this.schedule();
    });
  }

  async run(s) {
    const count = s.count, frames = s.frames.slice();
    s.sent = count;
    const started = performance.now();
    const data = await this.decode(frames);
    const updateMs = performance.now() - started;
    const final = s.sealed && s.count === count;
    s.agreement.update(data.text, final);
    const stable = [...this.history, ...s.agreement.stable].join(' ');
    this.onUpdate({stable, partial: s.agreement.draft.join(' '), window_s: count / RATE,
      captured_s: this.recorded / RATE, update_s: updateMs / 1000, final, ...data});
    if (final) {
      this.history.push(...s.agreement.stable);
      this.pending.shift();
    }
  }

  async finish() {
    this.closed = true;
    this.seal();
    this.schedule();
    while (this.running) await this.running;
    if (this.failure) throw this.failure;
    return this.history.join(' ');
  }
}
