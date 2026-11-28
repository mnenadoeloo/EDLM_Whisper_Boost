import {LiveSession, RATE} from './stream.mjs';

const FIRST = RATE, UPDATE = 1.5 * RATE, PAUSE = 0.2 * RATE;

// While the speaker pauses, decode a possible ending without closing the phrase.
// New voiced samples invalidate the candidate. The normal 800 ms endpoint and
// explicit stop remain the only ways (apart from the length cap) to seal a phrase.
export class EndpointSession extends LiveSession {
  schedule() {
    if (this.running || this.failure) return;
    const s = this.pending[0] ?? this.current;
    if (!s.count) return;
    const speechEnd = s.count - s.quiet;
    const candidate = s.candidate?.speechEnd === speechEnd ? s.candidate : null;
    if (!s.sealed) {
      if (s.quiet >= PAUSE) {
        if (candidate) return;
      } else if (s.count - s.sent < (s.sent ? UPDATE : FIRST)) return;
    }
    this.running = this.decodeEnding(s, speechEnd, candidate).catch(error => {
      this.closed = true;
      this.failure = error;
      this.onError(error);
    }).finally(() => {
      this.running = null;
      this.schedule();
    });
  }

  async decodeEnding(s, speechEnd, candidate) {
    const count = s.count;
    const paused = s.quiet >= PAUSE;
    s.sent = count;
    const start = performance.now();
    const result = candidate ? {...candidate.result, compute_s: 0, rtf: 0}
      : await this.decode(s.frames.slice());
    const decoded = candidate ? candidate.count : count;
    const voiceUnchanged = s.count - s.quiet === speechEnd;
    const coversEnding = voiceUnchanged && (paused || candidate || count === s.count);
    if (paused && voiceUnchanged && !candidate) s.candidate = {speechEnd, count, result};
    const final = s.sealed && coversEnding;

    // An early preview can contain a half-spoken word. Let the final decode
    // replace the whole tentative phrase and preserve completed phrases only.
    if (final) {
      if (result.text.trim()) this.history.push(result.text.trim());
      this.pending.shift();
    }
    this.onUpdate({...result, stable: this.history.join(' '), partial: final ? '' : result.text.trim(),
      window_s: decoded / RATE, captured_s: this.recorded / RATE,
      update_s: (performance.now() - start) / 1000, final: Boolean(final), reused: Boolean(candidate)});
  }
}
