class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Int16Array(1600);
    this.used = 0;
    this.stopped = false;
    this.port.onmessage = () => {
      this.stopped = true;
      if (this.used) this.port.postMessage(this.frame.slice(0, this.used));
      this.port.postMessage('flushed');
    };
  }
  process(inputs) {
    if (this.stopped) return false;
    const samples = inputs[0]?.[0];
    if (samples) for (const sample of samples) {
      this.frame[this.used++] = Math.max(-32768, Math.min(32767, Math.round(sample * 32768)));
      if (this.used === this.frame.length) {
        this.port.postMessage(this.frame, [this.frame.buffer]);
        this.frame = new Int16Array(1600);
        this.used = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-recorder', PCMRecorder);
