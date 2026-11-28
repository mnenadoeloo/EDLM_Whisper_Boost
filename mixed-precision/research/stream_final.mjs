// Replay the same PCM into three versions of the browser flow at recording speed.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {LiveSession, RATE, infer, replayPcm} from '../demo/stream.mjs';
import {EndpointSession} from '../demo/endpoint.mjs';

const [base, manifest] = process.argv.slice(2);
const readRows = async path => (await fs.readFile(path, 'utf8')).trim().split('\n').map(s => JSON.parse(s));
const config = await (await fetch(base + '/api/config')).json();
if (config.backend !== 'resident' || config.threads !== 4 || config.models.join() !== 'm3_enc1') {
  throw new Error('Expected the four-thread CPU server with m3_enc1 in memory');
}

async function load(row) {
  // prepare_audio.py writes a 44-byte PCM16 mono header followed by the exact samples.
  const wav = await fs.readFile(row.audio);
  if (createHash('sha256').update(wav).digest('hex') !== row.wav_sha256 ||
      wav.length !== 44 + Math.round(row.duration_s * RATE) * 2 ||
      wav.toString('ascii', 36, 40) !== 'data' || wav.readUInt32LE(24) !== RATE) {
    throw new Error('Recording differs from the prepared WAV: ' + row.id);
  }
  return new Int16Array(wav.buffer, wav.byteOffset + 44, (wav.length - 44) / 2);
}

const options = language => ({language, model: 'm3_enc1', context: 'conservative'});
const smoke = (await readRows('data/smoke.jsonl'))[0];
await infer(base, options(smoke.language), [await load(smoke)]);

async function measure(policy, row, pcm) {
  const started = performance.now();
  const clock = () => (performance.now() - started) / 1000;
  let samples = 0, first = null;
  const calls = [];
  async function decode(frames) {
    const start = clock(), available = samples / RATE;
    const result = await infer(base, options(row.language), frames);
    calls.push({start_s: start, available_s: available, end_s: clock(), audio_s: result.audio_s,
      compute_s: result.compute_s, context: result.context});
    return result;
  }

  const batch = [];
  const record = u => { if (first === null && (u.stable || u.partial)) first = clock(); };
  const Session = policy === 'endpoint' ? EndpointSession : LiveSession;
  const stream = policy === 'batch' ? null : new Session(decode, record, () => {});
  await replayPcm(pcm, {push(frame) {
    samples += frame.length;
    if (stream) stream.push(frame);
    else batch.push(frame);
  }});
  const capture = clock();
  const text = stream ? await stream.finish() : (await decode(batch)).text;
  const final = clock();
  if (!stream && text) first = final;
  if (samples !== pcm.length || !calls.length) throw new Error('Incomplete recording: ' + row.id);
  return {id: row.id, language: row.language, wav_sha256: row.wav_sha256, policy, text, duration_s: samples / RATE,
    first_s: first, capture_s: capture, final_s: final, tail_s: final - capture, calls};
}

const rows = await readRows(manifest);
const orders = [
  ['batch','classic','endpoint'], ['classic','endpoint','batch'], ['endpoint','batch','classic'],
  ['endpoint','classic','batch'], ['classic','batch','endpoint'], ['batch','endpoint','classic'],
];
const offsets = new Map();
for (const row of rows) {
  const pcm = await load(row), index = offsets.get(row.language) ?? 0;
  const order = orders[index % orders.length];
  for (const [i, policy] of order.entries()) {
    const result = await measure(policy, row, pcm);
    console.log(JSON.stringify({...result, position: i + 1}));
  }
  offsets.set(row.language, index + 1);
  console.error(row.id + ' completed');
}
