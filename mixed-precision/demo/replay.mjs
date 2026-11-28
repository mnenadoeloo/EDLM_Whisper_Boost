// Replay the manifest WAVs at recording speed using the browser's scheduler.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {LiveSession, RATE, infer, replayPcm} from './stream.mjs';

const [model = 'q4_0', mode = 'full', manifest = 'data/smoke.jsonl', base = 'http://127.0.0.1:8765', secondBase] = process.argv.slice(2);
if (!['full', 'adaptive', 'conservative'].includes(mode)) throw new Error('Expected full, adaptive or conservative');
const rows = (await fs.readFile(manifest, 'utf8')).trim().split('\n').map(line => JSON.parse(line));
const servers = [];
for (const url of [base, secondBase].filter(Boolean)) {
  const config = await (await fetch(url + '/api/config')).json();
  if (!config.runtime || !config.models.includes(model)) throw new Error('Runtime or model unavailable');
  servers.push({url, config});
}
if (secondBase && (servers[0].config.backend !== 'cli' || servers[1].config.backend !== 'resident'
    || servers[0].config.threads !== servers[1].config.threads)) throw new Error('Expected matched CLI and resident backends');

async function replay(row, {url, config}, order) {
  const buffer = await fs.readFile(row.audio);
  const wavSha = createHash('sha256').update(buffer).digest('hex');
  if (row.wav_sha256 && row.wav_sha256 !== wavSha) throw new Error('WAV checksum mismatch: ' + row.id);
  let pcm;
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') throw new Error('Expected RIFF WAVE');
  let valid = false;
  for (let p = 12; p + 8 <= buffer.length;) {
    const size = buffer.readUInt32LE(p + 4), type = buffer.toString('ascii', p, p + 4);
    if (type === 'fmt ') valid = buffer.readUInt16LE(p + 8) === 1 && buffer.readUInt16LE(p + 10) === 1
      && buffer.readUInt32LE(p + 12) === RATE && buffer.readUInt16LE(p + 22) === 16;
    if (type === 'data' && valid) {
      pcm = Int16Array.from({length: size / 2}, (_, i) => buffer.readInt16LE(p + 8 + 2 * i));
      break;
    }
    p += 8 + size + size % 2;
  }
  if (!pcm?.length) throw new Error('Expected 16 kHz mono PCM16 WAV');
  const updates = [], opts = {language: row.language, model, context: mode};
  let failure;
  const session = new LiveSession(async frames => {
    const hash = createHash('sha256');
    for (const f of frames) hash.update(Buffer.from(f.buffer, f.byteOffset, f.byteLength));
    return {...await infer(url, opts, frames), pcm_sha256: hash.digest('hex')};
  },
    data => updates.push({time_s: (performance.now() - start) / 1000, ...data}),
    error => { failure = error; });
  const start = performance.now();
  await replayPcm(pcm, session);
  if (failure) throw failure;
  const stopped = performance.now(), text = await session.finish(), elapsed = (performance.now() - start) / 1000;
  const latencies = updates.map(item => item.update_s).sort((a,b) => a - b);
  let revised = 0, revisable = 0;
  const tokens = u => (u.stable + ' ' + u.partial).toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  for (let i = 1; i < updates.length; i++) {
    if (!updates[i - 1].partial) continue;
    revisable++;
    const next = tokens(updates[i]);
    if (tokens(updates[i - 1]).some((word, j) => word !== next[j])) revised++;
  }
  console.log(JSON.stringify({id: row.id, language: row.language, model, mode, reference: row.reference, text,
    wav_sha256: wavSha, backend: config.backend, pair_position: order, model_load_s: config.model_load_s,
    threads: config.threads, duration_s: pcm.length / RATE,
    first_text_s: updates.find(u => u.stable || u.partial)?.time_s ?? null,
    first_confirmed_s: updates.find(u => u.stable)?.time_s ?? null,
    p95_update_s: latencies[Math.ceil(0.95 * latencies.length) - 1] ?? null,
    revised_updates: revised, revisable_updates: revisable,
    final_s: elapsed, after_stop_s: elapsed - (stopped - start) / 1000,
    compute_rtf: updates.reduce((sum,u) => sum + u.compute_s, 0) / (pcm.length / RATE), updates}));
}

const seen = new Map();
for (const row of rows) {
  const n = seen.get(row.language) ?? 0;
  const pair = n % 2 ? [...servers].reverse() : servers;
  for (const [i, server] of pair.entries()) await replay(row, server, i + 1);
  seen.set(row.language, n + 1);
}
