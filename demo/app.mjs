import {LiveSession, RATE, infer, replayPcm} from './stream.mjs';
const $ = id => document.getElementById(id);
let mode = 'file', busy = false, selected, mic, replay, config, objectUrl, sourceVersion = 0;
const sec = s => s.toLocaleString('ru-RU', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' с';
const options = () => ({model: $('model').value, language: $('language').value, context: $('context').value});
function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
function drawText(stable = '', partial = '') {
  $('stable').textContent = stable + (stable && partial ? ' ' : '');
  $('partial').textContent = partial;
  $('placeholder').hidden = Boolean(stable || partial);
}
function clearResult() {
  drawText();
  for (const id of ['audioMetric','timeMetric','rtfMetric','firstMetric','confirmMetric','tailMetric']) $(id).textContent = '—';
}
function controls() {
  const live = mode !== 'file', microphone = mode === 'mic';
  const active = Boolean((mic || replay) && !(mic || replay).stopping);
  $('filePanel').hidden = microphone;
  $('micPanel').hidden = !microphone;
  $('draftHint').hidden = !live;
  $('fileMode').setAttribute('aria-pressed', String(!live));
  $('micMode').setAttribute('aria-pressed', String(microphone));
  $('replayMode').setAttribute('aria-pressed', String(mode === 'replay'));
  for (const id of ['fileMode','micMode','replayMode','model','language','context','sample','file']) $(id).disabled = busy;
  $('action').textContent = microphone ? (active ? 'Остановить запись' : 'Начать запись')
    : mode === 'replay' ? (active ? 'Остановить поток' : 'Воспроизвести потоком') : 'Распознать';
  $('action').classList.toggle('recording', active);
  $('action').disabled = !config?.runtime || !$('model').value || (busy && !active) || (!microphone && !selected);
  $('audioLabel').textContent = live ? 'Окно распознавания' : 'Длина записи';
  $('timeLabel').textContent = live ? 'Последнее обновление' : 'Время CLI';
  $('rtfLabel').textContent = live ? 'RTF окна' : 'RTF';
  document.querySelectorAll('.liveMetric').forEach(el => el.hidden = !live);
}
function changeMode(next) {
  mode = next;
  clearResult();
  controls();
  status('');
}
$('fileMode').onclick = () => changeMode('file');
$('replayMode').onclick = () => changeMode('replay');
$('micMode').onclick = () => changeMode('mic');
function selectFile(blob, name) {
  selected = blob;
  $('fileName').textContent = name || 'Выбрать аудиофайл';
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = blob ? URL.createObjectURL(blob) : '';
  $('player').hidden = !blob;
  if (blob) $('player').src = objectUrl;
  else $('player').removeAttribute('src');
  clearResult(); controls(); status('');
}
$('file').onchange = () => {
  sourceVersion++; $('sample').value = '';
  selectFile($('file').files[0], $('file').files[0]?.name);
};
$('sample').onchange = async () => {
  const version = ++sourceVersion, id = $('sample').value;
  $('file').value = '';
  selectFile(null);
  if (!id) return;
  try {
    const response = await fetch('/api/sample?id=' + encodeURIComponent(id));
    if (!response.ok) throw new Error('Не удалось открыть пример.');
    const data = await response.blob();
    if (version !== sourceVersion) return;
    $('language').value = config.samples.find(sample => sample.id === id).language;
    selectFile(data, id);
  } catch (error) { if (version === sourceVersion) status(error.message, true); }
};
function metrics(data) {
  $('audioMetric').textContent = sec(data.audio_s);
  $('timeMetric').textContent = sec(data.update_s ?? data.compute_s);
  $('rtfMetric').textContent = data.rtf.toFixed(3);
}
async function readPcm() {
  status('Подготовка аудио…');
  const input = await new OfflineAudioContext(1, 1, RATE).decodeAudioData(await selected.arrayBuffer());
  if (input.length > (20 * 1024 * 1024 - 44) / 2) throw new Error('Выберите запись короче 10 минут 55 секунд.');
  const render = new OfflineAudioContext(1, input.length, RATE);
  const src = render.createBufferSource(); src.buffer = input; src.connect(render.destination); src.start();
  const mono = (await render.startRendering()).getChannelData(0);
  return Int16Array.from(mono, sample => Math.max(-32768, Math.min(32767, Math.round(sample * 32768))));
}
async function offline() {
  const pcm = await readPcm();
  status('Распознавание на CPU…');
  const data = await infer('', options(), [pcm]);
  drawText(data.text);
  metrics(data);
  status(data.text ? 'Готово.' : 'Речь не распознана.');
}
function newSession(run, onError) {
  const opts = options();
  return new LiveSession(frames => infer('', opts, frames), data => {
    drawText(data.stable, data.partial); metrics(data);
    const elapsed = sec((performance.now() - run.started) / 1000);
    if (!run.first && (data.partial || data.stable)) { $('firstMetric').textContent = elapsed; run.first = true; }
    if (!run.firstStable && data.stable) { $('confirmMetric').textContent = elapsed; run.firstStable = true; }
  }, onError);
}
async function playStream() {
  const pcm = await readPcm(), player = $('player');
  player.currentTime = 0;
  player.controls = false;
  try {
    await player.play();
    replay = {started: performance.now()};
    let error;
    const session = newSession(replay, cause => { error = cause; });
    controls(); status('Подача записи в реальном времени…');
    await replayPcm(pcm, session, () => replay.stopping);
    replay.stopping = true; controls();
    const stopped = replay.stopAt ?? performance.now();
    status('Завершение последней фразы…');
    const text = await session.finish();
    drawText(text);
    $('tailMetric').textContent = sec((performance.now() - stopped) / 1000);
    status(error?.message || (text ? 'Поток завершён.' : 'Речь не распознана.'), Boolean(error));
  } finally { player.pause(); player.controls = true; replay = null; }
}
async function stopMic() {
  const m = mic;
  if (!m || m.stopping) return;
  const stopAt = performance.now();
  m.stopping = true; controls(); status('Завершение последней фразы…');
  try {
    if (m.ctx.state === 'running') {
      const flushed = new Promise(resolve => m.flushed = resolve);
      m.rec.port.postMessage('stop');
      await flushed;
    }
    m.stream.getTracks().forEach(track => track.stop());
    await m.ctx.close();
    const text = await m.session.finish();
    drawText(text);
    $('tailMetric').textContent = sec((performance.now() - stopAt) / 1000);
    status(m.error || (text ? 'Запись завершена.' : 'Речь не распознана.'), Boolean(m.error));
  } catch (error) { status(error.message, true); }
  finally { m.stream.getTracks().forEach(track => track.stop()); mic = null; busy = false; controls(); }
}
async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1}});
  let ctx;
  try {
    ctx = new AudioContext({sampleRate: RATE});
    await ctx.audioWorklet.addModule('/mic.js');
    await ctx.resume();
    const rec = new AudioWorkletNode(ctx, 'pcm-recorder', {channelCount: 1, channelCountMode: 'explicit'});
    const src = ctx.createMediaStreamSource(stream);
    const m = {ctx, stream, rec, started: performance.now()};
    m.session = newSession(m, error => { m.error = error.message; void stopMic(); });
    mic = m;
    rec.port.onmessage = event => {
      if (event.data === 'flushed') m.flushed?.();
      else m.session.push(event.data);
    };
    src.connect(rec);
    // Keep the graph pulled by the output device. The worklet leaves its output
    // zero-filled, so microphone audio never feeds back through the speakers.
    rec.connect(ctx.destination);
    stream.getAudioTracks()[0].onended = stopMic;
    controls(); status('Идёт запись…');
  } catch (error) { stream.getTracks().forEach(track => track.stop()); if (ctx) await ctx.close(); throw error; }
}
$('action').onclick = async () => {
  if (mic) return stopMic();
  if (replay) {
    replay.stopAt = performance.now(); replay.stopping = true;
    $('player').pause(); controls(); return;
  }
  busy = true; clearResult(); controls();
  try {
    if (mode === 'file') await offline();
    else if (mode === 'replay') await playStream();
    else await startMic();
  }
  catch (error) { status(error.message, true); }
  finally { if (!mic) { busy = false; controls(); } }
};
try {
  config = await (await fetch('/api/config')).json();
  for (const model of config.models) $('model').add(new Option(model, model));
  if (config.models.includes('enc_attn')) {
    $('model').value = 'enc_attn';
    $('context').value = 'conservative';
  } else if (config.models.includes('q4_0')) $('model').value = 'q4_0';
  if (config.backend === 'resident' && $('model').value === 'm3_enc1') $('context').value = 'conservative';
  for (const sample of config.samples) $('sample').add(new Option(sample.language.toUpperCase() + ' · ' + sample.id, sample.id));
  $('sampleField').hidden = !config.samples.length;
  $('threads').textContent = 'whisper.cpp · ' + config.threads + ' потока CPU'
    + (config.backend === 'resident' ? ' · модель в памяти' : '');
  status(config.models.length && config.runtime ? '' : 'Сначала подготовьте runtime и checkpoint по инструкции запуска.', true);
} catch (error) { status(error.message, true); }
controls();
