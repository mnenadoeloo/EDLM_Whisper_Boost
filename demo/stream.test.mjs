import test from 'node:test';
import assert from 'node:assert/strict';
import {Agreement, LiveSession, FRAME} from './stream.mjs';
import {EndpointSession} from './endpoint.mjs';

test('confirmed words survive decoder rewrites before the boundary', () => {
  const a = new Agreement();
  a.update('Мы уже здесь', false);
  a.update('мы уже ждём', false);
  assert.equal(a.stable.join(' '), 'мы уже');
  a.update('ну, мы уже ждём тебя', false);
  assert.deepEqual(a.stable, ['мы', 'уже', 'ждём']);
  a.update('мы уже ждём тебя дома.', true);
  assert.equal(a.stable.join(' '), 'мы уже ждём тебя дома.');
});

test('slow decode skips stale partial windows and preserves sealed phrases in order', async () => {
  const requests = [], events = [], resolvers = [];
  const session = new LiveSession(frames => {
    requests.push(frames.length);
    return new Promise(resolve => resolvers.push(text => resolve({text})));
  }, state => events.push(state), assert.fail);
  const voice = new Int16Array(FRAME).fill(2000), silence = new Int16Array(FRAME);
  const feed = (count, frame) => { for (let i = 0; i < count; i++) session.push(frame); };
  feed(45, voice); feed(8, silence); feed(20, voice);
  assert.deepEqual(requests, [15]);
  const done = session.finish();
  resolvers.shift()('сегодня');
  await new Promise(setImmediate);
  assert.deepEqual(requests, [15, 53]);
  resolvers.shift()('сегодня вечером');
  await new Promise(setImmediate);
  assert.deepEqual(requests, [15, 53, 20]);
  resolvers.shift()('будет дождь');
  assert.equal(await done, 'сегодня вечером будет дождь');
  assert.deepEqual(events.map(e => e.final), [false, true, true]);
});

test('quiet input skips inference, a short final utterance is flushed', async () => {
  let count = 0;
  const session = new LiveSession(async () => { count++; return {text: 'да'}; }, () => {}, assert.fail);
  const quiet = new Int16Array(FRAME);
  for (let i = 0; i < 65; i++) session.push(quiet);
  assert.equal(count, 0);
  session.push(new Int16Array(FRAME).fill(1000));
  assert.equal(await session.finish(), 'да');
  assert.equal(count, 1);
});

test('quiet speech above the digital noise floor is kept', async () => {
  let sent = 0;
  const session = new LiveSession(async frames => {
    sent += frames.reduce((n, frame) => n + frame.length, 0);
    return {text: 'тихо'};
  }, () => {}, assert.fail);
  for (let i = 0; i < 5; i++) session.push(new Int16Array(FRAME).fill(4));
  for (let i = 0; i < 8; i++) session.push(new Int16Array(FRAME).fill(35));
  assert.equal(await session.finish(), 'тихо');
  assert.equal(sent, 10 * FRAME);
});

test('a decoding failure is propagated and stops scheduling', async () => {
  const session = new LiveSession(async () => { throw new Error('model unavailable'); }, assert.fail, () => {});
  session.push(new Int16Array(FRAME).fill(1500));
  await assert.rejects(() => session.finish(), /model unavailable/);
  assert.equal(session.closed, true);
});

test('continuous speech is split at 24 seconds without dropping any PCM', async () => {
  const requests = [], events = [];
  const session = new LiveSession(async frames => {
    requests.push(frames.reduce((n, f) => n + f.length, 0));
    return {text: requests.length === 3 ? 'потом' : 'сначала'};
  }, state => events.push(state), assert.fail);
  for (let i = 0; i < 310; i++) session.push(new Int16Array(FRAME).fill(1000));
  assert.equal(await session.finish(), 'сначала потом');
  assert.deepEqual(events.filter(e => e.final).map(e => e.window_s), [24, 7]);
});

test('a short pause starts a tentative ending; continued silence confirms it without another decode', async () => {
  const inputs = [], updates = [];
  const session = new EndpointSession(async frames => {
    inputs.push(frames.length);
    return {text: 'короткая фраза', compute_s: .4};
  }, event => updates.push(event), assert.fail);
  const quiet = new Int16Array(FRAME), voice = quiet.map(() => 700);
  for (let i = 0; i < 5; i++) session.push(voice);
  session.push(quiet);
  assert.deepEqual(inputs, []);
  session.push(quiet);
  await new Promise(setImmediate);
  assert.deepEqual(inputs, [7]);
  assert.equal(updates[0].partial, 'короткая фраза');
  assert.equal(updates[0].stable, '');
  for (let i = 0; i < 6; i++) session.push(quiet);
  assert.equal(await session.finish(), 'короткая фраза');
  assert.deepEqual(inputs, [7]);
  assert.equal(updates.at(-1).final, true);
  assert.equal(updates.at(-1).reused, true);
  assert.equal(updates.at(-1).compute_s, 0);
});

test('resuming speech while decoding a pause invalidates that ending and flushes a partial frame', async () => {
  const requests = [], resolve = [], events = [];
  const session = new EndpointSession(frames => {
    requests.push(frames.reduce((n, f) => n + f.length, 0));
    return new Promise(done => resolve.push(text => done({text})));
  }, event => events.push(event), assert.fail);
  const quiet = new Int16Array(FRAME), voice = quiet.map(() => 700);
  session.push(voice); session.push(quiet); session.push(quiet);
  session.push(voice.subarray(0, 901));
  const result = session.finish();
  resolve.shift()('ошибка');
  await new Promise(setImmediate);
  assert.deepEqual(requests, [3 * FRAME, 3 * FRAME + 901]);
  resolve.shift()('вся фраза');
  assert.equal(await result, 'вся фраза');
  assert.deepEqual(events.map(e => e.final), [false, true]);
});

test('earlier previews stay tentative and continuous speech keeps the 24-second split', async () => {
  const requests = [], events = [];
  const session = new EndpointSession(async frames => {
    requests.push(frames.length);
    return {text: frames.length === 10 ? 'обрывок' : 'фраза'};
  }, event => events.push(event), assert.fail);
  const voice = new Int16Array(FRAME).fill(700);
  for (let i = 0; i < 9; i++) session.push(voice);
  assert.deepEqual(requests, []);
  session.push(voice);
  await new Promise(setImmediate);
  assert.deepEqual(requests, [10]);
  for (let i = 10; i < 242; i++) session.push(voice);
  assert.equal(await session.finish(), 'фраза фраза');
  assert.deepEqual(events.filter(e => e.final).map(e => e.window_s), [24, .2]);
});
