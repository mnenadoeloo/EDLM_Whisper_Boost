"""Local browser demo for a CPU-only whisper.cpp build (Python standard library)."""

import argparse
import io
import json
import subprocess
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from research.benchmark import FLAGS, audio_context

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent
MODELS = ROOT / '.work/models'
CLI = ROOT / '.work/whisper.cpp/build/bin/whisper-cli'
MAX_BYTES = 20 * 1024 * 1024
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.mjs': ('app.mjs', 'text/javascript'), '/stream.mjs': ('stream.mjs', 'text/javascript'),
          '/mic.js': ('mic.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}


def samples():
    manifest = ROOT / 'data/smoke.jsonl'
    if not manifest.exists():
        return {}
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return {row['id']: row for row in rows if (ROOT / row['audio']).is_file()}


def available_models():
    return {path.stem: path for path in sorted(MODELS.glob('*.bin')) if path.is_file()}


def transcribe(wav, model, language, threads, floor, resident=None):
    with wave.open(io.BytesIO(wav), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Нужен монофайл WAV PCM16 16 кГц.')
        frames = audio.getnframes()
        pcm = audio.readframes(frames)
        if not frames or len(pcm) != frames * 2:
            raise ValueError('Пустой или повреждённый WAV.')
    duration = frames / 16000
    context = audio_context(duration, floor) if floor else 0
    if resident is not None:
        start = time.perf_counter()
        text = resident.decode(pcm, language, threads, context)
        elapsed = time.perf_counter() - start
        return {'text': text, 'audio_s': duration, 'compute_s': elapsed,
                'rtf': elapsed / duration, 'context': context}
    with tempfile.TemporaryDirectory(prefix='whisper-demo-') as directory:
        path = Path(directory) / 'clip.wav'
        path.write_bytes(wav)
        prefix = Path(directory) / 'transcript'
        cmd = [str(CLI), *FLAGS, '-m', str(model), '-f', str(path), '-l', language,
               '-t', str(threads), '-of', str(prefix)]
        if context:
            cmd += ['-ac', str(context)]
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.perf_counter() - start
        output = prefix.with_suffix('.txt')
        if proc.returncode or not output.is_file():
            raise RuntimeError('whisper-cli: ' + proc.stderr.strip()[-800:])
        return {'text': ' '.join(output.read_text().split()), 'audio_s': duration,
                'compute_s': elapsed, 'rtf': elapsed / duration, 'context': context}


class Handler(BaseHTTPRequestHandler):
    def send_body(self, body, kind='application/json', code=200):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def json(self, value, code=200):
        self.send_body(json.dumps(value, ensure_ascii=False).encode('utf8'), code=code)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ASSETS:
            name, mime = ASSETS[url.path]
            self.send_body((STATIC / name).read_bytes(), mime)
        elif url.path == '/api/config':
            self.json({'models': [self.server.resident_name] if self.server.resident else list(available_models()),
                       'backend': 'resident' if self.server.resident else 'cli',
                       'model_load_s': self.server.resident.load_s if self.server.resident else None,
                       'threads': self.server.threads,
                       'runtime': CLI.is_file(), 'samples': [{'id': k, 'language': row['language']}
                                                            for k, row in samples().items()]})
        elif url.path == '/api/sample':
            row = samples().get(parse_qs(url.query).get('id', [''])[0])
            if row:
                self.send_body((ROOT / row['audio']).read_bytes(), 'audio/wav')
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != '/api/transcribe':
            return self.send_error(404)
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':
            return self.send_error(403)
        params = parse_qs(url.query)
        model = available_models().get(params.get('model', [''])[0])
        language = params.get('language', [''])[0]
        floor = {'full': 0, 'adaptive': 256, 'conservative': 768}.get(params.get('context', ['full'])[0])
        if not model or language not in ('en', 'ru'):
            return self.json({'error': 'Выберите доступную модель и язык.'}, 400)
        if self.server.resident and model.stem != self.server.resident_name:
            return self.json({'error': 'Выберите модель, загруженную в память.'}, 400)
        if floor is None:
            return self.json({'error': 'Неизвестный режим аудиоконтекста.'}, 400)
        if not self.server.worker.acquire(blocking=False):
            return self.json({'error': 'Модель обрабатывает другое окно. Повторите после завершения.'}, 409)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= MAX_BYTES:
                return self.json({'error': 'Размер WAV должен быть от 1 байта до 20 МиБ.'}, 413)
            payload = self.rfile.read(size)
            self.json(transcribe(payload, model, language, self.server.threads, floor, self.server.resident))
        except (ValueError, EOFError, wave.Error) as exc:
            self.json({'error': str(exc) or 'Повреждённый WAV.'}, 400)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            self.json({'error': str(exc)}, 500)
        finally:
            self.server.worker.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resident-model', choices=sorted(available_models()), help='keep this model in CPU memory')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.threads = args.threads
    server.worker = threading.Lock()
    server.resident_name = args.resident_model
    server.resident = None
    with server:
        if args.resident_model:
            from demo.resident import Resident
            server.resident = Resident(available_models()[args.resident_model], ROOT / '.work/whisper.cpp')
        print(f'http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        finally:
            with server.worker:
                if server.resident:
                    server.resident.close()
