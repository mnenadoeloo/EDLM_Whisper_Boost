"""Load one CPU model for a local demo server and reuse it across serialized requests."""
import ctypes as ct
import os
import subprocess
import tempfile
import time
from pathlib import Path


class Resident:
    def __init__(self, model, runtime):
        libdir = runtime / "build/src"
        with tempfile.TemporaryDirectory() as temp:
            shim = Path(temp) / "resident.so"
            subprocess.run(["c++", "-std=c++17", "-O2", "-fPIC", "-shared", Path(__file__).with_suffix(".cpp"),
                            "-I", runtime / "include", "-I", runtime / "ggml/include", "-L", libdir,
                            "-lwhisper", f"-Wl,-rpath,{libdir}", "-o", shim], check=True)
            self.lib = ct.CDLL(str(shim))
        self.lib.resident_open.argtypes = [ct.c_char_p]
        self.lib.resident_open.restype = ct.c_void_p
        self.lib.resident_decode.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int, ct.c_char_p, ct.c_int, ct.c_int]
        self.lib.resident_decode.restype = ct.c_char_p
        self.lib.resident_close.argtypes = [ct.c_void_p]
        self.lib.resident_close.restype = None
        started = time.perf_counter()
        self.handle = self.lib.resident_open(os.fsencode(model))
        self.load_s = time.perf_counter() - started
        if not self.handle:
            raise RuntimeError(f"Could not load CPU model {model}")

    def decode(self, pcm, language, threads, context):
        result = self.lib.resident_decode(self.handle, pcm, len(pcm) // 2, language.encode(), threads, context)
        if result is None:
            raise RuntimeError("libwhisper failed to decode an audio window")
        return " ".join(result.decode().split())

    def close(self):
        if self.handle:
            self.lib.resident_close(self.handle)
            self.handle = None
