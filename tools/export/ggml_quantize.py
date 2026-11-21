"""llama.cpp's own quantizers (ggml_quantize_chunk in libggml-base, here Homebrew's), for quantizing weights we hold as
float32 to llama.cpp's calibrated types, with an importance matrix as llama-quantize uses one."""

import ctypes
import ctypes.util
from pathlib import Path

import gguf
import numpy as np

LIB_PATHS = ['/opt/homebrew/lib/libggml-base.dylib', ctypes.util.find_library('ggml-base') or '']
_lib = None


def lib():
    global _lib
    if _lib is None:
        path = next(p for p in LIB_PATHS if p and Path(p).exists())
        _lib = ctypes.CDLL(path)
        _lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                             ctypes.c_int64, ctypes.c_void_p]
        _lib.ggml_row_size.restype = ctypes.c_size_t
        _lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        _lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    return _lib


def quantize(kind: int, w: np.ndarray, importance: np.ndarray | None = None) -> np.ndarray:
    """w [rows, k] float32 to `kind`'s GGUF bytes [rows, row bytes]; `importance` [k] weighs each column's error."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    rows, k = w.shape
    L = lib()
    L.ggml_quantize_init(kind)
    row_bytes = L.ggml_row_size(kind, k)
    out = np.empty((rows, row_bytes), np.uint8)
    imat = None if importance is None else np.ascontiguousarray(importance, dtype=np.float32)
    size = L.ggml_quantize_chunk(kind, w.ctypes.data, out.ctypes.data, 0, rows, k, None if imat is None else imat.ctypes.data)
    assert size == out.nbytes, (size, out.nbytes)
    return out


def dequantize(kind: int, data: np.ndarray, k: int) -> np.ndarray:
    """GGUF bytes [rows, row bytes] back to float32 [rows, k] (gguf-py)."""
    return gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind)).reshape(data.shape[0], k).astype(np.float32)


class RawCheckpoint:
    """Safetensors tensors as numpy without torch (which can't share a process with libggml: two OpenMP runtimes):
    float8 and uint8 as uint8, bfloat16 and float32 as float32."""

    def __init__(self, path: Path):
        import json
        import struct
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.data = np.memmap(path, np.uint8, mode='r')

    def __call__(self, name: str) -> np.ndarray:
        h = self.header[name]
        start, end = h['data_offsets']
        raw = self.data[self.base + start:self.base + end]
        if h['dtype'] in ('F8_E4M3', 'U8'):
            return raw.reshape(h['shape'])
        if h['dtype'] == 'BF16':
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(h['shape'])
        if h['dtype'] == 'F32':
            return raw.view(np.float32).reshape(h['shape'])
        raise ValueError(h['dtype'])

    def has(self, name: str) -> bool:
        return name in self.header


def e4m3_table() -> np.ndarray:
    """float8 E4M3FN byte -> float32 (0x7f/0xff NaN)."""
    b = np.arange(256)
    e, m = (b >> 3) & 15, (b & 7).astype(np.float64)
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where((b & 0x7f) == 0x7f, np.nan, v)
    return np.where(b & 0x80, -v, v).astype(np.float32)


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def experts(W: RawCheckpoint, prefix: str, first: int = 0, count: int | None = None) -> np.ndarray:
    """A layer's experts (`prefix` = 'model.llm.layers.L.mlp.experts.w13_weight') [E, N, K] float32, decoded as
    transformers' Inkling2LoopDequantizeExperts does (NVFP4 or MXFP8)."""
    codes = W(prefix)
    sl = slice(first, None if count is None else first + count)
    codes = codes[sl]
    scale = W(f'{prefix}_scale')[sl]
    if W.has(f'{prefix}_amax'):
        factor = W(f'{prefix}_amax')[sl] / W(f'{prefix}_outer_scale_denominator')
        values = np.stack([E2M1[codes & 15], E2M1[codes >> 4]], -1).reshape(codes.shape[0], codes.shape[1], -1)
        s = e4m3_table()[scale] * factor[:, None, None]
        return (values.reshape(*s.shape, 16) * s[..., None]).reshape(values.shape)
    values = e4m3_table()[codes]
    s = np.exp2(scale.astype(np.float32) - 127)
    return (values.reshape(*s.shape, 32) * s[..., None]).reshape(values.shape)
