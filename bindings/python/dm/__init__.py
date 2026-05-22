"""
dm — Python binding for libdm (ctypes-based)

Install / use:
    export DM_LIB=/path/to/libdm.so   # or libdm.dylib / dm.dll
    python -c "import dm; print(dm.version())"

Or place libdm.so next to this package and it will be found automatically.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ── Library loading ───────────────────────────────────────────────────────────

def _load_lib() -> ctypes.CDLL:
    """Locate and load libdm.  Search order:
    1. DM_LIB environment variable
    2. package directory
    3. ctypes.util.find_library
    """
    candidates: list[str] = []

    env = os.environ.get("DM_LIB")
    if env:
        candidates.append(env)

    pkg_dir = Path(__file__).parent
    for name in ("libdm.so", "libdm.so.1", "libdm.dylib", "dm.dll"):
        candidates.append(str(pkg_dir / name))
        candidates.append(str(pkg_dir.parent.parent.parent / name))  # repo root

    found = ctypes.util.find_library("dm")
    if found:
        candidates.append(found)

    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue

    raise OSError(
        "libdm not found.  Set DM_LIB=/path/to/libdm.so or copy the library "
        "next to this package."
    )


_lib = _load_lib()

# ── C types / status ──────────────────────────────────────────────────────────

c_size_t  = ctypes.c_size_t
c_uint32  = ctypes.c_uint32
c_uint16  = ctypes.c_uint16
c_int     = ctypes.c_int
c_float   = ctypes.c_float
c_double  = ctypes.c_double
c_char_p  = ctypes.c_char_p
c_void_p  = ctypes.c_void_p
c_bool    = ctypes.c_bool

DM_Status = ctypes.c_int

DM_OK                =  0
DM_ERR_GENERIC       = -1
DM_ERR_IO            = -2
DM_ERR_MEMORY        = -3
DM_ERR_INVALID_PARAM = -4
DM_ERR_NOT_FOUND     = -5
DM_ERR_INCOMPATIBLE  = -6
DM_ERR_NOT_SUPPORTED = -7

# Opaque handle aliases
DM_Dataset   = ctypes.c_void_p
DM_Algorithm = ctypes.c_void_p
DM_Tokenizer = ctypes.c_void_p
DM_Vision    = ctypes.c_void_p
DM_LM        = ctypes.c_void_p
DM_BitSet    = ctypes.c_void_p
DM_DataGen   = ctypes.c_void_p
DM_GpuCtx    = ctypes.c_void_p


# ── Function signatures ───────────────────────────────────────────────────────

def _fn(name: str, restype, *argtypes):
    f = getattr(_lib, name)
    f.restype  = restype
    f.argtypes = list(argtypes)
    return f


# § 1  Core
_dm_version        = _fn("dm_version",        c_char_p)
_dm_version_number = _fn("dm_version_number", ctypes.c_uint32)
_dm_init           = _fn("dm_init",           DM_Status)
_dm_strerror       = _fn("dm_strerror",       c_char_p, DM_Status)

# § 2  Dataset
_dm_dataset_open    = _fn("dm_dataset_open",    DM_Dataset,  c_char_p, c_char_p)
_dm_dataset_count   = _fn("dm_dataset_count",   c_size_t,   DM_Dataset)
_dm_dataset_max_id  = _fn("dm_dataset_max_id",  c_uint32,   DM_Dataset)
_dm_dataset_free    = _fn("dm_dataset_free",     None,       DM_Dataset)

# § 3  Algorithm
_dm_algorithm_create = _fn("dm_algorithm_create", DM_Algorithm, c_char_p)
_dm_algorithm_run    = _fn("dm_algorithm_run",    DM_Status,
                            DM_Algorithm, c_char_p, c_char_p, c_double,
                            ctypes.POINTER(c_char_p))
_dm_algorithm_list   = _fn("dm_algorithm_list",   DM_Status, c_char_p, c_int)
_dm_algorithm_free   = _fn("dm_algorithm_free",   None, DM_Algorithm)

# § 4  Tokenizer
_dm_tokenizer_create     = _fn("dm_tokenizer_create",     DM_Tokenizer, c_char_p)
_dm_tokenizer_train      = _fn("dm_tokenizer_train",      DM_Status, DM_Tokenizer, c_char_p, c_int, c_char_p)
_dm_tokenizer_load       = _fn("dm_tokenizer_load",       DM_Status, DM_Tokenizer, c_char_p)
_dm_tokenizer_encode     = _fn("dm_tokenizer_encode",     DM_Status, DM_Tokenizer, c_char_p,
                                ctypes.POINTER(c_uint32), ctypes.POINTER(c_int))
_dm_tokenizer_decode     = _fn("dm_tokenizer_decode",     DM_Status, DM_Tokenizer,
                                ctypes.POINTER(c_uint32), c_int, c_char_p, c_int)
_dm_tokenizer_vocab_size = _fn("dm_tokenizer_vocab_size", c_int, DM_Tokenizer)
_dm_tokenizer_token_text = _fn("dm_tokenizer_token_text", c_char_p, DM_Tokenizer, c_uint32,
                                ctypes.POINTER(c_uint32))
_dm_tokenizer_free       = _fn("dm_tokenizer_free",       None, DM_Tokenizer)
_dm_tokenizer_volt_run   = _fn("dm_tokenizer_volt_run",   DM_Status,
                                c_char_p, c_int, c_int, c_int, c_char_p)

# § 5  Vision
_dm_vision_create      = _fn("dm_vision_create",      DM_Vision,  c_char_p)
_dm_vision_init        = _fn("dm_vision_init",         DM_Status,  DM_Vision, c_char_p, c_int, c_int, c_float, c_float)
_dm_vision_train       = _fn("dm_vision_train",        DM_Status,  DM_Vision, c_char_p, c_int, c_int, c_float)
_dm_vision_eval        = _fn("dm_vision_eval",         DM_Status,  DM_Vision, c_char_p,
                               ctypes.POINTER(c_float), ctypes.POINTER(c_float))
_dm_vision_predict     = _fn("dm_vision_predict",      DM_Status,  DM_Vision,
                               ctypes.POINTER(c_float), c_int, c_int,
                               ctypes.POINTER(c_float), c_int)
_dm_vision_save        = _fn("dm_vision_save",         DM_Status,  DM_Vision)
_dm_vision_free        = _fn("dm_vision_free",          None,       DM_Vision)
_dm_vision_forward_raw = _fn("dm_vision_forward_raw",  DM_Status,
                               ctypes.POINTER(c_float), c_int, c_int, c_int,
                               ctypes.POINTER(c_float), c_int)

# § 6  LM
_dm_lm_create   = _fn("dm_lm_create",   DM_LM,     c_char_p)
_dm_lm_train    = _fn("dm_lm_train",    DM_Status, DM_LM, c_char_p, c_char_p, c_int, c_int, c_float)
_dm_lm_generate = _fn("dm_lm_generate", DM_Status, DM_LM, c_char_p, c_int, c_char_p, c_int)
_dm_lm_load     = _fn("dm_lm_load",     DM_Status, DM_LM, c_char_p)
_dm_lm_free     = _fn("dm_lm_free",     None,      DM_LM)

# § 9  Benchmark
class _DM_BenchReport(ctypes.Structure):
    _fields_ = [
        ("phase_times_ms",       c_double * 4),
        ("peak_memory_kb",       c_size_t),
        ("user_cpu_ms",          c_double),
        ("sys_cpu_ms",           c_double),
        ("result_ram_bytes",     c_size_t),
        ("result_disk_est_bytes",c_size_t),
        ("num_patterns",         c_size_t),
        ("total_items",          c_size_t),
        ("throughput_mb_s",      c_double),
    ]

_dm_bench_reset      = _fn("dm_bench_reset",      None)
_dm_bench_start      = _fn("dm_bench_start",      None, c_int)
_dm_bench_stop       = _fn("dm_bench_stop",       None, c_int)
_dm_bench_record     = _fn("dm_bench_record",     None, c_size_t, c_size_t)
_dm_bench_get_report = _fn("dm_bench_get_report", _DM_BenchReport)
_dm_bench_print      = _fn("dm_bench_print",      None, c_char_p, c_char_p)

# § 10  BitSet
_dm_bitset_create   = _fn("dm_bitset_create",   DM_BitSet, c_size_t)
_dm_bitset_copy     = _fn("dm_bitset_copy",     DM_BitSet, DM_BitSet)
_dm_bitset_free     = _fn("dm_bitset_free",     None,      DM_BitSet)
_dm_bitset_set      = _fn("dm_bitset_set",      None,      DM_BitSet, c_size_t)
_dm_bitset_clear    = _fn("dm_bitset_clear",    None,      DM_BitSet, c_size_t)
_dm_bitset_get      = _fn("dm_bitset_get",      c_bool,    DM_BitSet, c_size_t)
_dm_bitset_and      = _fn("dm_bitset_and",      None,      DM_BitSet, DM_BitSet)
_dm_bitset_or       = _fn("dm_bitset_or",       None,      DM_BitSet, DM_BitSet)
_dm_bitset_not      = _fn("dm_bitset_not",      None,      DM_BitSet)
_dm_bitset_set_all  = _fn("dm_bitset_set_all",  None,      DM_BitSet)
_dm_bitset_popcount = _fn("dm_bitset_popcount", c_size_t,  DM_BitSet)

# § 11  DataGen
_dm_datagen_create = _fn("dm_datagen_create", DM_DataGen, c_char_p)
_dm_datagen_run    = _fn("dm_datagen_run",    DM_Status,  DM_DataGen, c_char_p, c_char_p, ctypes.c_uint)
_dm_datagen_free   = _fn("dm_datagen_free",   None,       DM_DataGen)

# § 13  CLI
_dm_cli_run = _fn("dm_cli_run", c_int, c_char_p, c_int, ctypes.POINTER(c_char_p))

# § 14  GPU
_dm_gpu_create       = _fn("dm_gpu_create",       DM_GpuCtx, c_int, c_char_p)
_dm_gpu_free         = _fn("dm_gpu_free",          None,      DM_GpuCtx)
_dm_gpu_ready        = _fn("dm_gpu_ready",         c_int,     DM_GpuCtx)
_dm_gpu_device_name  = _fn("dm_gpu_device_name",   c_int,     DM_GpuCtx, c_char_p, c_size_t)

# § 18  Experiment
_dm_timer_now                  = _fn("dm_timer_now",                  c_double)
_dm_peak_ram_mb                = _fn("dm_peak_ram_mb",                c_double)
_dm_file_size_mb               = _fn("dm_file_size_mb",               c_double, c_char_p)
_dm_dir_size_mb                = _fn("dm_dir_size_mb",                c_double, c_char_p)
_dm_ensure_dir                 = _fn("dm_ensure_dir",                 c_int,    c_char_p)
_dm_path_basename              = _fn("dm_path_basename",              c_char_p, c_char_p)
_dm_experiment_write_header    = _fn("dm_experiment_write_header",    None,     c_char_p)
_dm_experiment_generate_report = _fn("dm_experiment_generate_report", None,     c_char_p)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check(status: int, ctx: str = "") -> None:
    if status != DM_OK:
        msg = _dm_strerror(status)
        raise RuntimeError(f"{ctx}: {msg.decode() if msg else 'unknown error'} (code {status})")


def _enc(s: str | bytes) -> bytes:
    return s.encode() if isinstance(s, str) else s


# ── § 1  Core ─────────────────────────────────────────────────────────────────

def version() -> str:
    """Return the libdm version string."""
    return _dm_version().decode()


def version_number() -> tuple[int, int, int]:
    """Return (major, minor, patch)."""
    v = _dm_version_number()
    return (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF


def init() -> None:
    """Initialise the library (idempotent)."""
    _check(_dm_init(), "dm.init")


def strerror(code: int) -> str:
    return _dm_strerror(code).decode()


# ── § 2  Dataset ──────────────────────────────────────────────────────────────

class Dataset:
    """Wraps a DM_Dataset handle."""

    def __init__(self, path: str, type: str):
        """
        Parameters
        ----------
        type : str
            "transactional" | "utility" | "sequence" | "quantity" | "matrix"
        """
        h = _dm_dataset_open(_enc(path), _enc(type))
        if not h:
            raise RuntimeError(f"dm.Dataset: failed to open '{path}' as '{type}'")
        self._h = h

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_dataset_free(self._h)
            self._h = None

    def __enter__(self):  return self
    def __exit__(self, *_): self.__del__()

    @property
    def count(self) -> int:   return _dm_dataset_count(self._h)
    @property
    def max_id(self) -> int:  return _dm_dataset_max_id(self._h)


# ── § 3  Algorithm ────────────────────────────────────────────────────────────

class Algorithm:
    """Run one of the 132 registered data-mining algorithms."""

    def __init__(self, id: str):
        h = _dm_algorithm_create(_enc(id))
        if not h:
            raise ValueError(f"dm.Algorithm: unknown id '{id}'")
        self._h = h

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_algorithm_free(self._h)
            self._h = None

    def __enter__(self):  return self
    def __exit__(self, *_): self.__del__()

    def run(self, dataset_path: str, output_path: str,
            min_support: float, extra_args: Sequence[str] = ()) -> None:
        """Run the algorithm."""
        arr_type = c_char_p * (len(extra_args) + 1)
        arr = arr_type(*[_enc(a) for a in extra_args], None)
        _check(_dm_algorithm_run(self._h, _enc(dataset_path),
                                 _enc(output_path), min_support, arr),
               "dm.Algorithm.run")

    @staticmethod
    def list_all() -> List[str]:
        """Return all registered algorithm IDs."""
        buf = ctypes.create_string_buffer(32768)
        _check(_dm_algorithm_list(buf, len(buf)), "dm.Algorithm.list_all")
        return buf.value.decode().splitlines()


# ── § 4  Tokenizer ────────────────────────────────────────────────────────────

class Tokenizer:
    """Subword / byte-pair tokenizer."""

    def __init__(self, type: str):
        """
        type : "bpe" | "bpe_dropout" | "unigram" | "sentencepiece" |
               "wordpiece" | "gpe" | "parity_bpe" | "volt" |
               "maximal_munch" | "faro" | "tokenizer_lab"
        """
        h = _dm_tokenizer_create(_enc(type))
        if not h:
            raise ValueError(f"dm.Tokenizer: unknown type '{type}'")
        self._h = h

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_tokenizer_free(self._h)
            self._h = None

    def __enter__(self):  return self
    def __exit__(self, *_): self.__del__()

    def train(self, corpus_path: str, vocab_size: int, output_path: str) -> None:
        _check(_dm_tokenizer_train(self._h, _enc(corpus_path),
                                   vocab_size, _enc(output_path)),
               "dm.Tokenizer.train")

    def load(self, model_path: str) -> None:
        _check(_dm_tokenizer_load(self._h, _enc(model_path)), "dm.Tokenizer.load")

    def encode(self, text: str) -> List[int]:
        capacity = 8192
        ids = (c_uint32 * capacity)()
        length = c_int(capacity)
        status = _dm_tokenizer_encode(self._h, _enc(text), ids, ctypes.byref(length))
        if status == DM_ERR_MEMORY:
            capacity = length.value
            ids = (c_uint32 * capacity)()
            length = c_int(capacity)
            _check(_dm_tokenizer_encode(self._h, _enc(text), ids, ctypes.byref(length)),
                   "dm.Tokenizer.encode")
        else:
            _check(status, "dm.Tokenizer.encode")
        return list(ids[:length.value])

    def decode(self, ids: Sequence[int]) -> str:
        arr = (c_uint32 * len(ids))(*ids)
        buf = ctypes.create_string_buffer(len(ids) * 8 + 64)
        _check(_dm_tokenizer_decode(self._h, arr, len(ids), buf, len(buf)),
               "dm.Tokenizer.decode")
        return buf.value.decode()

    @property
    def vocab_size(self) -> int:
        return _dm_tokenizer_vocab_size(self._h)

    def token_text(self, token_id: int) -> str:
        length = c_uint32(0)
        ptr = _dm_tokenizer_token_text(self._h, c_uint32(token_id), ctypes.byref(length))
        if not ptr:
            return ""
        return ctypes.string_at(ptr, length.value).decode(errors="replace")

    @staticmethod
    def volt_run(corpus_path: str, min_size: int, max_size: int,
                 n_steps: int, output_path: str) -> None:
        _check(_dm_tokenizer_volt_run(_enc(corpus_path), min_size, max_size,
                                       n_steps, _enc(output_path)),
               "dm.Tokenizer.volt_run")


# ── § 5  Vision ───────────────────────────────────────────────────────────────

class Vision:
    """MobileNetV4-Tiny vision model."""

    def __init__(self, model_type: str = "mobilenet_tiny"):
        h = _dm_vision_create(_enc(model_type))
        if not h:
            raise RuntimeError("dm.Vision: create failed")
        self._h = h

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_vision_free(self._h)
            self._h = None

    def __enter__(self):  return self
    def __exit__(self, *_): self.__del__()

    def init_model(self, saved_model_dir: str, classes: int, image_size: int,
                   width_mult: float = 1.0, learning_rate: float = 1e-3) -> None:
        _check(_dm_vision_init(self._h, _enc(saved_model_dir),
                               classes, image_size, width_mult, learning_rate),
               "dm.Vision.init_model")

    def train(self, manifest_path: str, epochs: int, batch_size: int,
              lr: float = 1e-3) -> None:
        _check(_dm_vision_train(self._h, _enc(manifest_path), epochs, batch_size, lr),
               "dm.Vision.train")

    def eval(self, manifest_path: str) -> Tuple[float, float]:
        """Returns (loss, accuracy)."""
        loss = c_float(0.0)
        acc  = c_float(0.0)
        _check(_dm_vision_eval(self._h, _enc(manifest_path),
                               ctypes.byref(loss), ctypes.byref(acc)),
               "dm.Vision.eval")
        return loss.value, acc.value

    def predict(self, rgb: "array-like", h: int, w: int, n_classes: int) -> List[float]:
        """rgb must be a float32 sequence of length h*w*3 in [0,1]."""
        import array as _array
        if isinstance(rgb, (list, tuple)):
            flat = (c_float * len(rgb))(*rgb)
        else:
            flat = (c_float * (h * w * 3))(*rgb)
        probs = (c_float * n_classes)()
        _check(_dm_vision_predict(self._h, flat, h, w, probs, n_classes),
               "dm.Vision.predict")
        return list(probs)

    def save(self) -> None:
        _check(_dm_vision_save(self._h), "dm.Vision.save")


# ── § 6  Language Model ───────────────────────────────────────────────────────

class LM:
    """Tiny byte-level language model."""

    def __init__(self, model_type: str):
        """model_type: "tiny_transformer" | "tinystories" """
        h = _dm_lm_create(_enc(model_type))
        if not h:
            raise RuntimeError(f"dm.LM: create failed for type '{model_type}'")
        self._h = h

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_lm_free(self._h)
            self._h = None

    def __enter__(self):  return self
    def __exit__(self, *_): self.__del__()

    def train(self, corpus_path: str, checkpoint_dir: str,
              epochs: int, batch_size: int, lr: float = 1e-3) -> None:
        _check(_dm_lm_train(self._h, _enc(corpus_path), _enc(checkpoint_dir),
                            epochs, batch_size, lr),
               "dm.LM.train")

    def load(self, checkpoint_dir: str) -> None:
        _check(_dm_lm_load(self._h, _enc(checkpoint_dir)), "dm.LM.load")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        buf = ctypes.create_string_buffer(max_tokens * 4 + 256)
        _check(_dm_lm_generate(self._h, _enc(prompt), max_tokens, buf, len(buf)),
               "dm.LM.generate")
        return buf.value.decode()


# ── § 9  Benchmark ────────────────────────────────────────────────────────────

BENCH_LOAD  = 0
BENCH_ALGO  = 1
BENCH_WRITE = 2
BENCH_TOTAL = 3


class BenchReport:
    """Python view of DM_BenchReport."""
    def __init__(self, raw: _DM_BenchReport):
        self.phase_ms        = list(raw.phase_times_ms)
        self.peak_kb         = raw.peak_memory_kb
        self.user_ms         = raw.user_cpu_ms
        self.sys_ms          = raw.sys_cpu_ms
        self.result_ram      = raw.result_ram_bytes
        self.result_disk     = raw.result_disk_est_bytes
        self.num_patterns    = raw.num_patterns
        self.total_items     = raw.total_items
        self.throughput_mb_s = raw.throughput_mb_s

    def __repr__(self) -> str:
        return (f"BenchReport(total={self.phase_ms[3]:.1f}ms "
                f"patterns={self.num_patterns} "
                f"throughput={self.throughput_mb_s:.2f}MB/s)")


def bench_reset() -> None:               _dm_bench_reset()
def bench_start(phase: int) -> None:     _dm_bench_start(phase)
def bench_stop(phase: int) -> None:      _dm_bench_stop(phase)
def bench_record(n: int, t: int) -> None: _dm_bench_record(n, t)

def bench_report() -> BenchReport:
    return BenchReport(_dm_bench_get_report())

def bench_print(algo: str, dataset: str) -> None:
    _dm_bench_print(_enc(algo), _enc(dataset))


# ── § 10  BitSet ──────────────────────────────────────────────────────────────

class BitSet:
    """Compact bit array."""

    def __init__(self, n_bits: int):
        h = _dm_bitset_create(n_bits)
        if not h:
            raise MemoryError("dm.BitSet: allocation failed")
        self._h = h

    @classmethod
    def _from_handle(cls, h) -> "BitSet":
        obj = object.__new__(cls)
        obj._h = h
        return obj

    def copy(self) -> "BitSet":
        h = _dm_bitset_copy(self._h)
        if not h:
            raise MemoryError("dm.BitSet.copy: allocation failed")
        return BitSet._from_handle(h)

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_bitset_free(self._h)
            self._h = None

    def set(self, pos: int) -> None:          _dm_bitset_set    (self._h, pos)
    def clear(self, pos: int) -> None:        _dm_bitset_clear  (self._h, pos)
    def get(self, pos: int) -> bool:          return bool(_dm_bitset_get(self._h, pos))
    def set_all(self) -> None:                _dm_bitset_set_all(self._h)
    def flip(self) -> None:                   _dm_bitset_not    (self._h)
    def popcount(self) -> int:                return _dm_bitset_popcount(self._h)

    def __iand__(self, other: "BitSet") -> "BitSet":
        _dm_bitset_and(self._h, other._h); return self
    def __ior__(self, other: "BitSet") -> "BitSet":
        _dm_bitset_or(self._h, other._h);  return self
    def __getitem__(self, pos: int) -> bool:  return self.get(pos)
    def __setitem__(self, pos: int, v: bool):
        self.set(pos) if v else self.clear(pos)


# ── § 11  DataGen ─────────────────────────────────────────────────────────────

class DataGen:
    """Synthetic dataset / corpus generator."""

    def __init__(self, type: str):
        """type: "medm" | "textbook" """
        h = _dm_datagen_create(_enc(type))
        if not h:
            raise ValueError(f"dm.DataGen: unknown type '{type}'")
        self._h = h

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_datagen_free(self._h)
            self._h = None

    def __enter__(self):  return self
    def __exit__(self, *_): self.__del__()

    def run(self, spec: str, output_path: str, seed: int = 0) -> None:
        _check(_dm_datagen_run(self._h, _enc(spec), _enc(output_path), seed),
               "dm.DataGen.run")


# ── § 13  CLI ─────────────────────────────────────────────────────────────────

def cli_run(command: str, *args: str) -> int:
    """Run a dm CLI sub-command.  Returns the exit code."""
    argv = (c_char_p * (len(args) + 1))(*[_enc(a) for a in args], None)
    return _dm_cli_run(_enc(command), len(args), argv)


# ── § 14  GPU ─────────────────────────────────────────────────────────────────

class GpuCtx:
    """Vulkan compute context (soft-fails when no GPU present)."""

    def __init__(self, device_index: int = 0, shader_dir: Optional[str] = None):
        sd = _enc(shader_dir) if shader_dir else None
        self._h = _dm_gpu_create(device_index, sd)

    def __del__(self):
        if getattr(self, "_h", None):
            _dm_gpu_free(self._h)
            self._h = None

    @property
    def ready(self) -> bool:
        return bool(self._h and _dm_gpu_ready(self._h))

    @property
    def device_name(self) -> str:
        if not self._h:
            return "(no GPU)"
        buf = ctypes.create_string_buffer(256)
        _dm_gpu_device_name(self._h, buf, len(buf))
        return buf.value.decode()


# ── § 18  Experiment ─────────────────────────────────────────────────────────

def timer_now() -> float:
    """Return current wall-clock time in seconds."""
    return _dm_timer_now()

def peak_ram_mb() -> float:
    """Return peak resident set size in MB."""
    return _dm_peak_ram_mb()

def file_size_mb(path: str) -> float:
    return _dm_file_size_mb(_enc(path))

def dir_size_mb(path: str) -> float:
    return _dm_dir_size_mb(_enc(path))

def ensure_dir(path: str) -> None:
    if _dm_ensure_dir(_enc(path)) != 0:
        raise OSError(f"dm.ensure_dir: failed for '{path}'")

def path_basename(path: str) -> str:
    return _dm_path_basename(_enc(path)).decode()

def experiment_write_header(csv_path: str) -> None:
    _dm_experiment_write_header(_enc(csv_path))

def experiment_generate_report(results_root: str) -> None:
    _dm_experiment_generate_report(_enc(results_root))
