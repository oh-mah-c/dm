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

# § 5b  TinyViT
_dm_tinyvit_weight_count = _fn("dm_tinyvit_weight_count", c_size_t, c_int, c_int, c_int)
_dm_tinyvit_forward      = _fn("dm_tinyvit_forward",      DM_Status,
                               c_int, ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                               c_int, c_int, c_int, ctypes.POINTER(c_float))
_dm_tinyvit_load         = _fn("dm_tinyvit_load",         DM_Status,
                               c_char_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int),
                               ctypes.POINTER(c_int), ctypes.POINTER(ctypes.POINTER(c_float)))
_dm_tinyvit_free_weights = _fn("dm_tinyvit_free_weights", None, ctypes.POINTER(c_float))
_dm_tinyvit_save_labels  = _fn("dm_tinyvit_save_labels",  DM_Status,
                               c_char_p, c_int, c_int, c_int,
                               ctypes.POINTER(c_uint32), ctypes.POINTER(c_float), ctypes.POINTER(c_uint32))
_dm_tinyvit_distill_loss = _fn("dm_tinyvit_distill_loss", DM_Status,
                               ctypes.POINTER(c_float), ctypes.POINTER(c_uint32), ctypes.POINTER(c_float),
                               c_int, c_int, c_float, ctypes.POINTER(c_float))


# § 5c  MobileNet Tiny
_dm_mobilenet_tiny_forward_raw = _fn("dm_mobilenet_tiny_forward_raw", DM_Status,
                                     ctypes.POINTER(c_float), c_int, c_int, ctypes.c_uint32,
                                     ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                                     ctypes.POINTER(c_float))
_dm_mobilenet_tiny_head_load   = _fn("dm_mobilenet_tiny_head_load",   DM_Status,
                                     c_char_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int),
                                     ctypes.POINTER(c_int), ctypes.POINTER(ctypes.c_uint32),
                                     ctypes.POINTER(ctypes.POINTER(c_float)),
                                     ctypes.POINTER(ctypes.POINTER(c_float)))
_dm_mobilenet_tiny_head_save   = _fn("dm_mobilenet_tiny_head_save",   DM_Status,
                                     c_char_p, c_int, c_int, c_int, ctypes.c_uint32,
                                     ctypes.POINTER(c_float), ctypes.POINTER(c_float))
_dm_mobilenet_tiny_head_free   = _fn("dm_mobilenet_tiny_head_free",   None,
                                     ctypes.POINTER(c_float), ctypes.POINTER(c_float))


# § 6b  BERT
_dm_bert_weight_count_raw    = _fn("dm_bert_weight_count_raw",    c_size_t, c_int, c_int, c_int)
_dm_bert_load_raw            = _fn("dm_bert_load_raw",            DM_Status,
                                   c_char_p, ctypes.POINTER(c_int), ctypes.POINTER(c_int),
                                   ctypes.POINTER(c_int), ctypes.POINTER(ctypes.POINTER(c_float)))
_dm_bert_free_weights        = _fn("dm_bert_free_weights",        None, ctypes.POINTER(c_float))
_dm_bert_forward_raw         = _fn("dm_bert_forward_raw",         DM_Status,
                                   c_int, c_int, c_int, ctypes.POINTER(c_float),
                                   ctypes.POINTER(c_int), ctypes.POINTER(c_int), c_int,
                                   ctypes.POINTER(c_float), ctypes.POINTER(c_float))
_dm_bert_forward_masked_raw  = _fn("dm_bert_forward_masked_raw",  DM_Status,
                                   c_int, c_int, c_int, ctypes.POINTER(c_float),
                                   ctypes.POINTER(c_int), ctypes.POINTER(c_int),
                                   ctypes.POINTER(c_int), c_int,
                                   ctypes.POINTER(c_float), ctypes.POINTER(c_float))


# § 7  LM
_dm_lm_create   = _fn("dm_lm_create",   DM_LM,     c_char_p)
_dm_lm_train    = _fn("dm_lm_train",    DM_Status, DM_LM, c_char_p, c_char_p, c_int, c_int, c_float)
_dm_lm_generate = _fn("dm_lm_generate", DM_Status, DM_LM, c_char_p, c_int, c_char_p, c_int)
_dm_lm_load     = _fn("dm_lm_load",     DM_Status, DM_LM, c_char_p)
_dm_lm_free     = _fn("dm_lm_free",     None,      DM_LM)

# ─────────────────────────────────────────────────────────────────────────────
# § 8  Engine — dm_engine ops (TFE backend, NCHW float32)
# ─────────────────────────────────────────────────────────────────────────────

class _DM_Tensor(ctypes.Structure):
    """Mirror of the C DM_Tensor struct."""
    _fields_ = [
        ("n",    c_int),
        ("c",    c_int),
        ("h",    c_int),
        ("w",    c_int),
        ("data", ctypes.POINTER(c_float)),
    ]

_DM_Tensor_p = ctypes.POINTER(_DM_Tensor)

# Tensor lifecycle
_dm_tensor_alloc = _fn("dm_tensor_alloc", DM_Status, _DM_Tensor_p, c_int, c_int, c_int, c_int)
_dm_tensor_free  = _fn("dm_tensor_free",  None,      _DM_Tensor_p)
_dm_tensor_fill  = _fn("dm_tensor_fill",  None,      _DM_Tensor_p, c_float)
_dm_tensor_get   = _fn("dm_tensor_get",   c_float,   _DM_Tensor_p, c_int, c_int, c_int, c_int)
_dm_tensor_set   = _fn("dm_tensor_set",   None,      _DM_Tensor_p, c_int, c_int, c_int, c_int, c_float)
_dm_tensor_count = _fn("dm_tensor_count", c_size_t,  _DM_Tensor_p)

# Convolutions
_dm_op_conv2d_same    = _fn("dm_op_conv2d_same",    DM_Status,
                             _DM_Tensor_p, _DM_Tensor_p,
                             ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                             c_int, c_int, c_int)
_dm_op_depthwise_conv = _fn("dm_op_depthwise_conv", DM_Status,
                             _DM_Tensor_p, _DM_Tensor_p,
                             ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                             c_int, c_int)
_dm_op_pointwise_conv = _fn("dm_op_pointwise_conv", DM_Status,
                             _DM_Tensor_p, _DM_Tensor_p,
                             ctypes.POINTER(c_float), ctypes.POINTER(c_float), c_int)

# Linear
_dm_op_linear = _fn("dm_op_linear", DM_Status,
                     _DM_Tensor_p, _DM_Tensor_p,
                     ctypes.POINTER(c_float), ctypes.POINTER(c_float), c_int)

# Pooling
_dm_op_global_avg_pool  = _fn("dm_op_global_avg_pool",  DM_Status, _DM_Tensor_p, _DM_Tensor_p)
_dm_op_max_pool2d_same  = _fn("dm_op_max_pool2d_same",  DM_Status, _DM_Tensor_p, _DM_Tensor_p, c_int, c_int)

# Normalisation
_dm_op_batch_norm  = _fn("dm_op_batch_norm", DM_Status,
                          _DM_Tensor_p,
                          ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                          ctypes.POINTER(c_float), ctypes.POINTER(c_float), c_float)
_dm_op_layer_norm  = _fn("dm_op_layer_norm", DM_Status,
                          ctypes.POINTER(c_float), c_int, c_int,
                          ctypes.POINTER(c_float), ctypes.POINTER(c_float), c_float)

# Elementwise
_dm_op_tensor_add = _fn("dm_op_tensor_add", DM_Status, _DM_Tensor_p, _DM_Tensor_p)

# Activations
_dm_op_relu    = _fn("dm_op_relu",    None, _DM_Tensor_p)
_dm_op_relu6   = _fn("dm_op_relu6",   None, _DM_Tensor_p)
_dm_op_tanh    = _fn("dm_op_tanh",    None, _DM_Tensor_p)
_dm_op_sigmoid = _fn("dm_op_sigmoid", None, _DM_Tensor_p)
_dm_op_gelu    = _fn("dm_op_gelu",    None, ctypes.POINTER(c_float), c_int)

# Softmax
_dm_op_softmax      = _fn("dm_op_softmax",      None, _DM_Tensor_p)
_dm_op_softmax_rows = _fn("dm_op_softmax_rows", None, ctypes.POINTER(c_float), c_int, c_int)

# Matrix multiplication
_dm_op_matmul_nt = _fn("dm_op_matmul_nt", None,
                         ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                         ctypes.POINTER(c_float), c_int, c_int, c_int)
_dm_op_matmul_nn = _fn("dm_op_matmul_nn", None,
                         ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                         ctypes.POINTER(c_float), c_int, c_int, c_int)

# Backward passes
_dm_op_linear_backward = _fn("dm_op_linear_backward", DM_Status,
                               _DM_Tensor_p, _DM_Tensor_p, _DM_Tensor_p,
                               ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                               ctypes.POINTER(c_float), c_int)
_dm_op_relu_backward   = _fn("dm_op_relu_backward", None,
                               _DM_Tensor_p, _DM_Tensor_p, _DM_Tensor_p)
_dm_op_tanh_backward   = _fn("dm_op_tanh_backward", None,
                               _DM_Tensor_p, _DM_Tensor_p, _DM_Tensor_p)

# Maxout
_dm_op_maxout          = _fn("dm_op_maxout",          DM_Status,
                               _DM_Tensor_p, _DM_Tensor_p, c_int,
                               ctypes.POINTER(c_int))
_dm_op_maxout_backward = _fn("dm_op_maxout_backward", DM_Status,
                               _DM_Tensor_p, _DM_Tensor_p, c_int,
                               ctypes.POINTER(c_int))

# Dropout
_dm_op_dropout          = _fn("dm_op_dropout",          None,
                                _DM_Tensor_p, _DM_Tensor_p, c_float,
                                ctypes.POINTER(c_int))
_dm_op_dropout_backward = _fn("dm_op_dropout_backward", None,
                                _DM_Tensor_p, _DM_Tensor_p, c_float,
                                ctypes.POINTER(c_int))

# Optimisers
_dm_op_adam_step = _fn("dm_op_adam_step", None,
                         ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                         ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                         c_int, c_float, c_float, c_float, c_float, c_float, c_int)
_dm_op_adagrad_step = _fn("dm_op_adagrad_step", None,
                            ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                            ctypes.POINTER(c_float),
                            c_int, c_float, c_float, c_float)
_dm_op_sgd_momentum_step = _fn("dm_op_sgd_momentum_step", None,
                                 ctypes.POINTER(c_float), ctypes.POINTER(c_float),
                                 ctypes.POINTER(c_float),
                                 c_int, c_float, c_float, c_float, c_int)

# § 9  VAE (was § 8 before engine section was added)
_dm_vae_create_raw      = _fn("dm_vae_create_raw", c_void_p, c_int, c_int, c_int, c_float)
_dm_vae_free_raw        = _fn("dm_vae_free_raw", None, c_void_p)
_dm_vae_train_step_raw  = _fn("dm_vae_train_step_raw", c_float, c_void_p, ctypes.POINTER(c_float), c_int)
_dm_vae_encode_raw      = _fn("dm_vae_encode_raw", None, c_void_p, ctypes.POINTER(c_float), c_int, ctypes.POINTER(c_float), ctypes.POINTER(c_float))
_dm_vae_decode_raw      = _fn("dm_vae_decode_raw", None, c_void_p, ctypes.POINTER(c_float), c_int, ctypes.POINTER(c_float))

# § 9  GAN
_dm_gan_create_raw      = _fn("dm_gan_create_raw", c_void_p, c_int, c_int, c_int, c_int, c_int, c_float, c_float, c_float, c_int)
_dm_gan_free_raw        = _fn("dm_gan_free_raw", None, c_void_p)
_dm_gan_generate_raw    = _fn("dm_gan_generate_raw", None, c_void_p, ctypes.POINTER(c_float), c_int, ctypes.POINTER(c_float))
_dm_gan_train_d_step_raw = _fn("dm_gan_train_d_step_raw", c_float, c_void_p, ctypes.POINTER(c_float), ctypes.POINTER(c_float), c_int)
_dm_gan_train_g_step_raw = _fn("dm_gan_train_g_step_raw", c_float, c_void_p, ctypes.POINTER(c_float), c_int)

# § 10  Benchmark
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
    """Language model wrapper, including BERT encoder inference."""

    def __init__(self, model_type: str):
        """model_type: "bert" | "tiny_transformer" | "tinystories" """
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
        """Generate text for causal LMs; for BERT, pass token IDs and receive [CLS] values."""
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


# ── § 5b  TinyViT ─────────────────────────────────────────────────────────────

TINYVIT_5M  = 0
TINYVIT_11M = 1
TINYVIT_21M = 2

class TinyViT:
    """Wraps the TinyViT C99 forward pass and utilities."""

    def __init__(self, variant: int = TINYVIT_21M, classes: int = 1000, img_size: int = 224):
        self.variant = variant
        self.classes = classes
        self.img_size = img_size
        self._weights_ptr = None
        self._weights_owner = False

    def __del__(self):
        self.free_weights()

    def free_weights(self):
        if self._weights_ptr and self._weights_owner:
            _dm_tinyvit_free_weights(self._weights_ptr)
            self._weights_ptr = None
            self._weights_owner = False

    @staticmethod
    def weight_count(variant: int, classes: int, img_size: int) -> int:
        return _dm_tinyvit_weight_count(variant, classes, img_size)

    def load_weights(self, weight_path: str):
        self.free_weights()
        var = c_int(0)
        cls = c_int(0)
        sz = c_int(0)
        ptr = ctypes.POINTER(c_float)()
        _check(_dm_tinyvit_load(_enc(weight_path), ctypes.byref(var), ctypes.byref(cls),
                               ctypes.byref(sz), ctypes.byref(ptr)),
               "dm.TinyViT.load")
        self.variant = var.value
        self.classes = cls.value
        self.img_size = sz.value
        self._weights_ptr = ptr
        self._weights_owner = True

    def forward(self, input_nhwc: Sequence[float], batch: int = 1) -> List[float]:
        if not self._weights_ptr:
            raise RuntimeError("No weights loaded. Call load_weights() or set weights first.")
        expected_len = batch * self.img_size * self.img_size * 3
        if len(input_nhwc) != expected_len:
            raise ValueError(f"Input size mismatch. Expected {expected_len} elements, got {len(input_nhwc)}")

        in_arr = (c_float * len(input_nhwc))(*input_nhwc)
        logits = (c_float * (batch * self.classes))()
        _check(_dm_tinyvit_forward(self.variant, self._weights_ptr, in_arr, batch,
                                  self.classes, self.img_size, logits),
               "dm.TinyViT.forward")
        return list(logits)

    @staticmethod
    def save_labels(out_path: str, num_images: int, num_classes: int, topK: int,
                    indices: Sequence[int], values: Sequence[float], aug_seeds: Sequence[int]) -> None:
        idx_arr = (c_uint32 * len(indices))(*indices)
        val_arr = (c_float * len(values))(*values)
        seed_arr = (c_uint32 * len(aug_seeds))(*aug_seeds)
        _check(_dm_tinyvit_save_labels(_enc(out_path), num_images, num_classes, topK,
                                      idx_arr, val_arr, seed_arr),
               "dm.TinyViT.save_labels")

    @staticmethod
    def distill_loss(student_logits: Sequence[float], indices: Sequence[int],
                     teacher_values: Sequence[float], K: int, C: int, temperature: float) -> float:
        s_logits = (c_float * len(student_logits))(*student_logits)
        idx_arr = (c_uint32 * len(indices))(*indices)
        t_vals = (c_float * len(teacher_values))(*teacher_values)
        loss = c_float(0.0)
        _check(_dm_tinyvit_distill_loss(s_logits, idx_arr, t_vals, K, C, temperature, ctypes.byref(loss)),
               "dm.TinyViT.distill_loss")
        return loss.value


# ── § 5c  MobileNet Tiny ──────────────────────────────────────────────────────

class MobileNetTiny:
    """Wraps MobileNetV4-Tiny FFI bindings for inference and head checkpoint management."""

    def __init__(self, image_size: int = 224, classes: int = 1000, seed: int = 1337):
        self.image_size = image_size
        self.classes = classes
        self.seed = seed
        self._head_w = None
        self._head_b = None
        self._feature_dim = None

    def __del__(self):
        self.free_head()

    def free_head(self):
        if self._head_w or self._head_b:
            _dm_mobilenet_tiny_head_free(self._head_w, self._head_b)
            self._head_w = None
            self._head_b = None
            self._feature_dim = None

    def load_head(self, path: str):
        self.free_head()
        classes = c_int(0)
        feature_dim = c_int(0)
        image_size = c_int(0)
        seed = ctypes.c_uint32(0)
        w_ptr = ctypes.POINTER(c_float)()
        b_ptr = ctypes.POINTER(c_float)()

        _check(_dm_mobilenet_tiny_head_load(
            _enc(path),
            ctypes.byref(classes),
            ctypes.byref(feature_dim),
            ctypes.byref(image_size),
            ctypes.byref(seed),
            ctypes.byref(w_ptr),
            ctypes.byref(b_ptr)
        ), "dm.MobileNetTiny.load_head")

        self.classes = classes.value
        self._feature_dim = feature_dim.value
        self.image_size = image_size.value
        self.seed = seed.value
        self._head_w = w_ptr
        self._head_b = b_ptr

    def save_head(self, path: str):
        if not self._head_w or not self._head_b or not self._feature_dim:
            raise RuntimeError("No head weights loaded to save.")
        _check(_dm_mobilenet_tiny_head_save(
            _enc(path),
            self.classes,
            self._feature_dim,
            self.image_size,
            self.seed,
            self._head_w,
            self._head_b
        ), "dm.MobileNetTiny.save_head")

    def forward(self, input_nchw: Sequence[float]) -> List[float]:
        expected_len = 3 * self.image_size * self.image_size
        if len(input_nchw) != expected_len:
            raise ValueError(f"Input size mismatch. Expected {expected_len} elements, got {len(input_nchw)}")

        in_arr = (c_float * expected_len)(*input_nchw)
        logits = (c_float * self.classes)()
        _check(_dm_mobilenet_tiny_forward_raw(
            in_arr,
            self.image_size,
            self.classes,
            self.seed,
            self._head_w,
            self._head_b,
            logits
        ), "dm.MobileNetTiny.forward")
        return list(logits)


# ── § 6b  BERT ────────────────────────────────────────────────────────────────

BERT_BASE = 0
BERT_LARGE = 1

class BERT:
    """Wraps BERT FFI bindings for weights loading and sequence encoding."""

    def __init__(self, variant: int = BERT_BASE, vocab_size: int = 30522, max_seq_len: int = 512):
        self.variant = variant
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self._weights_ptr = None
        self._weights_owner = False

    def __del__(self):
        self.free_weights()

    def free_weights(self):
        if self._weights_ptr and self._weights_owner:
            _dm_bert_free_weights(self._weights_ptr)
            self._weights_ptr = None
            self._weights_owner = False

    @staticmethod
    def weight_count(variant: int, vocab_size: int, max_seq_len: int) -> int:
        return _dm_bert_weight_count_raw(variant, vocab_size, max_seq_len)

    def load_weights(self, path: str):
        self.free_weights()
        var = c_int(0)
        voc = c_int(0)
        seq = c_int(0)
        ptr = ctypes.POINTER(c_float)()

        _check(_dm_bert_load_raw(
            _enc(path),
            ctypes.byref(var),
            ctypes.byref(voc),
            ctypes.byref(seq),
            ctypes.byref(ptr)
        ), "dm.BERT.load_weights")

        self.variant = var.value
        self.vocab_size = voc.value
        self.max_seq_len = seq.value
        self._weights_ptr = ptr
        self._weights_owner = True

    def forward(self, token_ids: Sequence[int], segment_ids: Sequence[int], attention_mask: Optional[Sequence[int]] = None) -> Tuple[List[float], List[float]]:
        if not self._weights_ptr:
            raise RuntimeError("No weights loaded. Call load_weights() first.")

        seq = len(token_ids)
        if seq <= 0 or seq > self.max_seq_len:
            raise ValueError(f"Sequence length must be between 1 and {self.max_seq_len}, got {seq}")

        if len(segment_ids) != seq:
            raise ValueError("segment_ids length must match token_ids")

        hidden_dim = 768 if self.variant == BERT_BASE else 1024
        hidden_out = (c_float * (seq * hidden_dim))()
        cls_out = (c_float * hidden_dim)()

        tok_arr = (c_int * seq)(*token_ids)
        seg_arr = (c_int * seq)(*segment_ids)

        if attention_mask is not None:
            if len(attention_mask) != seq:
                raise ValueError("attention_mask length must match token_ids")
            att_arr = (c_int * seq)(*attention_mask)
            _check(_dm_bert_forward_masked_raw(
                self.variant,
                self.vocab_size,
                self.max_seq_len,
                self._weights_ptr,
                tok_arr,
                seg_arr,
                att_arr,
                seq,
                hidden_out,
                cls_out
            ), "dm.BERT.forward")
        else:
            _check(_dm_bert_forward_raw(
                self.variant,
                self.vocab_size,
                self.max_seq_len,
                self._weights_ptr,
                tok_arr,
                seg_arr,
                seq,
                hidden_out,
                cls_out
            ), "dm.BERT.forward")

        return list(hidden_out), list(cls_out)


# ── VAE ───────────────────────────────────────────────────────────────────────

class VAE:
    """Auto-Encoding Variational Bayes (VAE) with Adam optimizer."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, lr: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self._handle = _dm_vae_create_raw(input_dim, hidden_dim, latent_dim, lr)
        if not self._handle:
            raise MemoryError("Failed to allocate VAE model")

    def __del__(self):
        if getattr(self, "_handle", None):
            _dm_vae_free_raw(self._handle)
            self._handle = None

    def train_step(self, x_batch: List[float]) -> float:
        """Run one training step with a flat batch of input vectors. Returns loss."""
        batch_size = len(x_batch) // self.input_dim
        if batch_size * self.input_dim != len(x_batch):
            raise ValueError("Batch size must be a multiple of input_dim")
        x_arr = (c_float * len(x_batch))(*x_batch)
        return float(_dm_vae_train_step_raw(self._handle, x_arr, batch_size))

    def encode(self, x_batch: List[float]) -> Tuple[List[float], List[float]]:
        """Encode input vectors into mean and logvar."""
        batch_size = len(x_batch) // self.input_dim
        if batch_size * self.input_dim != len(x_batch):
            raise ValueError("Batch size must be a multiple of input_dim")
        x_arr = (c_float * len(x_batch))(*x_batch)
        mean_out = (c_float * (batch_size * self.latent_dim))()
        logvar_out = (c_float * (batch_size * self.latent_dim))()
        _dm_vae_encode_raw(self._handle, x_arr, batch_size, mean_out, logvar_out)
        return list(mean_out), list(logvar_out)

    def decode(self, z_batch: List[float]) -> List[float]:
        """Decode latent vectors back into the input space."""
        batch_size = len(z_batch) // self.latent_dim
        if batch_size * self.latent_dim != len(z_batch):
            raise ValueError("Batch size must be a multiple of latent_dim")
        z_arr = (c_float * len(z_batch))(*z_batch)
        out = (c_float * (batch_size * self.input_dim))()
        _dm_vae_decode_raw(self._handle, z_arr, batch_size, out)
        return list(out)


# ── GAN ───────────────────────────────────────────────────────────────────────

class GAN:
    """Generative Adversarial Network (GAN) with SGD Momentum / Nesterov optimizer."""

    def __init__(self, input_dim: int, g_hidden: int, noise_dim: int, d_hidden: int, maxout_k: int = 5, drop_prob: float = 0.5, lr: float = 0.01, momentum: float = 0.9, nesterov: bool = True):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self._handle = _dm_gan_create_raw(input_dim, g_hidden, noise_dim, d_hidden, maxout_k, drop_prob, lr, momentum, 1 if nesterov else 0)
        if not self._handle:
            raise MemoryError("Failed to allocate GAN model")

    def __del__(self):
        if getattr(self, "_handle", None):
            _dm_gan_free_raw(self._handle)
            self._handle = None

    def generate(self, z_batch: List[float]) -> List[float]:
        """Generate fake samples from latent noise vectors z."""
        batch_size = len(z_batch) // self.noise_dim
        if batch_size * self.noise_dim != len(z_batch):
            raise ValueError("Batch size must be a multiple of noise_dim")
        z_arr = (c_float * len(z_batch))(*z_batch)
        out = (c_float * (batch_size * self.input_dim))()
        _dm_gan_generate_raw(self._handle, z_arr, batch_size, out)
        return list(out)

    def train_d_step(self, real_x_batch: List[float], z_batch: List[float]) -> float:
        """Run one training step for the Discriminator. Returns D loss."""
        batch_size = len(real_x_batch) // self.input_dim
        if batch_size * self.input_dim != len(real_x_batch) or batch_size * self.noise_dim != len(z_batch):
            raise ValueError("Invalid batch dimensions for real_x or z")
        x_arr = (c_float * len(real_x_batch))(*real_x_batch)
        z_arr = (c_float * len(z_batch))(*z_batch)
        return float(_dm_gan_train_d_step_raw(self._handle, x_arr, z_arr, batch_size))

    def train_g_step(self, z_batch: List[float]) -> float:
        """Run one training step for the Generator. Returns G loss."""
        batch_size = len(z_batch) // self.noise_dim
        if batch_size * self.noise_dim != len(z_batch):
            raise ValueError("Batch size must be a multiple of noise_dim")
        z_arr = (c_float * len(z_batch))(*z_batch)
        return float(_dm_gan_train_g_step_raw(self._handle, z_arr, batch_size))


# ─────────────────────────────────────────────────────────────────────────────
# § 8  Engine — Python wrappers
#
# Users can build custom models by composing Tensor + op.* primitives,
# exactly like TensorFlow layers but backed by dm_engine (TFE).
#
# Quick example:
#   import dm
#   x = dm.Tensor(1, 3, 224, 224)   # NCHW batch=1, RGB 224×224
#   y = dm.Tensor(1, 64, 112, 112)
#   dm.op.conv2d_same(x, y, weights, bias, 64, 3, 2)
#   dm.op.relu(y)
# ─────────────────────────────────────────────────────────────────────────────

import array as _array
import types as _types


class Tensor:
    """
    NCHW float32 tensor backed by dm_engine.

    Layout: (n, c, h, w) — row-major, contiguous float32.
    The underlying C buffer is owned by this object.
    """

    def __init__(self, n: int, c: int, h: int, w: int, *, fill: float = 0.0):
        self._t = _DM_Tensor()
        _check(_dm_tensor_alloc(ctypes.byref(self._t), n, c, h, w),
               "dm.Tensor")
        if fill != 0.0:
            _dm_tensor_fill(ctypes.byref(self._t), fill)

    def __del__(self):
        if getattr(self, "_t", None) and self._t.data:
            _dm_tensor_free(ctypes.byref(self._t))

    def __enter__(self):  return self
    def __exit__(self, *_): self.__del__()

    @property
    def n(self) -> int:  return self._t.n
    @property
    def c(self) -> int:  return self._t.c
    @property
    def h(self) -> int:  return self._t.h
    @property
    def w(self) -> int:  return self._t.w
    @property
    def count(self) -> int: return int(_dm_tensor_count(ctypes.byref(self._t)))

    def fill(self, value: float) -> None:
        _dm_tensor_fill(ctypes.byref(self._t), value)

    def get(self, n: int, c: int, y: int, x: int) -> float:
        return float(_dm_tensor_get(ctypes.byref(self._t), n, c, y, x))

    def set(self, n: int, c: int, y: int, x: int, v: float) -> None:
        _dm_tensor_set(ctypes.byref(self._t), n, c, y, x, v)

    def to_list(self) -> List[float]:
        """Return a flat copy of the data buffer as a Python list."""
        n = self.count
        return list(self._t.data[0:n])

    def from_list(self, values: List[float]) -> None:
        """Overwrite the tensor data from a flat Python list."""
        n = self.count
        if len(values) != n:
            raise ValueError(f"Expected {n} values, got {len(values)}")
        for i, v in enumerate(values):
            self._t.data[i] = v

    def _ptr(self):
        """Internal: return ctypes pointer to the raw DM_Tensor struct."""
        return ctypes.byref(self._t)

    def __repr__(self) -> str:
        return f"dm.Tensor(n={self.n}, c={self.c}, h={self.h}, w={self.w})"


def _fp(values):
    """Convert a list/bytes/array to a ctypes float pointer (no copy on array.array)."""
    if isinstance(values, ctypes.Array):
        return values
    arr = (c_float * len(values))(*values)
    return arr


def _ip(values):
    """Convert a list to a ctypes int pointer."""
    if isinstance(values, ctypes.Array):
        return values
    return (c_int * len(values))(*values)


# ── op namespace ──────────────────────────────────────────────────────────────

class op:
    """
    Neural-op primitives — the dm_engine public API.

    All ops dispatch through TFE so they inherit XLA/cuDNN/oneDNN acceleration.
    Functions that operate on Tensor objects mutate `out` in-place.
    """

    @staticmethod
    def conv2d_same(in_: Tensor, out: Tensor,
                    w: List[float], b: List[float],
                    out_c: int, kernel: int, stride: int) -> None:
        """Standard conv2d, SAME padding.  w: OIHW [out_c][in_c][ky][kx]."""
        _check(_dm_op_conv2d_same(in_._ptr(), out._ptr(),
                                   _fp(w), _fp(b), out_c, kernel, stride),
               "dm.op.conv2d_same")

    @staticmethod
    def depthwise_conv(in_: Tensor, out: Tensor,
                       w: List[float], b: List[float],
                       kernel: int, stride: int) -> None:
        """Depthwise separable conv, SAME.  w: [c][ky][kx]."""
        _check(_dm_op_depthwise_conv(in_._ptr(), out._ptr(),
                                      _fp(w), _fp(b), kernel, stride),
               "dm.op.depthwise_conv")

    @staticmethod
    def pointwise_conv(in_: Tensor, out: Tensor,
                       w: List[float], b: List[float], out_c: int) -> None:
        """1×1 conv.  w: [out_c][in_c]."""
        _check(_dm_op_pointwise_conv(in_._ptr(), out._ptr(),
                                      _fp(w), _fp(b), out_c),
               "dm.op.pointwise_conv")

    @staticmethod
    def linear(in_: Tensor, out: Tensor,
               w: List[float], b: List[float], out_c: int) -> None:
        """Fully-connected.  in: [n,in_c,1,1] → out: [n,out_c,1,1]."""
        _check(_dm_op_linear(in_._ptr(), out._ptr(),
                              _fp(w), _fp(b), out_c),
               "dm.op.linear")

    @staticmethod
    def global_avg_pool(in_: Tensor, out: Tensor) -> None:
        _check(_dm_op_global_avg_pool(in_._ptr(), out._ptr()),
               "dm.op.global_avg_pool")

    @staticmethod
    def max_pool2d_same(in_: Tensor, out: Tensor,
                        kernel: int, stride: int) -> None:
        _check(_dm_op_max_pool2d_same(in_._ptr(), out._ptr(), kernel, stride),
               "dm.op.max_pool2d_same")

    @staticmethod
    def batch_norm(t: Tensor,
                   gamma: List[float], beta: List[float],
                   mean: List[float],  var: List[float],
                   eps: float = 1e-5) -> None:
        _check(_dm_op_batch_norm(t._ptr(),
                                  _fp(gamma), _fp(beta), _fp(mean), _fp(var), eps),
               "dm.op.batch_norm")

    @staticmethod
    def layer_norm(x: List[float], seq_len: int, d_model: int,
                   gamma: List[float], beta: List[float],
                   eps: float = 1e-5) -> List[float]:
        """Layer norm in-place on a flat float list [seq_len × d_model].  Returns updated list."""
        arr = _fp(x)
        _check(_dm_op_layer_norm(arr, seq_len, d_model,
                                  _fp(gamma), _fp(beta), eps),
               "dm.op.layer_norm")
        return list(arr)

    @staticmethod
    def add(out: Tensor, in_: Tensor) -> None:
        _check(_dm_op_tensor_add(out._ptr(), in_._ptr()), "dm.op.add")

    @staticmethod
    def relu(t: Tensor) -> None:    _dm_op_relu(t._ptr())
    @staticmethod
    def relu6(t: Tensor) -> None:   _dm_op_relu6(t._ptr())
    @staticmethod
    def tanh(t: Tensor) -> None:    _dm_op_tanh(t._ptr())
    @staticmethod
    def sigmoid(t: Tensor) -> None: _dm_op_sigmoid(t._ptr())

    @staticmethod
    def gelu(x: List[float]) -> List[float]:
        """GELU in-place on a flat float list.  Returns updated list."""
        arr = _fp(x)
        _dm_op_gelu(arr, len(x))
        return list(arr)

    @staticmethod
    def softmax(t: Tensor) -> None:
        """Softmax over the channel dim of an [n,c,1,1] tensor."""
        _dm_op_softmax(t._ptr())

    @staticmethod
    def softmax_rows(x: List[float], rows: int, cols: int) -> List[float]:
        """Softmax over rows of a flat [rows × cols] buffer.  Returns updated list."""
        arr = _fp(x)
        _dm_op_softmax_rows(arr, rows, cols)
        return list(arr)

    @staticmethod
    def matmul_nt(A: List[float], B: List[float],
                  M: int, N: int, K: int) -> List[float]:
        """C = A × Bᵀ   A[M×K], B[N×K] → C[M×N]."""
        C = (c_float * (M * N))()
        _dm_op_matmul_nt(_fp(A), _fp(B), C, M, N, K)
        return list(C)

    @staticmethod
    def matmul_nn(A: List[float], B: List[float],
                  M: int, K: int, N: int) -> List[float]:
        """C = A × B    A[M×K], B[K×N] → C[M×N]."""
        C = (c_float * (M * N))()
        _dm_op_matmul_nn(_fp(A), _fp(B), C, M, K, N)
        return list(C)

    @staticmethod
    def linear_backward(in_: Tensor, grad_out: Tensor, grad_in: Tensor,
                         grad_w: List[float], grad_b: List[float],
                         w: List[float], out_c: int) -> Tuple["List[float]", "List[float]"]:
        """Returns (updated_grad_w, updated_grad_b)."""
        gw = _fp(grad_w)
        gb = _fp(grad_b)
        _check(_dm_op_linear_backward(in_._ptr(), grad_out._ptr(), grad_in._ptr(),
                                       gw, gb, _fp(w), out_c),
               "dm.op.linear_backward")
        return list(gw), list(gb)

    @staticmethod
    def relu_backward(in_: Tensor, grad_out: Tensor, grad_in: Tensor) -> None:
        _dm_op_relu_backward(in_._ptr(), grad_out._ptr(), grad_in._ptr())

    @staticmethod
    def tanh_backward(out: Tensor, grad_out: Tensor, grad_in: Tensor) -> None:
        _dm_op_tanh_backward(out._ptr(), grad_out._ptr(), grad_in._ptr())

    @staticmethod
    def maxout(in_: Tensor, out: Tensor, k: int) -> List[int]:
        """Returns argmax buffer (int[n×c])."""
        argmax = (c_int * (in_.n * (in_.c // k)))()
        _check(_dm_op_maxout(in_._ptr(), out._ptr(), k, argmax), "dm.op.maxout")
        return list(argmax)

    @staticmethod
    def maxout_backward(grad_out: Tensor, grad_in: Tensor,
                         k: int, argmax: List[int]) -> None:
        _check(_dm_op_maxout_backward(grad_out._ptr(), grad_in._ptr(),
                                       k, _ip(argmax)),
               "dm.op.maxout_backward")

    @staticmethod
    def dropout(in_: Tensor, out: Tensor,
                drop_prob: float) -> List[int]:
        """Returns mask buffer (int[n×c×h×w])."""
        mask = (c_int * in_.count)()
        _dm_op_dropout(in_._ptr(), out._ptr(), drop_prob, mask)
        return list(mask)

    @staticmethod
    def dropout_backward(grad_out: Tensor, grad_in: Tensor,
                          drop_prob: float, mask: List[int]) -> None:
        _dm_op_dropout_backward(grad_out._ptr(), grad_in._ptr(),
                                 drop_prob, _ip(mask))

    @staticmethod
    def adam_step(param: List[float], grad: List[float],
                  m: List[float], v: List[float],
                  lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8, weight_decay: float = 0.0,
                  t: int = 1) -> Tuple["List[float]", "List[float]", "List[float]"]:
        """In-place Adam step.  Returns (updated_param, updated_m, updated_v)."""
        p = _fp(param); g = _fp(grad); mv = _fp(m); vv = _fp(v)
        _dm_op_adam_step(p, g, mv, vv, len(param), lr, beta1, beta2, eps,
                          weight_decay, t)
        return list(p), list(mv), list(vv)

    @staticmethod
    def adagrad_step(param: List[float], grad: List[float], g_sum: List[float],
                     lr: float = 1e-2, eps: float = 1e-8,
                     weight_decay: float = 0.0) -> Tuple["List[float]", "List[float]"]:
        """In-place Adagrad step.  Returns (updated_param, updated_g_sum)."""
        p = _fp(param); g = _fp(grad); gs = _fp(g_sum)
        _dm_op_adagrad_step(p, g, gs, len(param), lr, eps, weight_decay)
        return list(p), list(gs)

    @staticmethod
    def sgd_momentum_step(param: List[float], grad: List[float],
                           velocity: List[float],
                           lr: float = 1e-2, momentum: float = 0.9,
                           weight_decay: float = 0.0,
                           nesterov: bool = False) -> Tuple["List[float]", "List[float]"]:
        """In-place SGD+momentum step.  Returns (updated_param, updated_velocity)."""
        p = _fp(param); g = _fp(grad); vel = _fp(velocity)
        _dm_op_sgd_momentum_step(p, g, vel, len(param), lr, momentum,
                                  weight_decay, int(nesterov))
        return list(p), list(vel)
