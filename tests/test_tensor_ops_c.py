#!/usr/bin/env python3

import ctypes
import os
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class DMTensor(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("c", ctypes.c_int),
        ("h", ctypes.c_int),
        ("w", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_float))
    ]

def load_lib():
    env = os.environ.get("DM_LIB")
    if env:
        return ctypes.CDLL(env)
    path = str(ROOT / "libdm.so")
    return ctypes.CDLL(path)

_lib = load_lib()

# Define argtypes and restypes
_lib.dm_tensor_alloc.argtypes = [ctypes.POINTER(DMTensor), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dm_tensor_alloc.restype = ctypes.c_int

_lib.dm_tensor_free.argtypes = [ctypes.POINTER(DMTensor)]
_lib.dm_tensor_free.restype = None

_lib.dm_tensor_fill.argtypes = [ctypes.POINTER(DMTensor), ctypes.c_float]
_lib.dm_tensor_fill.restype = None

_lib.dm_op_relu.argtypes = [ctypes.POINTER(DMTensor)]
_lib.dm_op_relu.restype = None

_lib.dm_op_tensor_add.argtypes = [ctypes.POINTER(DMTensor), ctypes.POINTER(DMTensor)]
_lib.dm_op_tensor_add.restype = ctypes.c_int

_lib.dm_op_max_pool2d_same.argtypes = [ctypes.POINTER(DMTensor), ctypes.POINTER(DMTensor), ctypes.c_int, ctypes.c_int]
_lib.dm_op_max_pool2d_same.restype = ctypes.c_int

_lib.dm_op_batch_norm.argtypes = [
    ctypes.POINTER(DMTensor),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float
]
_lib.dm_op_batch_norm.restype = ctypes.c_int

def to_numpy(t: DMTensor) -> np.ndarray:
    shape = (t.n, t.c, t.h, t.w)
    size = t.n * t.c * t.h * t.w
    buf = ctypes.cast(t.data, ctypes.POINTER(ctypes.c_float * size))
    return np.frombuffer(buf.contents, dtype=np.float32).reshape(shape).copy()

def test_relu():
    t = DMTensor()
    assert _lib.dm_tensor_alloc(ctypes.byref(t), 1, 2, 2, 2) == 0
    
    # Fill with positive and negative values
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 4.0], dtype=np.float32)
    for i in range(8):
        t.data[i] = arr[i]
        
    _lib.dm_op_relu(ctypes.byref(t))
    
    out = to_numpy(t)
    expected = np.maximum(arr, 0.0).reshape(1, 2, 2, 2)
    assert np.allclose(out, expected)
    _lib.dm_tensor_free(ctypes.byref(t))
    print("test_relu passed")

def test_tensor_add():
    t1 = DMTensor()
    t2 = DMTensor()
    assert _lib.dm_tensor_alloc(ctypes.byref(t1), 1, 2, 2, 2) == 0
    assert _lib.dm_tensor_alloc(ctypes.byref(t2), 1, 2, 2, 2) == 0
    
    for i in range(8):
        t1.data[i] = float(i)
        t2.data[i] = 10.0 - float(i)
        
    assert _lib.dm_op_tensor_add(ctypes.byref(t1), ctypes.byref(t2)) == 0
    
    out = to_numpy(t1)
    expected = np.full((1, 2, 2, 2), 10.0, dtype=np.float32)
    assert np.allclose(out, expected)
    
    _lib.dm_tensor_free(ctypes.byref(t1))
    _lib.dm_tensor_free(ctypes.byref(t2))
    print("test_tensor_add passed")

def test_max_pool2d_same():
    t_in = DMTensor()
    t_out = DMTensor()
    assert _lib.dm_tensor_alloc(ctypes.byref(t_in), 1, 1, 4, 4) == 0
    
    # Fill with a sequence
    # 0  1  2  3
    # 4  5  6  7
    # 8  9  10 11
    # 12 13 14 15
    for i in range(16):
        t_in.data[i] = float(i)
        
    # Max Pool with kernel=2, stride=2
    # Expected output:
    # 5  7
    # 13 15
    assert _lib.dm_op_max_pool2d_same(ctypes.byref(t_in), ctypes.byref(t_out), 2, 2) == 0
    
    out = to_numpy(t_out)
    expected = np.array([[[[5.0, 7.0], [13.0, 15.0]]]], dtype=np.float32)
    assert np.allclose(out, expected)
    
    _lib.dm_tensor_free(ctypes.byref(t_in))
    _lib.dm_tensor_free(ctypes.byref(t_out))
    print("test_max_pool2d_same passed")

def test_batch_norm():
    t = DMTensor()
    assert _lib.dm_tensor_alloc(ctypes.byref(t), 1, 2, 2, 2) == 0
    
    # Fill channel 0 and 1
    # Channel 0: [1.0, 2.0, 3.0, 4.0]
    # Channel 1: [5.0, 6.0, 7.0, 8.0]
    for i in range(4):
        t.data[i] = float(i + 1)
        t.data[i + 4] = float(i + 5)
        
    gamma = (ctypes.c_float * 2)(2.0, 4.0)
    beta = (ctypes.c_float * 2)(1.0, 2.0)
    mean = (ctypes.c_float * 2)(2.5, 6.5)
    var = (ctypes.c_float * 2)(1.25, 1.25)
    eps = 1e-5
    
    assert _lib.dm_op_batch_norm(ctypes.byref(t), gamma, beta, mean, var, eps) == 0
    
    out = to_numpy(t)
    
    # Calculate expected mathematically
    # std = sqrt(1.25 + 1e-5)
    std = np.sqrt(1.25 + 1e-5)
    expected_c0 = (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) - 2.5) / std * 2.0 + 1.0
    expected_c1 = (np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32) - 6.5) / std * 4.0 + 2.0
    expected = np.concatenate([expected_c0, expected_c1]).reshape(1, 2, 2, 2)
    
    assert np.allclose(out, expected, rtol=1e-4)
    
    _lib.dm_tensor_free(ctypes.byref(t))
    print("test_batch_norm passed")

if __name__ == "__main__":
    test_relu()
    test_tensor_add()
    test_max_pool2d_same()
    test_batch_norm()
    print("All ResNet tensor operator C tests passed successfully!")
