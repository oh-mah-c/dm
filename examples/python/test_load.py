import os
import ctypes
path = "/home/autocookie/pomaieco/dm/libdm.so"
print(f"Trying to load {path}")
try:
    lib = ctypes.CDLL(path)
    print("Success:", lib)
except Exception as e:
    print("Error:", e)
