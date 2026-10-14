"""Execution and verification helpers for the official studies."""
from __future__ import annotations
import ctypes

import sys

def request_performance_qos():
    if sys.platform != "darwin":
        return {"requested": False, "reason": "non-macOS"}
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    function = library.pthread_set_qos_class_self_np
    function.argtypes = [ctypes.c_uint, ctypes.c_int]
    function.restype = ctypes.c_int
    status = int(function(0x19, 0))  # QOS_CLASS_USER_INITIATED, pthread/qos.h
    if status:
        raise OSError(status, "Cannot request user-initiated worker QoS")
    return {"requested": True, "class": "user_initiated", "hard_affinity": False}
