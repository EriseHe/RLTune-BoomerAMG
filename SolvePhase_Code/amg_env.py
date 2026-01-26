import ctypes, os

LIB = os.path.abspath("./libamg_env.dylib")
lib = ctypes.CDLL(LIB)

AMGEnv_p = ctypes.c_void_p

lib.amg_env_create.restype = AMGEnv_p
lib.amg_env_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_int,
    ctypes.c_ulonglong,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]


lib.amg_env_step.restype = ctypes.c_int
lib.amg_env_step.argtypes = [
    AMGEnv_p,
    ctypes.c_double,   # relax_weight
    ctypes.c_int,      # sweeps_down
    ctypes.c_int,      # sweeps_up
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int)
]

lib.amg_env_get_r0.restype = ctypes.c_double
lib.amg_env_get_r0.argtypes = [AMGEnv_p]

lib.amg_env_get_r.restype = ctypes.c_double
lib.amg_env_get_r.argtypes = [AMGEnv_p]

lib.amg_env_destroy.restype = None
lib.amg_env_destroy.argtypes = [AMGEnv_p]


env = lib.amg_env_create(
    20,20,20,
    7, 0,
    1e-8, 30,
    1.0, 0.0,               # k, c (7pt)
    26.0, -4.0, -0.15, -0.0125  # a0..a3 (dummy for 7pt)
)

print("r0 =", lib.amg_env_get_r0(env))

for i in range(10):
    r = ctypes.c_double()
    dt = ctypes.c_double()
    st = ctypes.c_int()
    lib.amg_env_step(env, 1.0, 2, 2, ctypes.byref(r), ctypes.byref(dt), ctypes.byref(st))
    print(i+1, "r=", r.value, "dt=", dt.value, "status=", st.value)
    if st.value != 0:
        break

lib.amg_env_destroy(env)
