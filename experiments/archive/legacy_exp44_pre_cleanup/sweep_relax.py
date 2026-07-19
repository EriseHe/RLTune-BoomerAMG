import ctypes, os, numpy as np
import matplotlib.pyplot as plt

LIB = os.path.abspath("./libamg_env.dylib")
lib = ctypes.CDLL(LIB)
AMGEnv_p = ctypes.c_void_p

lib.amg_env_create.restype = AMGEnv_p
lib.amg_env_create.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int]
lib.amg_env_step.restype = ctypes.c_int
lib.amg_env_step.argtypes = [AMGEnv_p,ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int)]
lib.amg_env_get_r0.restype = ctypes.c_double
lib.amg_env_get_r0.argtypes = [AMGEnv_p]
lib.amg_env_destroy.argtypes = [AMGEnv_p]

def run_once(nx, stencil, rhs_type, w, tol=1e-8, max_cycles=30):
    env = lib.amg_env_create(nx,nx,nx, stencil, rhs_type, tol, max_cycles)
    r = ctypes.c_double()
    dt = ctypes.c_double()
    st = ctypes.c_int()
    cycles = 0
    while True:
        lib.amg_env_step(env, float(w), ctypes.byref(r), ctypes.byref(dt), ctypes.byref(st))
        cycles += 1
        if st.value != 0:
            break
        if cycles >= max_cycles:
            break
    lib.amg_env_destroy(env)
    return cycles, r.value

if __name__ == "__main__":
    nx = 40
    stencil = 27
    rhs_type = 1  # deterministic “random” in your C code

    ws = np.linspace(0.3, 1.9, 17)
    cycles_list = []
    final_r_list = []

    for w in ws:
        cycles, final_r = run_once(nx, stencil, rhs_type, w)
        print(f"w={w:.2f} cycles={cycles} final_r={final_r:.3e}")
        cycles_list.append(cycles)
        final_r_list.append(final_r)

    plt.figure()
    plt.plot(ws, cycles_list, marker="o")
    plt.xlabel("RelaxWt")
    plt.ylabel("cycles to stop (tol or max)")
    plt.title(f"Sweep: stencil={stencil}, n={nx}^3")
    plt.tight_layout()
    plt.show()
