#include <stdio.h>
#include "amg_runtime.h"

int main() {
    double setup_time = 0.0, r0 = 0.0;
    AMGRuntime *env = amg_runtime_create(
        20, 20, 20, 7, 0, 42ULL,
        1.0, 0.0, 1.0, 1.0, 1.0, 0.0);
    if (!env) return 1;
    if (amg_runtime_prepare(
            env,
            -1.0, -1, -1, -1.0, -1, -1, -1, -1,
            -1.0, -1, -1, -1, -1.0, -1, -1.0, -1, -1,
            &setup_time, &r0) != AMG_RUNTIME_OK) {
        amg_runtime_destroy(env);
        return 2;
    }
    printf("r0 = %.6e setup=%.6f\n", r0, setup_time);

    for (int i=0;i<10;i++){
        double r, dt; int status;
        int rc = amg_runtime_step(
            env, 1.0, 1, 1, -1, -1, -1, -1, -1, -1, -1,
            -1.0, -1.0, -1.0, -1, -1.0, -1,
            1e-8, 20, &r, &dt, &status);
        if (rc != AMG_RUNTIME_OK) {
            amg_runtime_destroy(env);
            return 3;
        }
        printf("step %d: r=%.6e  dt=%.6f  status=%d\n", i+1, r, dt, status);
        if (status!=0) break;
    }
    amg_runtime_destroy(env);
    return 0;
}
