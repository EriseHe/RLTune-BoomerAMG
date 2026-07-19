#include <stdio.h>

/* forward declarations */
typedef struct AMGEnv AMGEnv;
AMGEnv* amg_env_create(int nx,int ny,int nz,int stencil_type,int rhs_type,double tol,int max_cycles);
int amg_env_step(AMGEnv *env,double relax_weight,double *res_norm_out,double *dt_out,int *status_out);
double amg_env_get_r0(AMGEnv *env);
double amg_env_get_r(AMGEnv *env);
void amg_env_destroy(AMGEnv *env);

int main() {
    AMGEnv *env = amg_env_create(20,20,20, 7, 0, 1e-8, 20);
    printf("r0 = %.6e\n", amg_env_get_r0(env));

    for (int i=0;i<10;i++){
        double r, dt; int status;
        amg_env_step(env, 1.0, &r, &dt, &status);
        printf("step %d: r=%.6e  dt=%.6f  status=%d\n", i+1, r, dt, status);
        if (status!=0) break;
    }
    amg_env_destroy(env);
    return 0;
}
