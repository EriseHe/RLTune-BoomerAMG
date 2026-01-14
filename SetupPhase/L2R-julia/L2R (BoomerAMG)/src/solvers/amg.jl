# amg.jl
# AMG solver using HYPRE's BoomerAMG
# Two interfaces:
#   1) amg_solve: returns wall-clock time (ms)  — legacy
#   2) amg_work_units: returns AMG-native work units using V-cycles × AP-complexity
#
# Key mapping:
#   SOR:  sor(A, b, x0, ω, tol)              → iterations (cost)
#   AMG:  amg_work_units(A, b, θ, tol)       → work units (cost)

using HYPRE
using HYPRE.LibHYPRE
using SparseArrays
using Libdl

#=============================================================================
  TUNABLE PARAMETER CONFIGURATION
  
  Change TUNABLE_PARAM to switch what the bandit tunes:
    :strong_threshold  - AMG strength threshold (default, most impactful)
    :relax_wt          - Relaxation weight (only works with RelaxType 0,3,6)
=============================================================================#
const TUNABLE_PARAM = :strong_threshold

const DEFAULT_STRONG_THRESHOLD = 0.25
const DEFAULT_RELAX_WT = 1.0
const DEFAULT_RELAX_TYPE = 3  # Hybrid GS/Jacobi (uses RelaxWt if tuning that)

# Low-level hooks to fetch cumulative nnz of {A,P} (AP-complexity)
const libhypre = HYPRE.LibHYPRE.HYPRE_jll.libHYPRE_handle
const _sym_set_cumnnzap = Libdl.dlsym_e(libhypre, :HYPRE_BoomerAMGSetCumNnzAP)
const _sym_get_cumnnzap = Libdl.dlsym_e(libhypre, :HYPRE_BoomerAMGGetCumNnzAP)
const HAS_CUMNNZAP = _sym_set_cumnnzap !== nothing && _sym_get_cumnnzap !== nothing

function require_cumnnzap()
    HAS_CUMNNZAP || error("CumNnzAP symbols not found in libHYPRE; update HYPRE_jll to >= 3.0.0.")
end

function boomeramg_set_cumnnzap(solver)
    require_cumnnzap()
    ccall(_sym_set_cumnnzap, Cint, (HYPRE_Solver, Cdouble), solver, 1.0)
    return true
end

function boomeramg_get_cumnnzap(solver)
    require_cumnnzap()
    x = Ref{Cdouble}(0.0)
    ccall(_sym_get_cumnnzap, Cint, (HYPRE_Solver, Ref{Cdouble}), solver, x)
    return x[]
end

"""
    build_solver(param; maxiter=10000, tol=1e-8)

Internal helper to build a BoomerAMG solver with the active tunable parameter.
"""
function build_solver(param::Float64; maxiter::Int=10000, tol::Float64=1e-8)
    if TUNABLE_PARAM == :strong_threshold
        return HYPRE.BoomerAMG(;
            MaxIter=maxiter,
            Tol=tol,
            StrongThreshold=param,
            RelaxWt=DEFAULT_RELAX_WT,
            PrintLevel=0,
        )
    elseif TUNABLE_PARAM == :relax_wt
        return HYPRE.BoomerAMG(;
            MaxIter=maxiter,
            Tol=tol,
            StrongThreshold=DEFAULT_STRONG_THRESHOLD,
            RelaxType=DEFAULT_RELAX_TYPE,
            RelaxWt=param,
            PrintLevel=0,
        )
    else
        error("Unknown TUNABLE_PARAM: $TUNABLE_PARAM")
    end
end

"""
    amg_stats(A, b, param, tol; maxiter=10000) -> (num_cycles, cum_nnz_ap)

Runs BoomerAMG and returns:
  num_cycles  - V-cycle iterations (HYPRE_BoomerAMGGetNumIterations)
  cum_nnz_ap  - cumulative nnz of {A,P} hierarchy (HYPRE_BoomerAMGGetCumNnzAP)
"""
function amg_stats(A, b, param::Float64, tol::Float64; maxiter::Int=10000)
    A_sparse = issparse(A) ? A : sparse(A)
    HYPRE.Init()

    solver = build_solver(param; maxiter=maxiter, tol=tol)

    num_cycles = maxiter
    cum_nnz_ap = NaN
    try
        # Enable cumulative nnz tracking before setup/solve (if available)
        boomeramg_set_cumnnzap(solver)

        HYPRE.solve(solver, A_sparse, b)

        num_cycles = try
            HYPRE.GetNumIterations(solver)
        catch
            maxiter
        end

        cum_nnz_ap = try
            boomeramg_get_cumnnzap(solver)
        catch
            error("Failed to query CumNnzAP; ensure HYPRE_jll >= 3.0.0.")
        end
    catch e
        @warn "AMG solver failed: $e"
        rethrow(e)
    finally
        finalize(solver)
    end

    return num_cycles, cum_nnz_ap
end

"""
    amg_work_units(A, b, param, tol; maxiter=10000, nu_pre=1, nu_post=1, base_nnz=nothing) -> (wu, num_cycles, ap_complexity)

Solves Ax = b using BoomerAMG and returns a work-unit proxy:
    wu = num_cycles * (nu_pre + nu_post) * ap_complexity

where ap_complexity = cum_nnz_AP / base_nnz.

If base_nnz is not provided, nnz(A) is used.
"""
function amg_work_units(
    A,
    b,
    param::Float64,
    tol::Float64;
    maxiter::Int=10000,
    nu_pre::Int=1,
    nu_post::Int=1,
    base_nnz=nothing,
)
    A_sparse = issparse(A) ? A : sparse(A)
    base_nnz_val = isnothing(base_nnz) ? nnz(A_sparse) : base_nnz

    num_cycles, cum_nnz_ap = amg_stats(A_sparse, b, param, tol; maxiter=maxiter)

    if !isfinite(cum_nnz_ap) || cum_nnz_ap <= 0
        error("Invalid CumNnzAP value ($cum_nnz_ap); cannot compute AP complexity.")
    end
    ap_complexity = cum_nnz_ap / base_nnz_val

    wu = num_cycles * (nu_pre + nu_post) * ap_complexity
    return wu, num_cycles, ap_complexity
end

"""
    amg_solve(A, b, param, tol; maxiter=10000) -> Float64

Legacy wall-clock interface (ms). Prefer amg_work_units for bandit feedback.
"""
function amg_solve(A, b, param::Float64, tol::Float64; maxiter::Int=10000)
    A_sparse = issparse(A) ? A : sparse(A)
    HYPRE.Init()
    solver = build_solver(param; maxiter=maxiter, tol=tol)
    time_ms = Inf
    try
        time_ms = 1000.0 * @elapsed HYPRE.solve(solver, A_sparse, b)
    catch e
        @warn "AMG solver failed: $e"
    finally
        finalize(solver)
    end
    return time_ms
end

# Convenience versions
amg_solve(A, b, param) = amg_solve(A, b, param, 1e-8)
amg_work_units(A, b, param; maxiter=10000, nu_pre=1, nu_post=1, base_nnz=nothing) =
    amg_work_units(A, b, param, 1e-8; maxiter=maxiter, nu_pre=nu_pre, nu_post=nu_post, base_nnz=base_nnz)

# Helper to get parameter name for plot labels
get_tunable_param_name() = TUNABLE_PARAM == :strong_threshold ? "θ" : "w"
