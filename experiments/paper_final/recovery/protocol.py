"""Prespecified recovery variants and historical stress-case selection."""

from dataclasses import asdict, dataclass

from experiments.paper_final.common.artifacts import ROOT, file_hash, read


@dataclass(frozen=True)
class RecoveryVariant:
    name: str
    label: str
    attempts: int
    unified: bool
    rl_recovery_cost: bool
    default_fallback: bool = False


VARIANTS = (
    RecoveryVariant("shared3_cost", "Shared 3 + recovery cost", 3, True, True, default_fallback=True),
    RecoveryVariant("shared3_no_cost", "Shared 3, no recovery cost", 3, True, False, default_fallback=True),
)
METHOD = "bandit_lstdq"
EXTRA_CANDIDATE_SEED_OFFSET = 600006


def recorded_variants(protocol):
    """Read each run's own budget, including historical three-plus-fallback runs."""
    return tuple(RecoveryVariant(**{
        **row,
        "default_fallback": row.get("default_fallback", protocol.get("schema_version", 1) < 2),
    }) for row in protocol["variants"])


def select_stress_case():
    """Rank completed diffusion-advection runs by joint-learner failures."""
    evidence = ROOT / "experiments/paper_final/reproduction/online/six_seeds.json"
    runs = [r for r in read(evidence)["runs"] if r["family"] == "diffusion_advection"]
    runs.sort(key=lambda r: (
        -r["windows"]["all_5000"][METHOD]["unrecovered_failure_count"],
        -r["unrecovered_failures"], r["seed"], r["name"],
    ))
    selected = runs[0]
    batch = "seeds_1_to_3" if selected["seed"] <= 3 else "seeds_4_to_6"
    config = evidence.parent / batch / selected["config"]
    if file_hash(config) != selected["config_sha256"]:
        raise ValueError("Selected input configuration differs from the accepted capture")
    ranking = [dict(
        name=r["name"], global_seed=r["seed"], base_seed=r["base_seed"],
        joint_unrecovered=r["windows"]["all_5000"][METHOD]["unrecovered_failure_count"],
        all_three_methods_unrecovered=r["unrecovered_failures"],
    ) for r in runs]
    return config, {
        "selection_rule": "Maximum full-stream LinUCB-LSTDQ failure count; ties use all-method count, then seed/name",
        "evidence": str(evidence.relative_to(ROOT)),
        "evidence_sha256": file_hash(evidence),
        "config": str(config.relative_to(ROOT)),
        "config_sha256": file_hash(config),
        "stream_sha256": selected["stream_sha256"],
        "ranking": ranking,
        "selected": ranking[0],
    }


def study_protocol():
    config, selection = select_stress_case()
    raw = read(config)
    return {
        "schema_version": 3,
        "module": "06_recovery",
        "exploratory": True,
        "selection": selection,
        "variants": [asdict(v) for v in VARIANTS],
        "method": METHOD,
        "cases": raw["stream"]["cases"],
        "problem": raw["problem"],
        "matrix_assembly": {
            "builder": "HYPRE GenerateDifConv",
            "advection_scheme": "upwind",
            "atype": 3,
            "scope": "Experiment-branch native wrapper; HYPRE source is unchanged",
            "pairing": "Same PDE coefficients, grid and RHS seeds as the forward run; assembled matrices differ",
        },
        "seeds": raw["seeds"],
        "solve_activation_problem": 1001,
        "failure_feedback": "Setup: immediate T on success, 2T on failure, retaining valid updates. RL: roll back when the whole case fails.",
        "retry_triggers": "setup failure, solve failure, nonfinite result, nonconvergence; execution errors abort",
        "learned_attempt_budget": 3,
        "default_fallback_budget": 1,
        "total_attempt_budget": 4,
        "failure_definition": "All three learned setup-solve attempts and the default fallback fail; the initial attempt is included in the learned budget",
        "fallback": "One fresh default setup and default solve after all three learned attempts fail",
        "rl_cost": "Native cycle time; the cost variant additionally receives all subsequent native setup/solve work on the same problem",
        "delayed_feedback": "Unified attempts provisionally learn their own cycle costs. After a later attempt succeeds, the cost variant restores the case snapshot and replays recorded transitions once with the realized remaining recovery costs. No native solve or action selection is replayed; replay time is charged. Failed cases roll back RL learning in all variants.",
        "candidate_pairing": "Original three-row candidate block per problem in both variants",
        "concurrency": "Two single-thread workers; macOS user-initiated QoS, no hard CPU affinity",
        "interpretation": "One deliberately failure-heavy seed/grid, selected from existing outcomes; exploratory comparison, not a new six-seed paper result",
    }
