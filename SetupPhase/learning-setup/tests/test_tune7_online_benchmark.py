from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np


LEARNING_SETUP_ROOT = Path(__file__).resolve().parents[1]
BENCH_HELPER_MODULE = "utils.tune7_online_benchmark"
BENCH_SCRIPT_PATH = LEARNING_SETUP_ROOT / "scripts" / "benchmark_tune7_online.py"

if str(LEARNING_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(LEARNING_SETUP_ROOT))

from learners.HierarchicalLinUCB_AMG_v5 import HierarchicalLinUCB_AMG_v5
from learners.HierarchicalSquareCB_AMG_v4 import HierarchicalSquareCB_AMG_v4
from learners.NystromKernelUCB_AMG_v4 import NystromKernelUCB_AMG_v4
from learners.SharedLinTS_AMG_v4 import SharedLinTS_AMG_v4
from learners.SquareCB_AMG_v4 import SquareCB_AMG_v4
from learners._candidate_subset import Tune7CandidateMixer, Tune7LatticeCatalog
from learners._tune7_online_features import Tune7FeatureBuilder


def _make_solver_stub():
    solver_stub = types.ModuleType("solver")

    class _Result:
        def __init__(self, runtime_sec: float, residual_norm: float, iterations: int):
            self.runtime_sec = float(runtime_sec)
            self.residual_norm = float(residual_norm)
            self.iterations = int(iterations)

    def _solve(*, params, tol=1e-8, max_iter=10000, **mkw):
        nx = int(mkw.get("nx", 1))
        ny = int(mkw.get("ny", 1))
        nz = int(mkw.get("nz", 1))
        base = 1e-3 * (nx + ny + nz)
        runtime = (
            base
            + 0.05 * float(params["strong_threshold"])
            + 0.02 * float(params["max_row_sum"])
            + 0.03 * float(params["trunc_factor"])
            + 1e-3 * float(params["P_max_elmts"])
            + 2e-3 * float(params["agg_num_levels"])
            + 5e-4 * float(params["coarsen_type"])
            + 7e-4 * float(params["interp_type"])
        )
        return _Result(runtime_sec=runtime, residual_norm=float(tol) * 0.1, iterations=min(25, int(max_iter) - 1))

    solver_stub.solve = _solve
    solver_stub.TUNABLE_PARAMS = ()
    return solver_stub


def _load_benchmark_helper():
    previous_solver = sys.modules.get("solver")
    sys.modules["solver"] = _make_solver_stub()
    try:
        sys.modules.pop(BENCH_HELPER_MODULE, None)
        return importlib.import_module(BENCH_HELPER_MODULE)
    finally:
        if previous_solver is not None:
            sys.modules["solver"] = previous_solver
        else:
            sys.modules.pop("solver", None)


def _load_benchmark_script():
    module_name = "_benchmark_tune7_online_unit"
    previous_solver = sys.modules.get("solver")
    sys.modules["solver"] = _make_solver_stub()
    sys.modules.pop(module_name, None)
    sys.modules.pop(BENCH_HELPER_MODULE, None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, BENCH_SCRIPT_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {BENCH_SCRIPT_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_solver is not None:
            sys.modules["solver"] = previous_solver
        else:
            sys.modules.pop("solver", None)


def _build_tune7_fixture():
    helper = _load_benchmark_helper()
    actions, parameter_spec = helper.build_actions_tune7(
        th_grid=np.asarray([0.10, 0.25, 0.40], dtype=float),
        mxrs_grid=np.asarray([0.70, 0.80, 0.90], dtype=float),
        tr_grid=np.asarray([0.00, 0.20], dtype=float),
        p_max_values=[2, 4],
        agg_nl_values=[0, 1, 2, 3, 4, 5],
        coarsen_type_values=[0, 2, 6, 8, 10],
        interp_values=[6, 8],
    )
    actions, default_arm = helper.ensure_default_arm(actions)
    feature_builder = Tune7FeatureBuilder(tuple(actions), parameter_spec)
    return helper, actions, parameter_spec, default_arm, feature_builder


def test_lattice_catalog_and_mixer_preserve_regimes_and_split():
    _helper, actions, _parameter_spec, default_arm, feature_builder = _build_tune7_fixture()
    catalog = Tune7LatticeCatalog(
        regime_ids=feature_builder.regime_indices,
        lattice_indices=feature_builder.lattice_indices,
        grid_shape=feature_builder.grid_shape,
    )
    selector = Tune7CandidateMixer(
        len(actions),
        catalog=catalog,
        candidate_pool_size=512,
        always_include_arms=[int(default_arm)],
        elite_cache_size=128,
        rng=np.random.default_rng(7),
        local_size=128,
        elite_size=128,
        random_size=256,
        random_mode="regime_stratified",
    )
    default_regime = int(feature_builder.regime_indices[int(default_arm)])
    eligible_elite = np.flatnonzero(feature_builder.regime_indices != default_regime)
    for rank, arm in enumerate(eligible_elite[:256]):
        selector.observe(int(arm), float(rank) / 100.0)

    candidate, stats = selector.candidate_subset(
        incumbent_arm=int(default_arm),
        local_seed_arms=np.asarray([int(default_arm)], dtype=int),
        return_stats=True,
    )
    assert candidate.size == 512
    assert stats["candidate_count"] == 512
    assert stats["local"] == 128
    assert stats["elite"] == 128
    assert stats["random"] == 256
    assert int(default_arm) in set(int(v) for v in candidate)

    local = catalog.local_from_seeds(
        seed_arms=np.asarray([int(default_arm)], dtype=int),
        excluded=np.asarray([int(default_arm)], dtype=int),
        need=32,
    )
    local_regimes = set(int(feature_builder.regime_indices[int(v)]) for v in local)
    assert local_regimes == {default_regime}
    assert catalog.lookup.shape[1:] == feature_builder.grid_shape


def test_nystrom_landmarks_are_context_driven_without_warmup_buffer():
    helper, _actions, _parameter_spec, _default_arm, feature_builder = _build_tune7_fixture()
    contexts = [
        np.asarray([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float),
        np.asarray([1.1, 0.0, 0.4, 0.2, 0.8, 0.1, 0.9, 0.3], dtype=float),
        np.asarray([0.9, 0.3, 0.1, 0.6, 0.2, 0.7, 0.5, 0.4], dtype=float),
    ]
    landmark_data = helper.select_nystrom_landmarks(
        contexts=contexts,
        feature_builder=feature_builder,
        m=4,
        seed=17,
    )
    assert landmark_data["landmark_x"].shape == (4, 8)
    assert landmark_data["landmark_z"].shape == (4, feature_builder.z_dim)
    assert landmark_data["landmark_regime"].shape == (4,)
    assert float(landmark_data["sigma_x"]) > 0.0
    assert float(landmark_data["sigma_z"]) > 0.0


def test_generate_instances_can_repeat_same_instance():
    helper = _load_benchmark_helper()
    sampler_kwargs = {
        "nx": 6,
        "ny": 6,
        "nz": 6,
        "n_min": 6,
        "n_max": 6,
        "c_min": 1.0,
        "c_max": 1000.0,
    }
    repeated = helper.generate_instances(
        T=4,
        seed=17,
        sampler_kwargs=sampler_kwargs,
        repeat_same_instance=True,
    )
    assert len(repeated) == 4
    first_mkw, first_context = repeated[0]
    for mkw, context in repeated[1:]:
        assert mkw == first_mkw
        assert np.allclose(context, first_context)
        assert mkw is not first_mkw


def test_feature_builder_shapes_match_frozen_v1_contract():
    _helper, actions, parameter_spec, _default_arm, feature_builder = _build_tune7_fixture()
    x = np.asarray([1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
    arms = np.asarray([0, 1, 2], dtype=int)

    phi = feature_builder.phi_v4_matrix(x, arms)
    quad = feature_builder.quadratic_oracle_matrix(x, arms)
    regime = feature_builder.hierarchical_regime_matrix(x, np.asarray([0, 1], dtype=int))
    numeric = feature_builder.hierarchical_numeric_matrix(x, arms)

    assert phi.shape == (3, 8 + (1 + 4) * feature_builder.g_dim)
    assert feature_builder.lattice_indices.shape == (len(actions), 5)
    assert feature_builder.grid_shape == (3, 3, 2, 2, 6)
    assert quad.shape == (
        3,
        8
        + 5
        + 5
        + 3
        + feature_builder.e_c_dim
        + feature_builder.e_i_dim
        + feature_builder.regime_dim
        + 4 * 5
        + 4 * feature_builder.e_c_dim
        + 4 * feature_builder.e_i_dim,
    )
    assert regime.shape == (2, 8 + feature_builder.regime_dim + 4 * feature_builder.regime_dim)
    assert numeric.shape == (3, 8 + 5 + 5 + 3 + 4 * 5)


def test_new_tune7_learners_support_observe_and_candidate_restricted_predict():
    helper, actions, parameter_spec, _default_arm, feature_builder = _build_tune7_fixture()
    context = np.asarray([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
    params = dict(actions[0])
    landmarks = {
        "landmark_x": np.repeat(context.reshape(1, -1), 4, axis=0),
        "landmark_z": feature_builder.z_matrix(np.arange(4, dtype=int)),
        "landmark_regime": feature_builder.regime_indices[:4],
        "sigma_x": 1.0,
        "sigma_z": 1.0,
    }
    learners = [
        SharedLinTS_AMG_v4(actions, context_dim=8, parameter_spec=parameter_spec, seed=1),
        SquareCB_AMG_v4(actions, context_dim=8, parameter_spec=parameter_spec, oracle_kind="linear", seed=2),
        SquareCB_AMG_v4(actions, context_dim=8, parameter_spec=parameter_spec, oracle_kind="quadratic", seed=3),
        NystromKernelUCB_AMG_v4(actions, context_dim=8, parameter_spec=parameter_spec, seed=4, **landmarks),
        HierarchicalSquareCB_AMG_v4(actions, context_dim=8, parameter_spec=parameter_spec, seed=5),
        HierarchicalLinUCB_AMG_v5(actions, context_dim=8, parameter_spec=parameter_spec, seed=6),
    ]
    candidate_arms = np.asarray([0, 3, 7, 11], dtype=int)

    for learner in learners:
        learner.observe(context=context, params=params, loss=0.6, bounded_loss=0.3)
        chosen, info = learner.predict_from_candidate_arms(context, candidate_arms)
        assert chosen in [actions[int(v)] for v in candidate_arms]
        assert "pred_mean" in info
        learner.update(0.4, bounded_loss=0.2)
        assert learner.t == 2


def test_hierarchical_lincb_v5_updates_only_chosen_regime_and_default_injection_is_local():
    _helper, actions, parameter_spec, default_arm, feature_builder = _build_tune7_fixture()
    context = np.asarray([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
    learner = HierarchicalLinUCB_AMG_v5(
        actions,
        context_dim=8,
        parameter_spec=parameter_spec,
        always_include_arms=[int(default_arm)],
        feature_builder=feature_builder,
        seed=17,
    )

    default_regime = int(feature_builder.regime_indices[int(default_arm)])
    for regime in range(feature_builder.regime_dim):
        selector = learner._selector_for_regime(regime)
        include = set(int(v) for v in selector.always_include_arms)
        if regime == default_regime:
            assert int(default_arm) in include
        else:
            assert int(default_arm) not in include

    params = learner.predict(context)
    chosen_arm = int(feature_builder.arm_for_params(params))
    chosen_regime = int(feature_builder.regime_indices[chosen_arm])
    assert chosen_regime == int(learner._last_regime)

    before_counts = learner.regime_counts.copy()
    learner.update(0.35, bounded_loss=0.20)
    delta = learner.regime_counts - before_counts
    assert int(np.sum(delta)) == 1
    assert int(delta[chosen_regime]) == 1
    assert int(np.count_nonzero(delta)) == 1
    assert learner.t == 1


def test_adapters_force_default_on_first_round_for_tune7_only():
    helper, actions, parameter_spec, default_arm, _feature_builder = _build_tune7_fixture()
    context = np.asarray([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)

    v2_actions = helper.build_actions_tune3(
        th_grid=np.asarray([0.10, 0.25], dtype=float),
        mxrs_grid=np.asarray([0.80, 0.90], dtype=float),
        tr_grid=np.asarray([0.00, 0.20], dtype=float),
    )
    v2_actions, v2_default_arm = helper.ensure_default_arm(v2_actions)
    v2 = helper.DirectPolicyAdapter(
        helper.SharedLinUCB_AMG_v2(v2_actions, context_dim=8, alpha=1.0, l2_reg=1.0, seed=11),
    )
    assert not v2._should_force_initial_guess()

    tune7 = helper.DirectPolicyAdapter(
        SharedLinTS_AMG_v4(
            actions,
            context_dim=8,
            parameter_spec=parameter_spec,
            always_include_arms=[int(default_arm)],
            seed=12,
        ),
        initial_params=actions[int(default_arm)],
        initial_guess_rounds=1,
    )
    chosen_tune7, _ = tune7.select(context)
    assert chosen_tune7 == actions[int(default_arm)]
    tune7.update(loss=0.4, params=chosen_tune7)
    assert tune7.learner.t == 1


def test_benchmark_script_smoke_run_writes_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("T", "4")
    monkeypatch.setenv("FIXED_N", "6")
    monkeypatch.setenv("GRID_N", "2")
    monkeypatch.setenv("P_MAX_ELMTS_VALUES", "2,4")
    monkeypatch.setenv("AGG_NUM_LEVELS_VALUES", "0,1")
    monkeypatch.setenv("COARSEN_TYPE_VALUES", "0,2,6,8,10")
    monkeypatch.setenv("TUNE7_INTERP_TYPES", "6,8")

    module = _load_benchmark_script()
    module.PLOTS_BASE_DIR = tmp_path
    module.main()

    run_dirs = sorted(tmp_path.iterdir())
    assert len(run_dirs) == 1
    summary_path = run_dirs[0] / "benchmark_tune7_online_summary.json"
    csv_path = run_dirs[0] / "per_instance_runtime_data.csv"
    assert summary_path.exists()
    assert csv_path.exists()

    summary = json.loads(summary_path.read_text())
    assert set(summary["branch_labels"]) == {
        "Shared LinUCB v2 | tune3",
        "Shared LinTS v4 | tune7",
        "SquareCB linear | tune7",
        "SquareCB quadratic | tune7",
        "Nyström KernelUCB | tune7",
        "Hierarchical SquareCB | tune7",
        "Hierarchical LinUCB v5 | tune7",
    }
    assert "full_stream_cumulative_raw_penalized_loss_sec" in summary
    assert "failed_case_count" in summary
    assert "warmup_rounds" not in summary
    assert "loss_cap" not in summary
    assert "post_warmup_cumulative_raw_penalized_loss_sec" not in summary
    assert "external_tune7_candidate_pool_size" not in summary
    assert "external_tune7_candidate_split" not in summary
    assert summary["branch_hyperparams"]["Shared LinTS v4 | tune7"]["selector_type"] == "native_uniform_elite"
    assert summary["branch_hyperparams"]["Hierarchical SquareCB | tune7"]["second_candidate_pool_size"] == 512
    assert summary["branch_hyperparams"]["Hierarchical SquareCB | tune7"]["second_random_size"] == 32
    assert summary["branch_hyperparams"]["Hierarchical SquareCB | tune7"]["top_gamma_scale"] == 10.0
    assert summary["branch_hyperparams"]["Hierarchical LinUCB v5 | tune7"]["selector_type"] == "native_hierarchical_regime_value_lincb"
    assert summary["branch_hyperparams"]["Hierarchical LinUCB v5 | tune7"]["candidate_pool_size"] == 512
    assert summary["instance_stream_mode"] == "iid_stream"

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    failed_by_method = {}
    for row in rows:
        failed_by_method.setdefault(row["method"], 0)
        failed_by_method[row["method"]] += int(row["failed"])
        assert "select_sec" in row
        assert "loss_eval_sec" in row
        assert "update_sec" in row
    assert summary["failed_case_count"] == failed_by_method


def test_benchmark_script_can_repeat_same_instance(monkeypatch, tmp_path):
    monkeypatch.setenv("T", "2")
    monkeypatch.setenv("FIXED_N", "6")
    monkeypatch.setenv("GRID_N", "2")
    monkeypatch.setenv("P_MAX_ELMTS_VALUES", "2,4")
    monkeypatch.setenv("AGG_NUM_LEVELS_VALUES", "0,1")
    monkeypatch.setenv("COARSEN_TYPE_VALUES", "0,2,6,8,10")
    monkeypatch.setenv("TUNE7_INTERP_TYPES", "6,8")
    monkeypatch.setenv("TUNE7_BENCH_REPEAT_SAME_INSTANCE", "1")

    module = _load_benchmark_script()
    module.PLOTS_BASE_DIR = tmp_path
    module.main()

    run_dirs = sorted(tmp_path.iterdir())
    assert len(run_dirs) == 1
    summary_path = run_dirs[0] / "benchmark_tune7_online_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["instance_stream_mode"] == "repeated_same_instance"
    assert summary["repeat_same_instance"] is True
