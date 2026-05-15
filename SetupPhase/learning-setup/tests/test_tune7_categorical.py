from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


LEARNING_SETUP_ROOT = Path(__file__).resolve().parents[1]
TEST_FINAL_PATH = LEARNING_SETUP_ROOT / "scripts" / "test_final.py"

if str(LEARNING_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(LEARNING_SETUP_ROOT))

from learners.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from learners._amg_action_features import (
    GenericActionFeatureEncoder,
    action_from_ordered_values,
    action_key_from_parameter_space_spec,
)


def _load_test_final_module():
    module_name = "_test_final_unit"
    sys.modules.pop(module_name, None)

    solver_stub = types.ModuleType("solver")

    def _unexpected_solve(*_args, **_kwargs):
        raise AssertionError("solve() should not be called in tune7 categorical unit tests")

    solver_stub.solve = _unexpected_solve
    previous_solver = sys.modules.get("solver")
    sys.modules["solver"] = solver_stub
    try:
        spec = importlib.util.spec_from_file_location(module_name, TEST_FINAL_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {TEST_FINAL_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_solver is not None:
            sys.modules["solver"] = previous_solver
        else:
            sys.modules.pop("solver", None)


def test_tune7_builder_is_categorical_only_and_includes_default(monkeypatch):
    monkeypatch.setenv("P_MAX_ELMTS_VALUES", "2")
    monkeypatch.setenv("AGG_NUM_LEVELS_VALUES", "1")
    monkeypatch.setenv("COARSEN_TYPE_VALUES", "0")
    monkeypatch.setenv("TUNE7_INTERP_TYPES", "8")

    mod = _load_test_final_module()
    actions, p_max_values, agg_nl_values, coarsen_values, interp_values, parameter_spec = mod._build_actions_tune7(
        th_grid=np.asarray([0.10], dtype=float),
        mxrs_grid=np.asarray([0.80], dtype=float),
        tr_grid=np.asarray([0.20], dtype=float),
    )

    assert parameter_spec.parameter_names == (
        "strong_threshold",
        "max_row_sum",
        "trunc_factor",
        "P_max_elmts",
        "agg_num_levels",
        "coarsen_type",
        "interp_type",
    )
    assert p_max_values == [2]
    assert agg_nl_values == [1]
    assert coarsen_values == [0]
    assert interp_values == [8]
    assert any(mod._same_action(action, mod.DEFAULT_PARAMS) for action in actions)


def test_tune7_action_keys_stay_stable_for_the_default_action(monkeypatch):
    monkeypatch.setenv("P_MAX_ELMTS_VALUES", "2")
    monkeypatch.setenv("AGG_NUM_LEVELS_VALUES", "1")
    monkeypatch.setenv("COARSEN_TYPE_VALUES", "0")
    monkeypatch.setenv("TUNE7_INTERP_TYPES", "8")

    mod = _load_test_final_module()
    actions, *_rest, parameter_spec = mod._build_actions_tune7(
        th_grid=np.asarray([0.10], dtype=float),
        mxrs_grid=np.asarray([0.80], dtype=float),
        tr_grid=np.asarray([0.20], dtype=float),
    )

    default_ordered = [mod.DEFAULT_PARAMS[param.name] for param in parameter_spec.parameters]
    default_action = action_from_ordered_values(default_ordered, parameter_spec)
    default_key = action_key_from_parameter_space_spec(default_action, parameter_spec)
    action_keys = {
        action_key_from_parameter_space_spec(action, parameter_spec)
        for action in actions
    }

    assert default_key in action_keys
    assert default_action == {
        "strong_threshold": float(mod.DEFAULT_PARAMS["strong_threshold"]),
        "max_row_sum": float(mod.DEFAULT_PARAMS["max_row_sum"]),
        "trunc_factor": float(mod.DEFAULT_PARAMS["trunc_factor"]),
        "P_max_elmts": int(mod.DEFAULT_PARAMS["P_max_elmts"]),
        "agg_num_levels": int(mod.DEFAULT_PARAMS["agg_num_levels"]),
        "coarsen_type": int(mod.DEFAULT_PARAMS["coarsen_type"]),
        "interp_type": int(mod.DEFAULT_PARAMS["interp_type"]),
    }


def test_tune7_feature_shape_matches_the_categorical_spec(monkeypatch):
    monkeypatch.setenv("P_MAX_ELMTS_VALUES", "2,4")
    monkeypatch.setenv("AGG_NUM_LEVELS_VALUES", "0,1")
    monkeypatch.setenv("COARSEN_TYPE_VALUES", "0,10")
    monkeypatch.setenv("TUNE7_INTERP_TYPES", "6,8")

    mod = _load_test_final_module()
    actions, *_rest, parameter_spec = mod._build_actions_tune7(
        th_grid=np.asarray([0.10, 0.25], dtype=float),
        mxrs_grid=np.asarray([0.80, 0.90], dtype=float),
        tr_grid=np.asarray([0.00, 0.20], dtype=float),
    )

    encoder = GenericActionFeatureEncoder(parameter_spec)
    encoded = encoder.encode_actions(actions)

    assert encoded.shape == (len(actions), encoder.feature_dim)
    assert encoder.feature_dim > 0


def test_tune7_learner_respects_default_arm_when_candidate_restricted(monkeypatch):
    monkeypatch.setenv("P_MAX_ELMTS_VALUES", "2")
    monkeypatch.setenv("AGG_NUM_LEVELS_VALUES", "1")
    monkeypatch.setenv("COARSEN_TYPE_VALUES", "0")
    monkeypatch.setenv("TUNE7_INTERP_TYPES", "8")

    mod = _load_test_final_module()
    actions, *_rest, parameter_spec = mod._build_actions_tune7(
        th_grid=np.asarray([0.10], dtype=float),
        mxrs_grid=np.asarray([0.80], dtype=float),
        tr_grid=np.asarray([0.20], dtype=float),
    )
    actions, default_arm_index = mod._ensure_default_arm(actions)

    learner = SharedLinUCB_AMG_v4(
        actions,
        context_dim=8,
        parameter_spec=parameter_spec,
        alpha=1.0,
        l2_reg=1.0,
        candidate_pool_size=1,
        always_include_arms=[int(default_arm_index)],
        seed=0,
    )

    context = np.asarray([1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
    action = learner.predict(context)

    assert mod._same_action(action, mod.DEFAULT_PARAMS)
    assert learner.candidate_stats_history[-1]["candidate_count"] == 1

    learner.update(1.25)

    assert learner.t == 1
    assert np.isclose(learner.history[-1].loss, 1.25)


def test_tune7_source_has_no_variant_branching_left():
    source = TEST_FINAL_PATH.read_text(encoding="utf-8")

    assert "agg_conditional" not in source
    assert "tune7-categorical" not in source
    assert "TUNE7_VARIANT" not in source
    assert "_build_actions_tune7_agg_conditional" not in source
    assert "def _build_actions_tune7(" in source
