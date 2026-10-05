"""Compatibility imports for historical joint experiment entry points.

New code imports action_spaces, setup_branches, native_evaluation, or evaluation
directly. All implementations have one canonical owner.
"""

from __future__ import annotations

import copy
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple
import numpy as np
from solve.controllers.ppo import (
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    _SpaceOnlyEnv,
)
from setup.space import (
    SetupConfigurationSpace,
    SetupObsEncoder,
    build_setup_parameter_spec,
    build_setup_param_space,
)
from setup.learners.linucb import (
    run_same_context_setup_reselection,
    validate_linucb_v5_paper_contract,
    validate_linucb_v6_experimental_contract,
)
from setup.learners.common import (
    AOTCandidateSchedule,
    FactorizedActionFeatureCache,
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    ParameterSpec,
    RBFActionFeatureEncoder,
    SharedSetupLearnerSpec,
    resolve_tune7_candidate_strategy,
)
from setup.registry import (
    build_online_setup_learner,
    make_setup_learner_spec,
    normalize_setup_kind,
)
from hypre.bindings import (
    AMGNativeError,
    SolveStatus,
    augment_setup_params,
    create_env,
    run_with_default_fallback,
    solve,
)
from solve.core.outcomes import classify_rl_failure
from problems.amg import DIFCONV_CONTEXT_DIM
from problems.registry import (
    SCALAR_ANISOTROPIC_DIFFUSION,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
    normalize_problem_kind,
)
from problems.streams import (
    generate_difconv_instances as _generate_difconv_instances,
    generate_scalar_anisotropic_diffusion_instances,
    generate_scalar_anisotropic_diffusion_advection_instances,
)
from setup.utils.setup_amg import (
    build_actions_from_spec,
    build_actions_th_mxrs_tr,
    init_param_trace,
    progress_bar,
    record_param_trace,
)

from .action_spaces import (
    parse_float_list_env,
    build_action_space_bundle,
    same_action,
    build_actions_tune3,
    build_actions_tune5,
    ActionSpaceBundle,
    build_grids_from_env,
    parse_int_list_env,
    build_actions_tune7_agg_conditional,
    build_actions_tune7_categorical,
    ensure_default_arm,
    EXP44_MATRIX_GRID_N,
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
    TRACE_KEYS_FINAL,
    DEFAULT_SETUP_PARAMS,
    DEFAULT_COARSEN_TYPE_VALUES,
    DEFAULT_P_MAX_ELMTS_VALUES,
    DEFAULT_AGG_NUM_LEVELS_VALUES,
    DEFAULT_AGG_TR_VALUES,
    DEFAULT_AGG_PMX_VALUES,
    DEFAULT_TUNE7_INTERP_TYPES,
    DEFAULT_TUNE7_AGG_INTERP_TYPES,
)
from .setup_branches import (
    run_bandit_step_test_final,
    normalize_method_name,
    clone_branch_for_independent_updates,
    RandomPolicy,
    GenericBanditPolicy,
    build_test10_branches,
    default_branch_label,
    BranchRun,
    TestFinalBanditConfig,
    default_test_final_bandit_config_from_env,
    _safe_one_at_a_time_actions,
    generate_difconv_instances,
    build_online_linucb_branch,
    FixedPolicy,
    validate_expected_setup_action_count,
    build_test_final_bandit_policy,
    build_single_branch,
    family_seed_map,
)
from .native_evaluation import (
    solve_schedule_case,
    solve_fixed_w_case,
    classify_no_rl_failure,
    solve_setup_aware_rl_case,
    solve_default_baseline_case,
    _actual_failure_result,
    solve_no_rl_case,
)
from .evaluation import (
    eval_runner,
    fixed_trace,
    alloc_branch_metrics,
    run_interleaved_branch_scenario,
)
