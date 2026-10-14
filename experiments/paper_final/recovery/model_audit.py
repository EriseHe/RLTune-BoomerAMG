"""Execution and verification helpers for the official studies."""
from __future__ import annotations
import numpy as np

from experiments.paper_final.recovery.retry_comparison import FailureAwareRetryLinUCB

from experiments.paper_final.recovery.system import request_performance_qos

from problems.registry import context_for_setup_method

def audit_final_model(runtime, records):
    """Check every retained observation against a direct batch ridge calculation."""
    model = runtime.branch.policy.model
    features, targets, failures = [], [], []
    for row in records:
        context = context_for_setup_method(problem_kind=runtime.args.problem,
            setup_kind=runtime.method.setup_kind, setup_context=runtime.method.setup_context,
            matrix_kwargs=row["mkw"], stream_context=np.asarray(row["context"]))
        for attempt in row["outcome"]["primary_attempts"]:
            features.append(model._phi(np.asarray(context), attempt["arm_index"]))
            targets.append(attempt["learning_cost"])
            failures.append(float(attempt["status"] != "success"))
    phi = np.asarray(features)
    expected_inverse = np.linalg.inv(model.l2_reg*np.eye(model.d_phi) + phi.T @ phi)
    if model.t != len(features) or not np.allclose(model.A_inv, expected_inverse, rtol=1e-7, atol=1e-9):
        raise ValueError("Setup precision or observation count differs from executed attempts")
    if not np.allclose(model.b, phi.T @ targets, rtol=1e-10, atol=1e-8):
        raise ValueError("Runtime labels differ from retained observations")
    if isinstance(model, FailureAwareRetryLinUCB) and not np.allclose(
        model.failure_b, phi.T @ failures, rtol=1e-10, atol=1e-8
    ):
        raise ValueError("Failure model differs from immediate binary observations")
    return dict(passed=True, observations=len(features), failure_observations=int(sum(failures)),
                precision_max_abs_error=float(np.max(np.abs(model.A_inv-expected_inverse))))
