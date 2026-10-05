"""Historical checkpoint decision-history supplement, with its original format."""
from __future__ import annotations

import copy
from dataclasses import asdict
import numpy as np

from setup.learners.linucb.SharedLinUCB_AMG_v4 import SharedLinUCBv4Step
from experiments.paper_final.common.artifacts import ready


def decision_state(model):
    """Supplement the existing statistics checkpoint with decision history."""
    return ready({
        "history": [asdict(step) for step in model.history],
        "candidate_stats_history": model.candidate_stats_history,
        "local_neighbor_cache": [[list(k), v.tolist()] for k,v in model._local_neighbor_cache.items()],
    })


def restore_decision_state(model, state):
    model.history = [SharedLinUCBv4Step(**{k: (float("nan") if v is None else v) for k,v in r.items()})
                     for r in state["history"]]
    model.candidate_stats_history = copy.deepcopy(state["candidate_stats_history"])
    model._local_neighbor_cache = {
        (int(k[0]), bool(k[1]), int(k[2])): np.asarray(v, dtype=np.int64)
        for k,v in state["local_neighbor_cache"]}
