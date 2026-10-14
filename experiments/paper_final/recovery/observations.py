"""Execution and verification helpers for the official studies."""
from __future__ import annotations
import numpy as np

from experiments.paper_final.recovery.retry_comparison import ConservativeRetryLinUCB, FailureAwareRetryLinUCB

class ObservedSelection:
    """Capture already computed scores; preserve the selector and its RNG calls."""

    def _selection_score_subset(self, x, **kwargs):
        score, mean, width = super()._selection_score_subset(x, **kwargs)
        arms = kwargs["arms"]
        self._capture = dict(arms=np.arange(self.K) if arms is None else arms,
                             score=score, mean=mean, width=width, alpha=kwargs["alpha"])
        return score, mean, width

    def _record_choice(self, retry):
        data = self._capture
        if data is None:
            record = dict(arm=int(self._last_arm), strategy="initial_guess")
        else:
            location = int(np.flatnonzero(data["arms"] == self._last_arm)[0])
            objective = data["retry_score"] if retry else data["score"]
            ranked = np.lexsort((data["score"], objective))
            def candidate(i):
                result = dict(arm=int(data["arms"][i]), mean=float(data["mean"][i]),
                    width=float(data["width"][i]), runtime_lcb=float(data["score"][i]))
                if retry and "failure_mean" in data:
                    result.update(failure_mean=float(data["failure_mean"][i]),
                                  failure_upper=float(data["failure_upper"][i]))
                elif retry:
                    result["runtime_ucb"] = float(data["retry_score"][i])
                return result
            strategy = ("failure_upper" if "failure_mean" in data else "runtime_ucb") if retry else "runtime_lcb"
            record = dict(strategy=strategy,
                observations=self.t, alpha=float(data["alpha"]), selected=candidate(location),
                best_two_gap=float(objective[ranked[1]]-objective[ranked[0]]) if len(ranked)>1 else None,
                candidate_count=len(ranked), candidate_arms=data["arms"].tolist(),
                top_candidates=[candidate(int(i)) for i in ranked[:8]])
        self.selection_diagnostics.append(record)
        self._capture = None

    def predict(self, context):
        self._capture = None
        result = super().predict(context)
        self._record_choice(False)
        return result

    def select_recovery(self, context, *, excluded_arms):
        self._capture = None
        result = super().select_recovery(context, excluded_arms=excluded_arms)
        self._record_choice(True)
        return result


class ObservedFailureAware(ObservedSelection, FailureAwareRetryLinUCB):
    def _recovery_best_locations(self, x, **kwargs):
        prediction = self._linear_prediction_subset(
            x, theta=self.A_inv @ self.failure_b, arms=kwargs["candidates"],
            action_features=kwargs["candidate_features"])
        upper = prediction + self.failure_beta * kwargs["runtime_uncert"]
        runtime_score = kwargs["runtime_score"]
        best = int(np.lexsort((runtime_score, upper))[0])
        self._capture.update(failure_mean=prediction, failure_upper=upper, retry_score=upper)
        return np.flatnonzero(np.isclose(upper, upper[best], atol=1e-12, rtol=0)
            & np.isclose(runtime_score, runtime_score[best], atol=1e-12, rtol=0))


class ObservedConservative(ObservedSelection, ConservativeRetryLinUCB):
    def _recovery_best_locations(self, x, **kwargs):
        score = kwargs["runtime_mean"] + self._effective_alpha() * kwargs["runtime_uncert"]
        self._capture["retry_score"] = score
        return np.flatnonzero(np.isclose(score, np.min(score), atol=1e-12, rtol=0))


def observed_model(model):
    if isinstance(model, FailureAwareRetryLinUCB):
        cls = ObservedFailureAware
    elif isinstance(model, ConservativeRetryLinUCB):
        cls = ObservedConservative
    else:
        raise TypeError("Unknown retry learner")
    observed = object.__new__(cls)
    observed.__dict__ = model.__dict__.copy()
    observed.selection_diagnostics, observed._capture = [], None
    return observed
