"""Public setup-learning algorithms grouped by bandit family."""

from .bayesian import Bayesianbandits_AMG_v1, Bayesianbandits_AMG_v2
from .linucb import (
    LinUCB_AMG,
    LinUCB_AMG_v2,
    SharedLinUCB_AMG,
    SharedLinUCB_AMG_v2,
    SharedLinUCB_AMG_v3,
    SharedLinUCB_AMG_v4,
)
from .thompson import RFF_TS_AMG, SharedBootstrapTS_AMG, SharedLinTS_AMG
from .tsallis import TsallisINF_AMG

__all__ = [
    "Bayesianbandits_AMG_v1",
    "Bayesianbandits_AMG_v2",
    "LinUCB_AMG",
    "LinUCB_AMG_v2",
    "RFF_TS_AMG",
    "SharedBootstrapTS_AMG",
    "SharedLinTS_AMG",
    "SharedLinUCB_AMG",
    "SharedLinUCB_AMG_v2",
    "SharedLinUCB_AMG_v3",
    "SharedLinUCB_AMG_v4",
    "TsallisINF_AMG",
]
