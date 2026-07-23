"""Module alias for the canonical :mod:`setup.registry` API."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("setup.registry")
