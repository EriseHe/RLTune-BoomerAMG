"""Build the setup observation encoder for the official solve controller."""

from __future__ import annotations
from dataclasses import replace
from setup.space import (
    DEFAULT_SETUP_PARAMS,
    SetupConfigurationSpace,
    SetupObsEncoder,
    build_setup_parameter_spec,
)


def make_setup_obs_encoder(
    *,
    configuration_space: SetupConfigurationSpace | None = None,
    parameter_resolution: int | None = None,
    strict_categories: bool = False,
) -> SetupObsEncoder:
    parameter_spec, _fixed_params = build_setup_parameter_spec(
        tune_dim=7,
        tune7_variant="categorical",
        parameter_resolution=parameter_resolution,
        coarsen_type_values=(
            configuration_space.coarsen_types if configuration_space else None
        ),
        interp_type_values=(
            configuration_space.interp_types if configuration_space else None
        ),
        agg_interp_type_values=(
            configuration_space.agg_interp_types if configuration_space else None
        ),
    )
    if strict_categories:
        # The native default remains a valid observation even outside a grid.
        parameter_spec = replace(
            parameter_spec,
            parameters=tuple(
                replace(param, values=(*param.values, param.default))
                if param.kind == "categorical" and param.default not in param.values
                else param
                for param in parameter_spec.parameters
            ),
        )
    return SetupObsEncoder(
        parameter_spec,
        dict(DEFAULT_SETUP_PARAMS),
        tuple(parameter_spec.parameter_names),
        strict_categories=strict_categories,
    )
