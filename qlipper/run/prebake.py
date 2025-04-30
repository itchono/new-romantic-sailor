import logging
from functools import partial
from typing import Any, Callable

from diffrax import (
    CubicInterpolation,
    backward_hermite_coefficients,
)
from equinox import Module
from jax import Array, jit

from qlipper.configuration import SimConfig
from qlipper.sim.cr3bp import generate_fake_arrays
from qlipper.sim.dynamics_cartesian import dyn_cartesian
from qlipper.sim.ephemeris import generate_ephem_arrays, lookup_body_id
from qlipper.sim.params import GuidanceParams, Params
from qlipper.sim.perturbations import PERTURBATIONS
from qlipper.sim.propulsion import PROPULSION_MODELS
from qlipper.steering import STEERING_LAWS

logger = logging.getLogger(__name__)


class DummyInterpolant(Module):
    def evaluate(self, t: float) -> Array:
        return Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def prebake_sim_config(cfg: SimConfig) -> Params:
    """
    Prebake the SimConfig struct into a SimInternalConfig struct
    that can be passed into the actual problem being solved.

    TODO: make the ephemeris generation available as its
    own function, so that it can be reused in other places
    """

    # Generate ephemeris interpolant arrays
    sun = lookup_body_id("sun")
    earth = lookup_body_id("earth")
    moon = lookup_body_id("moon")

    # Heuristic: 2 samples per day, minimum 10 samples
    num_ephem_samples = max(int((cfg.t_span[1] - cfg.t_span[0]) / 86400 * 2), 10)

    # Only generate ephem interpolants for the bodies that are actually used

    logger.info(
        "Generating ephemeris interpolant arrays - "
        f"{num_ephem_samples} samples will be used"
    )

    if cfg.ephemeris == "cr3bp":
        # Generate fake arrays for CR3BP
        sun_ephem = DummyInterpolant()
        moon_t_sample, moon_state_sample = generate_fake_arrays(
            earth, moon, cfg.epoch_jd, cfg.t_span, num_ephem_samples
        )
        interp_coeffs = backward_hermite_coefficients(moon_t_sample, moon_state_sample)
        logger.info("Using CR3BP ephemeris")
        moon_ephem = CubicInterpolation(moon_t_sample, interp_coeffs)

    else:
        if "sun_gravity" in cfg.perturbations:
            # Generate sun ephemeris interpolant
            sun_t_sample, sun_state_sample = generate_ephem_arrays(
                earth, sun, cfg.epoch_jd, cfg.t_span, num_ephem_samples
            )
            sun_state_sample = sun_state_sample * 1e3  # convert from km to m
            interp_coeffs = backward_hermite_coefficients(
                sun_t_sample, sun_state_sample.T
            )
            sun_ephem = CubicInterpolation(sun_t_sample, interp_coeffs)
        else:
            logger.warning(
                "Sun gravity perturbation not enabled, using dummy interpolant"
            )
            sun_ephem = DummyInterpolant()

        # Generate moon ephemeris interpolant
        if "moon_gravity" in cfg.perturbations:
            moon_t_sample, moon_state_sample = generate_ephem_arrays(
                earth, moon, cfg.epoch_jd, cfg.t_span, num_ephem_samples
            )
            moon_state_sample = moon_state_sample * 1e3  # convert from km to m
            interp_coeffs = backward_hermite_coefficients(
                moon_t_sample, moon_state_sample.T
            )
            moon_ephem = CubicInterpolation(moon_t_sample, interp_coeffs)
        else:
            logger.warning(
                "Moon gravity perturbation not enabled, using dummy interpolant"
            )
            moon_ephem = DummyInterpolant()

    return Params(
        y_target=cfg.y_target,
        conv_tol=cfg.conv_tol,
        earth_guidance=GuidanceParams(
            w_oe=cfg.earth_w_oe,
            penalty_weight=cfg.earth_penalty_weight,
            penalty_scaling=cfg.earth_penalty_scaling,
            rp_min=cfg.earth_rp_min,
        ),
        moon_guidance=GuidanceParams(
            w_oe=cfg.moon_w_oe,
            penalty_weight=cfg.moon_penalty_weight,
            penalty_scaling=cfg.moon_penalty_scaling,
            rp_min=cfg.moon_rp_min,
        ),
        characteristic_accel=cfg.characteristic_accel,
        epoch_jd=cfg.epoch_jd,
        sun_ephem=sun_ephem,
        moon_ephem=moon_ephem,
    )


def prebake_ode(cfg: SimConfig) -> Callable[[float, Array, Any], Array]:
    """
    Bake a version of the ode so that it can be JIT-compiled

    Parameters
    ----------
    ode : Callable[[float, Array, Any, Any], Array]
        The original ODE function
    cfg : SimConfig
        The simulation configuration

    Returns
    -------
    baked_ode : Callable[[float, Array, Any], Array]
        The baked ODE function
    """

    steering_law = STEERING_LAWS[cfg.steering_law]
    propulsion_model = PROPULSION_MODELS[cfg.propulsion_model]

    ode = dyn_cartesian

    baked_ode = jit(
        partial(
            ode,
            steering_law=steering_law,
            propulsion_model=propulsion_model,
            perturbations=[PERTURBATIONS[p] for p in cfg.perturbations],
        )
    )

    return baked_ode
