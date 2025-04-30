from typing import NamedTuple

from diffrax import CubicInterpolation
from jax import Array


class GuidanceParams(NamedTuple):
    w_oe: Array
    rp_min: float
    penalty_scaling: float
    penalty_weight: float


class Params(NamedTuple):
    """
    Subset of SimConfig struct passed into the actual
    problem being solved; only contains the necessary
    information needed as the ODE level for solving

    ONLY JITTABLE TYPES ALLOWED
    """

    y_target: Array
    conv_tol: float
    earth_guidance: GuidanceParams
    moon_guidance: GuidanceParams
    characteristic_accel: float
    epoch_jd: float
    sun_ephem: CubicInterpolation  # meters
    moon_ephem: CubicInterpolation  # meters
