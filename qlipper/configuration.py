from __future__ import annotations

from dataclasses import asdict, dataclass
from json import JSONEncoder, loads
from pathlib import Path

import jax.numpy as jnp
from jax import Array


class ArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@dataclass(frozen=True)
class SimConfig:
    """
    Simulation configuration.

    Attributes
    ----------
    name : str
        Mission name.
    y0 : Array, shape (6,)
        Initial state vector.
    y_target : Array, shape (5,)
        Target state vector.
    propulsion_model : str
        Propulsion model.
    steering_law : str
        Steering law.
    t_span : tuple[float, float]
        Time span in seconds.
    solver : str
        ODE solver.
    conv_tol : float
        Guidance convergence tolerance.
    w_oe : Array, shape (5,)
        Guidance weights.
    w_penalty : float
        Penalty weight.
    penalty_function : str
        Penalty function.
    kappa : float
        Cone angle parameter (degrees).
    dynamics : str
        Dynamics type.
    perturbations : list[str]
        List of perturbation functions.
    characteristic_accel : float
        Characteristic acceleration.
    epoch_jd : float
        Epoch Julian date.
    """

    name: str  # mission name
    y0: Array  # shape (6,)
    y_target: Array  # shape (5,)
    propulsion_model: str
    steering_law: str
    t_span: tuple[float, float]
    conv_tol: float
    earth_w_oe: Array
    earth_penalty_weight: float
    earth_penalty_scaling: float
    earth_rp_min: float
    moon_w_oe: Array
    moon_penalty_weight: float
    moon_penalty_scaling: float
    moon_rp_min: float
    perturbations: list[str]
    characteristic_accel: float
    epoch_jd: float
    ephemeris: str = "real"

    def __post_init__(self):
        # Validate the configuration
        assert len(self.y0) == 6, "Initial state vector must have length 6"
        assert len(self.y_target) == 5, "Target state vector must have length 5"

        # enforce array types
        for key in [
            "y0",
            "y_target",
            "earth_w_oe",
            "moon_w_oe",
        ]:
            object.__setattr__(
                self, key, jnp.array(getattr(self, key), dtype=jnp.float64)
            )

    def serialize(self) -> str:
        return ArrayEncoder(indent=4).encode(asdict(self))

    @classmethod
    def deserialize(cls, s: str) -> SimConfig:
        return cls(**loads(s))

    @classmethod
    def from_file(cls, path: Path) -> SimConfig:
        return cls.deserialize(Path(path).read_text())
