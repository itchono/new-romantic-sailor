import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.sim.params import GuidanceParams


def periapsis_penalty(state: ArrayLike, guidance_params: GuidanceParams) -> float:
    a = state[0]
    f = state[1]
    g = state[2]

    ecc = jnp.sqrt(f**2 + g**2)
    rp = a * (1 - ecc)

    penalty = jnp.exp(
        (1 - rp / guidance_params.rp_min) * guidance_params.penalty_scaling
    )

    return penalty
