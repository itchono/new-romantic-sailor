# cr3bp ephemeris computation

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from qlipper.constants import MU_EARTH


def generate_fake_arrays(
    observer: int, target: int, epoch: float, t_span: ArrayLike, num_samples: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate arrays of ephemeris data for a given observer and target

    Parameters
    ----------
    observer : int
        The observer body
    target : int
        The target body
    epoch : float
        The epoch of the ephemeris data
    t_span : ArrayLike
        The time span over which to compute the ephemeris
    num_samples : int
        The number of samples to take over t_span

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        The times and positions of the ephemeris data
    """
    # ensure it's moon and Earth only

    # observer = 399
    # target = 301

    if observer != 399 or target != 301:
        raise ValueError("Only observer=399 and target=301 are supported")

    MOON_DISTANCE = 384400e3

    # compute orbital speed of moon
    v_orbit = np.sqrt(MU_EARTH / MOON_DISTANCE)

    # compute period and true anomaly
    T_orbit = 2 * np.pi * MOON_DISTANCE / v_orbit
    t = jnp.linspace(t_span[0], t_span[1], num_samples)

    theta = 2 * np.pi * t / T_orbit

    x = MOON_DISTANCE * jnp.cos(theta)
    y = MOON_DISTANCE * jnp.sin(theta)
    z = jnp.zeros_like(x)
    vx = -v_orbit * jnp.sin(theta)
    vy = v_orbit * jnp.cos(theta)
    vz = jnp.zeros_like(x)

    # rotate these slightly in the x-z plane to avoid singularities
    angle = 1e-6

    x, z = (
        x * jnp.cos(angle) - z * jnp.sin(angle),
        x * jnp.sin(angle) + z * jnp.cos(angle),
    )
    vx, vz = (
        vx * jnp.cos(angle) - vz * jnp.sin(angle),
        vx * jnp.sin(angle) + vz * jnp.cos(angle),
    )

    return t, jnp.stack([x, y, z, vx, vy, vz], axis=1)
