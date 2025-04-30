import logging

import jax
import jax.numpy as jnp
from diffrax import CubicInterpolation, backward_hermite_coefficients

logger = logging.getLogger(__name__)


def infer_n_interp(y_mee: jax.Array, seg_per_orbit: int) -> int:
    """
    Determine the number of interpolation points based on the
    smoothly capturing the shape of each orbit.

    Parameters
    ----------
    y_mee : Array, shape (N, 6)
        State vectors in Modified Equinoctial Elements.
    seg_per_orbit : int
        Number of segments per orbit.

    Returns
    -------
    n_interp : int
        Number of interpolation points.

    """
    l_step = 2 * jnp.pi / seg_per_orbit
    n_interp = (y_mee[-1, -1] - y_mee[0, -1]) / l_step

    return n_interp.astype(int)


def filter_idx_non_increasing(arr: jax.Array) -> jax.Array:
    """
    Removes indices where the array is non-increasing.

    Parameters
    ----------
    arr : Array, shape (N,)
        Array to flag (1-D array).

    Returns
    -------
    good_idx : Array, shape (M,)
        Indices where the array is increasing,
        s.t. arr[good_idx] is strictly increasing.

    """
    delta = jnp.diff(arr, prepend=arr[0] - 1)
    good_idx = (delta > 0).nonzero()

    if (bad_count := jnp.count_nonzero(delta <= 0)) > 0:
        msg = f"Removed {bad_count}/{len(arr)} points where array is decreasing"
        logger.warning(msg)
    return good_idx


def interpolate_time(
    t: jax.Array,
    y: jax.Array,
    n_interp: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Interpolate in linear time.

    Parameters
    ----------
    t : Array, shape (N,)
        Time vector in seconds elapsed.
    y : Array, shape (N, 6)
        State vectors.
    n_interp : int
        Number of total points in the interpolated time vector.

    Returns
    -------
    t_interp : Array, shape (N_interp,)
        Interpolated time vector in seconds elapsed.
    y_interp : Array, shape (N_interp, 6)
        Interpolated state vector.

    """
    # filter out points where time is not increasing
    good_idx = filter_idx_non_increasing(t)
    t = t[good_idx]
    y = y[good_idx]

    t_interp = jnp.linspace(t[0], t[-1], n_interp)

    # interpolate the state vector
    coeffs = backward_hermite_coefficients(t, y)
    interp = CubicInterpolation(t, coeffs)
    y_interp = jax.vmap(interp.evaluate)(t_interp)
    return t_interp, y_interp


def interpolate_mee(
    t: jax.Array,
    y: jax.Array,
    seg_per_orbit: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Interpolate the state vector to a higher resolution by using
    a fixed number of segments per orbit.

    Parameters
    ----------
    t : Array, shape (N,)
        Time vector in seconds elapsed.
    y : Array, shape (N, 6)
        State vectors.
    seg_per_orbit : int
        Number of segments per orbit.

    Returns
    -------
    t_interp : Array, shape (N_interp,)
        Interpolated time vector in seconds elapsed.
    y_interp : Array, shape (N_interp, 6)
        Interpolated state vector.

    """
    n_interp = infer_n_interp(y, seg_per_orbit)
    l_eval = jnp.linspace(y[0, -1], y[-1, -1], n_interp)

    # use true longitude as the interpolation key
    l_v = y[:, -1]

    # take only points where L is increasing to avoid errors in interpolation
    good_idx = filter_idx_non_increasing(l_v)
    l_v = l_v[good_idx]
    y = y[good_idx]
    t = t[good_idx]
    s_v = jnp.concatenate((y[:, :-1], t[:, None]), axis=1)

    # interpolate the state vector
    coeffs = backward_hermite_coefficients(l_v, s_v)
    interp = CubicInterpolation(l_v, coeffs)

    interpolated = jax.vmap(interp.evaluate)(l_eval)
    y_interp = jnp.concatenate((interpolated[:, :-1], l_eval[:, None]), axis=1)
    t_interp = interpolated[:, -1]

    return t_interp, y_interp
