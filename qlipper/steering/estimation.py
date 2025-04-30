import jax
import jax.numpy as jnp
from jax.numpy import cos, sin, sqrt
from jax.typing import ArrayLike


def d_oe_xx_mee_a(state: ArrayLike, mu: float, max_accel: float) -> jax.Array:
    """
    Approximates maximum rate of change of the state vector under
    only two-body dynamics.

    Parameters
    ----------
    state : ArrayLike
        State vector in modified equinoctial elements (a-based)
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    oe_dot_max : jax.Array
        Maximum rate of change of the state vector

    Notes
    -----
    Exact form for a, h, k are used, and a grid search is used
    to find the maximum rate of change for f and g across L in [0, 2pi].

    Modification made to max h and k rates according to Yuan et al. 2007,
    mainly to eliminate problems incurred by e > 1.

    Modification mde to max a rate to eliminate problems incurred by hyperbolic orbits.
    """
    # unpack state vector
    a, f, g, h, k, L = state

    # convert SMA and ecc to p
    p = a * (1 - f**2 - g**2)  # semi-latus rectum p = a(1 - e^2)
    e = sqrt(f**2 + g**2)

    # a is analytical (compensate for hyperbolic orbits)
    a_dot_max = (
        2 * jnp.abs(a) * sqrt(jnp.abs(a) / mu) * sqrt(jnp.abs((1 + e) / (1 - e)))
    )

    q = 1 + f * cos(L) + g * sin(L)

    # source: https://doi.org/10.1108/00022660710732699
    def f_dot_max_as_L(L: float) -> float:
        """Maximum RoC in f as a function of L."""
        q = 1 + f * cos(L) + g * sin(L)
        return (
            1
            / q
            * sqrt(p / mu)
            * sqrt(
                (f + sin(L) * (q + 1)) ** 2
                + (q * sin(L)) ** 2
                + g**2 * (k * cos(L) - h * sin(L)) ** 2,
            )
        )

    def g_dot_max_as_L(L: float) -> float:
        """Maximum RoC in g as a function of L."""
        q = 1 + f * cos(L) + g * sin(L)
        return (
            1
            / q
            * sqrt(p / mu)
            * sqrt(
                (g + sin(L) * (q + 1)) ** 2
                + (q * cos(L)) ** 2
                + f**2 * (k * cos(L) - h * sin(L)) ** 2,
            )
        )

    N_GRID = 50
    truelong_eval = jnp.linspace(-jnp.pi, jnp.pi, N_GRID)
    f_dot_max = jnp.max(f_dot_max_as_L(truelong_eval))
    g_dot_max = jnp.max(g_dot_max_as_L(truelong_eval))

    # h and k are analytical
    s_squared = 1 + h**2 + k**2
    h_dot_max = 1 / 2 * jnp.sqrt(p / mu) * s_squared / (2 * q)
    k_dot_max = 1 / 2 * jnp.sqrt(p / mu) * s_squared / (2 * q)

    return (
        jnp.array([a_dot_max, f_dot_max, g_dot_max, h_dot_max, k_dot_max]) * max_accel
    )


def d_oe_xx_mee_p(state: ArrayLike, mu: float, max_accel: float) -> jax.Array:
    """
    Approximates maximum rate of change of the state vector under
    only two-body dynamics.

    Parameters
    ----------
    state : ArrayLike
        State vector in modified equinoctial elements (p-based)
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    oe_dot_max : jax.Array
        Maximum rate of change of the state vector

    Notes
    -----
    Exact form for p, h, k are used, and a grid search is used
    to find the maximum rate of change for f and g across L in [0, 2pi].

    Modification made to max h and k rates according to Yuan et al. 2007,
    mainly to eliminate problems incurred by e > 1.
    """
    # unpack state vector
    p, f, g, h, k, L = state
    q = 1 + f * cos(L) + g * sin(L)

    # p is analytical
    p_dot_max = 2 * p / q * jnp.sqrt(p / mu)

    # source: https://doi.org/10.1108/00022660710732699
    def f_dot_max_as_L(L: float) -> float:
        """Maximum RoC in f as a function of L."""
        q = 1 + f * cos(L) + g * sin(L)
        return (
            1
            / q
            * sqrt(p / mu)
            * sqrt(
                (f + sin(L) * (q + 1)) ** 2
                + (q * sin(L)) ** 2
                + g**2 * (k * cos(L) - h * sin(L)) ** 2,
            )
        )

    def g_dot_max_as_L(L: float) -> float:
        """Maximum RoC in g as a function of L."""
        q = 1 + f * cos(L) + g * sin(L)
        return (
            1
            / q
            * sqrt(p / mu)
            * sqrt(
                (g + sin(L) * (q + 1)) ** 2
                + (q * cos(L)) ** 2
                + f**2 * (k * cos(L) - h * sin(L)) ** 2,
            )
        )

    N_GRID = 50
    truelong_eval = jnp.linspace(-jnp.pi, jnp.pi, N_GRID)
    f_dot_max = jnp.max(f_dot_max_as_L(truelong_eval))
    g_dot_max = jnp.max(g_dot_max_as_L(truelong_eval))

    # h and k are analytical
    s_squared = 1 + h**2 + k**2
    h_dot_max = 1 / 2 * jnp.sqrt(p / mu) * s_squared / (2 * q)
    k_dot_max = 1 / 2 * jnp.sqrt(p / mu) * s_squared / (2 * q)

    return (
        jnp.array([p_dot_max, f_dot_max, g_dot_max, h_dot_max, k_dot_max]) * max_accel
    )
