from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def lvlh_to_steering(dir_lvlh: ArrayLike) -> tuple[float, float]:
    """
    Convert direction vector in LVLH frame to steering angles.

    Parameters
    ----------
    dir_lvlh : ArrayLike
        Direction vector in LVLH frame.

    Returns
    -------
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].
    """
    beta = jnp.atan2(dir_lvlh[2], jnp.linalg.norm(dir_lvlh[:2]))
    alpha = jnp.arctan2(dir_lvlh[0], dir_lvlh[1])
    return alpha, beta


@jax.jit
def steering_to_lvlh(alpha: float, beta: float) -> jax.Array:
    """
    Convert steering angles to direction vector in LVLH frame.

    Parameters
    ----------
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].

    Returns
    -------
    dir_lvlh : Array
        Direction vector in LVLH frame.
    """
    dir_lvlh = jnp.array(
        [
            jnp.cos(beta) * jnp.sin(alpha),
            jnp.cos(beta) * jnp.cos(alpha),
            jnp.sin(beta),
        ]
    )
    return dir_lvlh


@partial(jax.jit, static_argnums=(1,))
def mee_to_cartesian(mee: ArrayLike, mu: float) -> jax.Array:
    """
    Convert modified equinoctial elements to Cartesian elements.

    Parameters
    ----------
    mee : ArrayLike
        Modified equinoctial elements [a(m), f, g, h, k, L(rad)].
    mu : float
        Gravitational parameter of the central body;
        changing mu triggers a JIT recompile.

    Returns
    -------
    cart : Array
        Cartesian elements [x, y, z, vx, vy, vz] (m and m/s).

    Notes
    -----
    Formulation from https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf

    """
    # unpack state vector
    a, f, g, h, k, L = mee

    # convert SMA and ecc to p
    p = a * (1 - f**2 - g**2)

    # shorthand quantities defined in the document
    alpha_sq = h**2 - k**2
    s_sq = 1 + h**2 + k**2

    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)
    r = p / q

    # states
    pos = (
        r
        / s_sq
        * jnp.array(
            [
                jnp.cos(L) + alpha_sq * jnp.cos(L) + 2 * h * k * jnp.sin(L),
                jnp.sin(L) - alpha_sq * jnp.sin(L) + 2 * h * k * jnp.cos(L),
                2 * (h * jnp.sin(L) - k * jnp.cos(L)),
            ]
        )
    )

    vel = (
        1
        / s_sq
        * jnp.sqrt(mu / p)
        * jnp.array(
            [
                -(
                    jnp.sin(L)
                    + alpha_sq * jnp.sin(L)
                    - 2 * h * k * jnp.cos(L)
                    + g
                    - 2 * f * h * k
                    + alpha_sq * g
                ),
                -(
                    -jnp.cos(L)
                    + alpha_sq * jnp.cos(L)
                    + 2 * h * k * jnp.sin(L)
                    - f
                    + 2 * g * h * k
                    + alpha_sq * f
                ),
                2 * (h * jnp.cos(L) + k * jnp.sin(L) + f * h + g * k),
            ]
        )
    )

    return jnp.concatenate([pos, vel])


@partial(jax.jit, static_argnums=(1,))
def cartesian_to_mee(cart: ArrayLike, mu: float) -> jax.Array:
    """
    Convert Cartesian elements to modified equinoctial elements.

    Parameters
    ----------
    cart : ArrayLike
        Cartesian elements [x, y, z, vx, vy, vz] (m and m/s).
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    mee : Array
        Modified equinoctial elements [a(m), f, g, h, k, L(rad)].
    mu : float
        Gravitational parameter of the central body.

    Notes
    -----
    Transcribed from Fortran Astrodynamics Toolkit by jacobwilliams

    """
    pos = cart[0:3]
    vel = cart[3:6]
    rdv = pos @ vel
    rhat = pos / jnp.linalg.norm(pos, ord=2)
    rmag = jnp.linalg.norm(pos, ord=2)
    hvec = jnp.cross(pos, vel)
    hmag = jnp.linalg.norm(hvec, ord=2)
    hhat = hvec / hmag
    vhat = (rmag * vel - rdv * rhat) / hmag
    p = hmag**2 / mu
    k = hhat[0] / (1 + hhat[2] + 1e-10)
    h = -hhat[1] / (1 + hhat[2] + 1e-10)
    kk = k**2
    hh = h**2
    s2 = 1 + hh + kk
    tkh = 2 * k * h
    ecc = jnp.cross(vel, hvec) / mu - rhat
    fhat = jnp.array([1 - kk + hh, tkh, -2 * k])
    ghat = jnp.array([tkh, 1 + kk - hh, 2 * h])
    fhat = fhat / s2
    ghat = ghat / s2
    f = ecc @ fhat
    g = ecc @ ghat
    L = jnp.atan2(rhat[1] - vhat[0], rhat[0] + vhat[1])

    # convert a to p
    a = p / (1 - f**2 - g**2)

    return jnp.array([a, f, g, h, k, L])


@partial(jax.jit, static_argnums=(1,))
def batch_cartesian_to_mee(cart: ArrayLike, mu: float) -> jax.Array:
    """
    Vmapped version of cartesian_to_mee, which ensures
    that true longitude is unwrapped correctly.

    Parameters
    ----------
    cart : ArrayLike
        Cartesian elements (N, 6)
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    mee : Array
        Modified equinoctial elements (N, 6)
    """
    mee = jax.vmap(partial(cartesian_to_mee, mu=mu))(cart)

    # unwrap true longitude
    l_unwrap = jnp.unwrap(mee[:, 5])

    return jnp.column_stack((mee[:, :5], l_unwrap))


@partial(jax.jit, static_argnums=(1,))
def batch_mee_to_cartesian(mee: ArrayLike, mu: float) -> jax.Array:
    """
    Vmapped version of mee_to_cartesian.

    Does nothing special, just for consistency with its sibling function.

    Parameters
    ----------
    mee : ArrayLike
        Modified equinoctial elements (N, 6)
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    cart : Array
        Cartesian elements (N, 6)
    """
    return jax.vmap(partial(mee_to_cartesian, mu=mu))(mee)


@jax.jit
def rot_inertial_lvlh(y: ArrayLike) -> jax.Array:
    """
    Generates the rotation matrix C_IO rotating vectors from the lvlh frame
    to the inertial frame. i.e. v_I = C_IO @ v_O

    Parameters
    ----------
    y : ArrayLike
        Cartesian State vector [m, m/s]

    Returns
    -------
    C_IO : Array
        Rotation matrix from LVLH to Inertial frame
    """
    pos = y[:3]
    vel = y[3:]

    pos_unit = pos / jnp.linalg.norm(pos)
    vel_unit = vel / jnp.linalg.norm(vel)

    h = jnp.cross(pos_unit, vel_unit)
    h_normed = h / jnp.linalg.norm(h)

    return jnp.column_stack((pos_unit, jnp.cross(h_normed, pos_unit), h_normed))


@jax.jit
def rot_lvlh_inertial(cart: ArrayLike) -> jax.Array:
    """
    Generates the rotation matrix C_OI rotating vectors from the inertial frame
    to the lvlh frame. i.e. v_O = C_OI @ v_I

    Parameters
    ----------
    cart : ArrayLike
        Cartesian State vector [m, m/s]

    Returns
    -------
    C_OI : Array
        Rotation matrix from Inertial to LVLH frame
    """

    return rot_inertial_lvlh(cart).T


@jax.jit
def delta_angle_mod(a: float, b: float) -> float:
    """
    Shortest phase difference between two angles, i.e.
    how much a is ahead of b. Returns in range [-pi, pi].

    Source: https://stackoverflow.com/a/2007279

    Parameters
    ----------
    a : float
        First angle.
    b : float
        Second angle.

    Returns
    -------
    float
        Shortest phase difference between the two angles.

    """
    return ((a - b + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def a_mee_to_p_mee(y_mee: ArrayLike) -> jax.Array:
    """
    Converts a-based MEE to p-based MEE.
    """
    a = y_mee[0]
    f = y_mee[1]
    g = y_mee[2]
    p = a * (1 - f**2 - g**2)
    return y_mee.at[0].set(p)


def p_mee_to_a_mee(y_mee: ArrayLike) -> jax.Array:
    """
    Converts p-based MEE to a-based MEE.
    """
    p = y_mee[0]
    f = y_mee[1]
    g = y_mee[2]
    a = p / (1 - f**2 - g**2)
    return y_mee.at[0].set(a)
