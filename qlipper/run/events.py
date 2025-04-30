from typing import Callable

from diffrax import Event
from jax import Array

from qlipper.configuration import SimConfig
from qlipper.constants import LENGTH_SCALING, MU_EARTH, MU_MOON, R_EARTH, R_MOON
from qlipper.converters import cartesian_to_mee, mee_to_cartesian
from qlipper.sim import Params
from qlipper.sim.dynamics_cartesian import CARTESIAN_DYN_SCALING
from qlipper.sim.loss import l2_loss


# helper functions for converting between different dynamics
def call_with_unscaled_mee(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        """
        Expects y to be modified equinoctial elements, with
        its semilatus rectum scaled
        """
        return f(t, y.at[0].mul(LENGTH_SCALING), args)

    return wrapper


def call_with_unscaled_cart(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        """
        Expects y to be order unity scaled cartesian elements
        """
        return f(t, y * CARTESIAN_DYN_SCALING, args)

    return wrapper


def call_cvt_mee_to_cart(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        """
        Expects y to be modified equinoctial elements (outer layer)
        """
        return f(t, mee_to_cartesian(y, MU_EARTH), args)

    return wrapper


def call_cvt_cart_to_mee(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        """
        Expects y to be cartesian elements (outer layer)
        """
        return f(t, cartesian_to_mee(y, MU_EARTH), args)

    return wrapper


def call_cvt_mee_to_lunar(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        """
        Expects y to be earth centric mees
        """
        cart_state = mee_to_cartesian(y, MU_EARTH)
        moon_state = args.moon_ephem.evaluate(t)
        rel_mee = cartesian_to_mee(cart_state - moon_state, MU_MOON)

        return f(t, rel_mee, args)

    return wrapper


def guidance_converged(t: float, y: Array, args: Params, **kwargs) -> bool:
    """
    Termination condition for the ODE solver.

    Parameters
    ----------
    t : float
        Current time
    y : Array
        MEE state vector (unscaled)
    args : Params
        Simulation parameters

    Returns
    -------
    bool
        True if the termination condition is met
    """
    # Check if the guidance loss is below the convergence tolerance
    loss = l2_loss(y, args.y_target, args.moon_guidance.w_oe)

    return loss < args.conv_tol


def crashed_into_earth(t: float, y: Array, args: Params, **kwargs) -> bool:
    """
    Terminates solve if we crash into the Earth

    Parameters
    ----------
    t : float
        Current time
    y : Array
        Cartesian state vector
    args : Params
        Simulation parameters

    Returns
    -------
    bool
        True if we crash into the Earth
    """
    r = y[:3]
    return r @ r < R_EARTH**2


def crashed_into_moon(t: float, y: Array, args: Params, **kwargs) -> bool:
    """
    Terminates solve if we crash into the Moon

    Parameters
    ----------
    t : float
        Current time
    y : Array
        Cartesian state vector
    args : Params
        Simulation parameters

    Returns
    -------
    bool
        True if we crash into the Moon
    """
    r = y[:3] - args.moon_ephem.evaluate(t)[:3]
    return r @ r < R_MOON**2


def get_termination_events(cfg: SimConfig) -> Event:
    """
    Select the correct convergence criterion

    Parameters
    ----------
    cfg : SimConfig
        The simulation configuration

    Returns
    -------
    Event :
        The event to be passed to the ODE solver
    """

    # HACK: convergence wrt Moon
    conv = (
        call_cvt_mee_to_lunar(guidance_converged)
        if cfg.steering_law == "bbq_law"
        else guidance_converged
    )

    fcn_list = [
        call_with_unscaled_cart(call_cvt_cart_to_mee(conv)),
        call_with_unscaled_cart(crashed_into_earth),
        call_with_unscaled_cart(crashed_into_moon),
    ]

    return Event(fcn_list)
