from typing import Callable

from diffrax import Event
from jax import Array

from qlipper.configuration import SimConfig
from qlipper.constants import P_SCALING, R_EARTH, R_MOON
from qlipper.converters import cartesian_to_mee, mee_to_cartesian
from qlipper.sim import Params
from qlipper.sim.dynamics_cartesian import CARTESIAN_DYN_SCALING
from qlipper.sim.loss import l2_loss


# helper functions for converting between different dynamics
def call_with_unscaled_mee(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        return f(t, y.at[0].mul(P_SCALING), args)

    return wrapper


def call_with_unscaled_cart(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        return f(t, y * CARTESIAN_DYN_SCALING, args)

    return wrapper


def call_cvt_mee_to_cart(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        return f(t, mee_to_cartesian(y), args)

    return wrapper


def call_cvt_cart_to_mee(f: Callable) -> Callable:
    def wrapper(t: float, y: Array, args: Params, **kwargs) -> Array:
        return f(t, cartesian_to_mee(y), args)

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
    loss = l2_loss(y, args.y_target, args.w_oe)

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
    r = y[:3] - args.moon_ephem.evaluate(t)
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

    match cfg.dynamics:
        case "mee":
            fcn_list = [
                call_with_unscaled_mee(guidance_converged),
                call_with_unscaled_mee(call_cvt_mee_to_cart(crashed_into_earth)),
                call_with_unscaled_mee(call_cvt_mee_to_cart(crashed_into_moon)),
            ]
        case "cartesian":
            fcn_list = [
                call_with_unscaled_cart(call_cvt_cart_to_mee(guidance_converged)),
                call_with_unscaled_cart(crashed_into_earth),
                call_with_unscaled_cart(crashed_into_moon),
            ]
        case _:
            raise ValueError(f"Unknown dynamics: {cfg.dynamics}")

    return Event(fcn_list)