from typing import Any

import jax
import numpy as np
from jax.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D

from qlipper.configuration import SimConfig
from qlipper.constants import R_EARTH, R_MOON
from qlipper.postprocess.interpolation import interpolate_time
from qlipper.postprocess.plotting_utils import plot_sphere
from qlipper.sim.params import Params


def plot_trajectory_cart(
    t: ArrayLike,
    y: ArrayLike,
    cfg: SimConfig,
    params: Params,
    plot_kwargs: dict[str, Any] = {},
) -> None:
    """
    Plots full 3D trajectory from cartesian states.

    Single colour output.

    Parameters
    ----------
    t : ArrayLike
        Time array.
    y : ArrayLike
        Cartesian state array.
    cfg : SimConfig
        Simulation configuration.
    save_path : Path | None, optional
        Path to save the plot, by default None.
    save_kwargs : dict[str, Any], optional
        Keyword arguments for saving the plot, by default {}.
    show : bool, optional
        Whether to show the plot, by default False.
    """

    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax: Axes3D = fig.add_subplot(projection="3d")

    # MATLAB default view
    ax.view_init(elev=30, azim=-127.5)

    plot_sphere(
        ax,
        radius=R_EARTH,
        plot_kwargs={"color": (0.3010, 0.7450, 0.9330), "alpha": 0.6},
    )

    t, y = interpolate_time(t, y, len(y))

    # split the trajectory into segments based on L
    NUM_SEGMENTS = 50
    idx_breakpoints = np.linspace(0, len(y), NUM_SEGMENTS + 1, dtype=int)

    cm = plt.get_cmap("turbo")

    for i in range(NUM_SEGMENTS):
        default_plot_kwargs = {"linewidth": 1, "color": cm(i / NUM_SEGMENTS)}
        actual_plot_kwargs = default_plot_kwargs | plot_kwargs

        # segments have overlapping points
        plot_slice = slice(idx_breakpoints[i], idx_breakpoints[i + 1] + 1)

        ax.plot(
            y[plot_slice, 0],
            y[plot_slice, 1],
            y[plot_slice, 2],
            **actual_plot_kwargs,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # plot the moon, if applicable (in future: generalize)
    if "moon_gravity" in cfg.perturbations:
        # moon ephemeris
        y = jax.vmap(params.moon_ephem.evaluate)(t)

        ax.plot(
            y[:, 0],
            y[:, 1],
            y[:, 2],
            label="Moon",
            color="gray",
            linestyle="--",
            alpha=0.2,
        )

    ax.set_title("Earth Inertial Coordinates")
    # equal aspect ratio
    ax.set_aspect("equal")

    t_start = t[0]
    t_end = t[-1]
    t_span = t_end - t_start

    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cm),
        ax=ax,
        label="MET [d]",
        format=FuncFormatter(lambda x, _: f"{((t_start + x*t_span)/86400):.0f}"),
        location="bottom",
    )


def plot_cart_wrt_moon(
    t: ArrayLike,
    y: ArrayLike,
    cfg: SimConfig,
    params: Params,
    plot_kwargs: dict[str, Any] = {},
) -> None:
    # convert to moon inertial coordinates
    # moon ephemeris
    moon_state = jax.vmap(params.moon_ephem.evaluate)(t)
    y = y - moon_state

    # interpolate wrt time for better plotting
    dt_max = 500
    t, y = interpolate_time(t, y, int((t[-1] - t[0]) / dt_max))

    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax: Axes3D = fig.add_subplot(projection="3d")

    # MATLAB default view
    ax.view_init(elev=30, azim=-127.5)

    plot_sphere(
        ax,
        radius=R_MOON,
        plot_kwargs={"color": (0.3, 0.3, 0.3), "alpha": 0.6},
    )

    # split the trajectory into segments based on L
    NUM_SEGMENTS = 50
    idx_breakpoints = np.linspace(0, len(y), NUM_SEGMENTS + 1, dtype=int)

    cm = plt.get_cmap("turbo")

    for i in range(NUM_SEGMENTS):
        default_plot_kwargs = {"linewidth": 1, "color": cm(i / NUM_SEGMENTS)}
        actual_plot_kwargs = default_plot_kwargs | plot_kwargs

        # segments have overlapping points
        plot_slice = slice(idx_breakpoints[i], idx_breakpoints[i + 1] + 1)

        ax.plot(
            y[plot_slice, 0],
            y[plot_slice, 1],
            y[plot_slice, 2],
            **actual_plot_kwargs,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title("Moon Inertial Coordinates")
    # equal aspect ratio
    ax.set_aspect("equal")

    t_start = t[0]
    t_end = t[-1]
    t_span = t_end - t_start

    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cm),
        ax=ax,
        label="MET [d]",
        format=FuncFormatter(lambda x, _: f"{((t_start + x*t_span)/86400):.1f}"),
        location="bottom",
    )
