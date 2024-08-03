from pathlib import Path
from typing import Any

import numpy as np
from jax.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D

from qlipper.configuration import SimConfig
from qlipper.constants import R_EARTH
from qlipper.converters import batch_mee_to_cartesian
from qlipper.postprocess.interpolation import interpolate_mee
from qlipper.postprocess.plotting_utils import plot_sphere


def plot_elements_mee(
    t: ArrayLike,
    y: ArrayLike,
    cfg: SimConfig,
    save_path: Path | None = None,
    save_kwargs: dict[str, Any] = {},
    show: bool = False,
) -> None:
    """
    Plot the modified equinoctial elements over time.

    Parameters
    ----------
    t : ArrayLike
        Time array.
    y : ArrayLike
        Modified equinoctial elements array.
    cfg : SimConfig
        Simulation configuration.
    save_path : Path | None, optional
        Path to save the plot, default no save
    save_kwargs : dict[str, Any], optional
        Keyword arguments for saving the plot in plt.savefig
    show : bool, optional
        Whether to show the plot, default False
    """
    tof_days = t / 86400

    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    axs: list[plt.Axes]

    axs[0].plot(tof_days, y[:, 0], label="p", color="C0")
    axs[0].axhline(cfg.y_target[0], color="C0", linestyle="--", label="_")
    axs[0].set_ylabel("Orbit Size (m)")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(tof_days, y[:, 1], label="f", color="C1")
    axs[1].axhline(cfg.y_target[1], color="C1", linestyle="--", label="_")
    axs[1].plot(tof_days, y[:, 2], label="g", color="C2")
    axs[1].axhline(cfg.y_target[2], color="C2", linestyle="--", label="_")
    axs[1].set_ylabel("Eccentricity Vector")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(tof_days, y[:, 3], label="h", color="C3")
    axs[2].axhline(cfg.y_target[3], color="C3", linestyle="--", label="_")
    axs[2].plot(tof_days, y[:, 4], label="k", color="C4")
    axs[2].axhline(cfg.y_target[4], color="C4", linestyle="--", label="_")
    axs[2].set_ylabel("Nodal Position")
    axs[2].legend()
    axs[2].grid()

    # ticks and spines
    axs[2].set_xlabel(f"Time (days after JD {cfg.epoch_jd:.2f})")

    fig.suptitle("Modified Equinoctial Elements")

    if save_path is not None:
        fig.savefig(save_path, **save_kwargs)

    if show:
        plt.show()


def plot_trajectory_mee(
    t: ArrayLike,
    y: ArrayLike,
    plot_kwargs: dict[str, Any] = {},
    save_path: Path | None = None,
    save_kwargs: dict[str, Any] = {},
    show: bool = False,
) -> None:
    """
    Plots full 3D trajectory from modified equinoctial elements.

    Single colour output.

    Parameters
    ----------
    t : ArrayLike
        Time array.
    y : ArrayLike
        Modified equinoctial elements array.
    save_path : Path | None, optional
        Path to save the plot, by default None.
    save_kwargs : dict[str, Any], optional
        Keyword arguments for saving the plot, by default {}.
    show : bool, optional
        Whether to show the plot, by default False.
    """

    n_orbits = y[-1, -1] // (2 * np.pi)

    # interpolate smootly in L
    t_interp, y_interp = interpolate_mee(t, y, seg_per_orbit=100)

    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax: Axes3D = fig.add_subplot(projection="3d")

    # MATLAB default view
    ax.view_init(elev=30, azim=180 - 37.5)

    plot_sphere(ax, radius=R_EARTH, plot_kwargs={"color": "C0", "alpha": 0.5})

    # split the trajectory into segments based on L
    NUM_SEGMENTS = 20

    t_chunked = np.array_split(t_interp, NUM_SEGMENTS)
    y_chunked = np.array_split(y_interp, NUM_SEGMENTS, axis=0)

    cm = plt.get_cmap("turbo")

    for i in range(NUM_SEGMENTS):
        cart = batch_mee_to_cartesian(y_chunked[i])
        default_plot_kwargs = {"linewidth": 1, "color": cm(i / NUM_SEGMENTS)}
        actual_plot_kwargs = default_plot_kwargs | plot_kwargs
        ax.plot(cart[:, 0], cart[:, 1], cart[:, 2], **actual_plot_kwargs)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title("Earth Inertial Coordinates")
    # equal aspect ratio
    ax.set_aspect("equal")

    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cm),
        ax=ax,
        label="Orbit Number",
        format=FuncFormatter(lambda x, _: f"{x*n_orbits:.0f}"),
    )

    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    if show:
        plt.show()