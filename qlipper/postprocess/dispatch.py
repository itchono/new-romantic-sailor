import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from matplotlib import pyplot as plt

from qlipper.configuration import SimConfig
from qlipper.constants import MU_EARTH, OUTPUT_DIR
from qlipper.converters import batch_mee_to_cartesian
from qlipper.postprocess.plot_cart import plot_cart_wrt_moon, plot_trajectory_cart
from qlipper.postprocess.plot_extra import plot_blending_weight
from qlipper.postprocess.plot_mee import plot_elements_mee
from qlipper.run.prebake import prebake_sim_config
from qlipper.sim.params import Params

logger = logging.getLogger(__name__)


def distance_rel_moon(t, y: ArrayLike, params: Params) -> float:
    moon_position = params.moon_ephem.evaluate(t)[:3]

    r_rel_moon = y[:3] - moon_position
    return r_rel_moon


def postprocess_run(
    run_id: str, t: ArrayLike, y: ArrayLike, cfg: SimConfig, show_plots: bool = False
) -> None:
    """
    Postprocess the results of a mission run.

    Parameters
    ----------
    run_id : str
        Unique identifier for the mission
    t : ArrayLike
        Time vector.
    y : ArrayLike
        State vector.
    cfg : SimConfig
        Simulation configuration.
    show_plots : bool, optional
        Whether to show the plots, default False.
    """
    logger.info(f"Postprocessing run: {run_id}")
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    # Plotting
    plot_save_dir = Path(OUTPUT_DIR) / cfg.name / run_id / "plots"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    # convert mee to cartesian
    y_cart = batch_mee_to_cartesian(y, MU_EARTH)

    # generate params from cfg
    params = prebake_sim_config(cfg)

    if cfg.steering_law == "bbq_law":
        dist_rel_moon = jax.vmap(distance_rel_moon, in_axes=(0, 0, None))(
            t, y_cart, params
        )
        dist_rel_moon = jnp.linalg.norm(dist_rel_moon, axis=1)

        within_lunar_soi = dist_rel_moon < 60000e3

        plot_trajectory_cart(t, y_cart, cfg, params)
        plt.savefig(plot_save_dir / "trajectory.pdf")

        # plot elements only when not within lunar SOI
        plot_elements_mee(
            t[~within_lunar_soi],
            y[~within_lunar_soi, :],
            cfg,
            params,
        )
        plt.savefig(plot_save_dir / "elements.pdf")

        # find when the spacecraft is close to the moon
        idx_close = np.where(~within_lunar_soi)[0][-1]

        if idx_close == len(t) - 1:
            logger.warning(
                "Spacecraft never got close to the moon, plotting last 5% of trajectory"
            )
            idx_close = int(len(t) * 0.95)

        plot_elements_mee(
            t,
            y,
            cfg,
            params,
            wrt_moon=True,
        )
        plt.savefig(plot_save_dir / "elements_wrt_moon.pdf")

        plot_cart_wrt_moon(
            t,
            y_cart,
            cfg,
            params,
        )
        plt.savefig(plot_save_dir / "trajectory_moon.pdf")

        plot_cart_wrt_moon(
            t[idx_close:],
            y_cart[idx_close:, :],
            cfg,
            params,
        )
        plt.savefig(plot_save_dir / "trajectory_moon_close.pdf")

        plot_elements_mee(
            t[idx_close:],
            y[idx_close:, :],
            cfg,
            params,
            wrt_moon=True,
        )
        plt.savefig(plot_save_dir / "elements_wrt_moon_close.pdf")

        plot_blending_weight(t, y_cart, dist_rel_moon, params)
        plt.savefig(plot_save_dir / "blending_weight.pdf")
    else:
        plot_trajectory_cart(t, y_cart, cfg, params)
        plt.savefig(plot_save_dir / "trajectory.pdf")

        plot_elements_mee(t, y, cfg, params)
        plt.savefig(plot_save_dir / "elements.pdf")

    if show_plots:
        plt.show()
    logger.info(f"Postprocessing complete for run: {run_id}")


def postprocess_from_folder(folder: Path, show_plots: bool = False) -> None:
    """
    Postprocess a qlipper run, writing outputs to the same folder.

    Parameters
    ----------
    folder : Path
        Folder containing the run.
    show_plots : bool, optional
        Whether to show the plots, default False.
    """

    # run sanity checks
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory.")
    if not (folder / "cfg.json").is_file():
        raise FileNotFoundError(f"{folder} does not contain a cfg.json file.")
    if not (folder / "vars.npz").is_file():
        raise FileNotFoundError(f"{folder} does not contain a vars.npz file.")

    # get run_id
    run_id = folder.name

    # assert run_id.startswith("run_"), f"{run_id} is not a valid run ID"

    # load config
    cfg = SimConfig.from_file(folder / "cfg.json")

    # load arrays
    loaded_obj = np.load(folder / "vars.npz")
    t = loaded_obj["t"]
    y = loaded_obj["y"]

    postprocess_run(run_id, t, y, cfg, show_plots)
