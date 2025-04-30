import jax
from matplotlib import pyplot as plt

from qlipper.sim.params import Params
from qlipper.steering.multibody import blending_weight


def plot_blending_weight(
    t: jax.Array, y: jax.Array, dist_rel_moon: jax.Array, params: Params
):
    bw = jax.vmap(blending_weight, in_axes=(0, 0, None))(t, y, params)

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True, sharex=True)

    axs[0].plot(t / 86400, bw, label="Blending weight")
    axs[0].set_ylabel("Blending weight")
    axs[0].grid()

    axs[1].plot(t / 86400, dist_rel_moon / 1e3, label="Distance to Moon")
    axs[1].set_ylabel("Distance to Moon [km]")
    axs[1].grid()

    axs[1].set_xlabel("Time [days]")
