import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def l2_loss(y: ArrayLike, y_target: ArrayLike, weights: ArrayLike) -> float:
    """
    L2 norm loss function for guidance.

    Parameters
    ----------
    y : ArrayLike
        State vector in modified equinoctial elements.
    y_target : ArrayLike
        Target state vector.
    weights : ArrayLike
        Weights for each state variable.

    Returns
    -------
    loss : float
        guidance loss.

    Notes
    -----
    Normalizes SMA by LENGTH_SCALING.
    """
    w_filter = jnp.array(weights) > 0

    return jnp.linalg.norm(
        (y[:5] - y_target[:5]).at[0].divide(y_target[0]) * w_filter,  # noqa: PD008
        ord=2,
    )
