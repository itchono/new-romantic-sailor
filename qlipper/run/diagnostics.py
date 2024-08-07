from diffrax import (
    ODETerm,
    Tsit5,
)
from jax import Array

from qlipper.configuration import SimConfig
from qlipper.run.mission_runner import preprocess_y0
from qlipper.run.prebake import (
    prebake_ode,
    prebake_sim_config,
)


def single_step_debug(cfg: SimConfig, step_time: float = 1) -> tuple[str, Array, Array]:
    """
    Run a single step of the simulation with the given configuration.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.
    step_time : float
        Time to step the simulation forward.

    Returns
    -------
    y : Array, shape (N, 6)
        State vector at the end of one step.
    """

    # prebake
    term = ODETerm(prebake_ode(cfg))
    ode_args = prebake_sim_config(cfg)
    y0 = preprocess_y0(cfg)

    # TODO: improve diagnostics tooling, maybe put into unit tests?
    print(y0)

    solver = Tsit5()

    state = solver.init(term, *cfg.t_span, y0=y0, args=ode_args)

    y, _, _, state, _ = solver.step(
        term, 0, step_time, y0, ode_args, state, made_jump=False
    )

    return y
