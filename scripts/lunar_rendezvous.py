import jax.numpy as jnp

from qlipper.configuration import SimConfig
from qlipper.postprocess import postprocess_run
from qlipper.run.diagnostics import single_step_debug
from qlipper.run.mission_runner import run_mission

sim_case = SimConfig(
    name="lunar",
    y0=jnp.array([100000e3, 0, 0, 0, 0, 0]),
    y_target=jnp.array([20000e3, 0, 0, 0, 0]),
    propulsion_model="constant_thrust",
    steering_law="bbq_law",
    t_span=(0, 86400 * 200),
    conv_tol=0.5,
    earth_w_oe=jnp.array([10, 1, 1, 1, 1]),
    earth_penalty_weight=1,
    earth_penalty_scaling=100,
    earth_rp_min=6400e3,
    moon_w_oe=jnp.array([1, 1, 1, 0, 0]),
    moon_penalty_weight=1,
    moon_penalty_scaling=100,
    moon_rp_min=3000e3,
    perturbations=["moon_gravity"],
    epoch_jd=2451545.0,
    characteristic_accel=0.001,
    ephemeris="cr3bp",
)

# print(single_step_debug(sim_case))
secondary_params = {
    "conv_tol": 0.1,
    "propulsion_model": "no_thrust",
    # "y_target": jnp.array([10000e3, 0, 0, 0, 0]),
    "t_span": (0, 86400 * 30),
    # "moon_w_oe": jnp.array([1, 1, 5, 1, 1]),
}
# secondary_params = {}
run_id, t, y = run_mission(sim_case, overrun_params=secondary_params)
postprocess_run(run_id, t, y, sim_case)
