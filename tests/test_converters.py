from functools import partial

import jax
import jax.numpy as jnp
import pytest

from qlipper.constants import MU_EARTH
from qlipper.converters import (
    batch_cartesian_to_mee,
    batch_mee_to_cartesian,
    cartesian_to_mee,
    mee_to_cartesian,
    rot_inertial_lvlh,
)


def test_cartesian_to_mee():
    cart = jnp.array([1e6, 2e6, 3e6, 4e3, 5e3, 6e3])
    mee = cartesian_to_mee(cart, MU_EARTH)

    mee_ref = jnp.array(
        [
            1.3547e5 / (1 - 0.8809**2 - 0.4217**2),
            -0.8809,
            0.4217,
            -1.3798,
            -0.6899,
            -0.6087,
        ]
    )

    assert mee == pytest.approx(mee_ref, rel=2e-3)


def test_mee_to_cartesian():
    mee = jnp.array(
        [
            1.3547e5 / (1 - 0.8809**2 - 0.4217**2),
            -0.8809,
            0.4217,
            -1.3798,
            -0.6899,
            -0.6087,
        ]
    )
    cart = mee_to_cartesian(mee, MU_EARTH)

    cart_ref = jnp.array([1e6, 2e6, 3e6, 4e3, 5e3, 6e3])

    assert cart == pytest.approx(cart_ref, rel=1e-3)


def test_roundtrip_mee_cartesian():
    cart = jnp.array([1e6, 2e6, 3e6, 4e3, 5e3, 6e3])

    mee = cartesian_to_mee(cart, MU_EARTH)
    cart_back = mee_to_cartesian(mee, MU_EARTH)

    assert cart_back == pytest.approx(cart, rel=1e-8)


def test_converters_batched():
    cart = jnp.array([[1e6, 2e6, 3e6, 4e3, 5e3, 6e3], [1e6, 2e6, 3e6, 4e3, 5e3, 6e3]])

    mee = batch_cartesian_to_mee(cart, MU_EARTH)

    cart_back = batch_mee_to_cartesian(mee, MU_EARTH)

    assert cart_back == pytest.approx(cart, rel=1e-8)


def test_unwrapping():
    ascending_l = jnp.linspace(0, 17 * jnp.pi, 50)

    mee = jnp.array([1e5, 0, 0, 0, 0, 0])[None, :].repeat(50, axis=0)
    mee = mee.at[:, 5].set(ascending_l)

    cart = batch_mee_to_cartesian(mee, MU_EARTH)

    mee_unwrapped = batch_cartesian_to_mee(cart, MU_EARTH)

    assert jnp.allclose(mee_unwrapped[:, 5], ascending_l)
    assert jnp.diff(mee_unwrapped[:, 5]).max() < 2 * jnp.pi
    assert jnp.diff(mee_unwrapped[:, 5]).min() > 0


def test_non_unwrapping_fails():
    ascending_l = jnp.linspace(0, 17 * jnp.pi, 50)

    mee = jnp.array([1e5, 0, 0, 0, 0, 0])[None, :].repeat(50, axis=0)
    mee = mee.at[:, 5].set(ascending_l)

    cart = batch_mee_to_cartesian(mee, MU_EARTH)

    mee_out = jax.vmap(partial(cartesian_to_mee, mu=MU_EARTH))(cart)

    assert not jnp.allclose(mee_out[:, 5], ascending_l)
    assert jnp.diff(mee_out[:, 5]).min() < 0


def test_rot_inertial_lvlh():
    # Reference: MATLAB code
    mee = jnp.array([1.3547e5, -0.8809, 0.4217, -1.3798, -0.6899, -0.6087])
    cart = mee_to_cartesian(mee, MU_EARTH)

    rot = rot_inertial_lvlh(cart)

    ref = jnp.array(
        [
            [0.267232581529630, 0.872880567632737, -0.408247794872378],
            [0.534517508499001, 0.218233785274841, 0.816495589744755],
            [0.801796595451134, -0.436409818420443, -0.408250768412266],
        ]
    )

    assert rot == pytest.approx(ref, rel=1e-4)
