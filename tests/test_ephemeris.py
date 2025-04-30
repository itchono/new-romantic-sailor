import numpy as np
import pytest
from jplephem.calendar import compute_julian_date

from qlipper.sim.ephemeris import (
    _ensure_ephemeris,
    compute_kernel_at_times,
    generate_ephem_arrays,
    path_to_named_string,
    resolve_spk_path,
)


@pytest.fixture
def kernel():
    return _ensure_ephemeris()


def test_retrieval(kernel):
    assert kernel is not None


def test_resolve_spk_path(kernel):
    path = resolve_spk_path(kernel, 399, 4)
    assert path == [399, 3, 0, 4]


def test_path_to_named_string():
    path = [399, 3, 0, 4]
    assert (
        path_to_named_string(path)
        == "Earth (399) -> Earth Barycenter (3) -> Solar System Barycenter (0) -> Mars Barycenter (4)"
    )


def test_simple_ephemeris(kernel):
    position_ref = kernel[0, 4].compute(2457061.5)
    position = compute_kernel_at_times(kernel, 0, 4, [2457061.5])[:3, 0]

    assert position_ref == pytest.approx(position, rel=1e-6)


def test_multileg_ephemeris(kernel):
    position_ref = kernel[0, 4].compute(2457061.5)
    position_ref -= kernel[0, 3].compute(2457061.5)
    position_ref -= kernel[3, 399].compute(2457061.5)
    position = compute_kernel_at_times(kernel, 399, 4, [2457061.5])[:3, 0]

    assert position_ref == pytest.approx(position, rel=1e-6)


def test_generating_interpolants():
    jd_epoch = compute_julian_date(2015, 2, 8)

    t, y = generate_ephem_arrays(0, 4, jd_epoch, [0, 86400 * 3], 4)

    assert t[0] == 0
    assert t[-1] == 86400 * 3
    assert y.shape == (6, 4)

    ref_position = np.array(
        [
            [2.057e08, 2.053e08, 2.049e08, 2.045e08],
            [4.251e07, 4.453e07, 4.654e07, 4.855e07],
            [1.394e07, 1.487e07, 1.581e07, 1.674e07],
        ]
    )

    assert y[:3, :] == pytest.approx(ref_position, rel=5e-4)
