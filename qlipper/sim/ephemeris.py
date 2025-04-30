import logging
import urllib.request
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jplephem.names import target_name_pairs, target_names
from jplephem.spk import SPK, Segment
from numpy.typing import NDArray

DE440S_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
EPHEM_CACHE_DIR = Path(__file__).parent / "kernel_cache"


logger = logging.getLogger(__name__)


def _ensure_ephemeris() -> SPK:
    # cache the ephemeris file
    EPHEM_CACHE_DIR.mkdir(exist_ok=True)

    ephemeris_file = EPHEM_CACHE_DIR / "de440s.bsp"

    if not ephemeris_file.exists():
        logger.info("First Time Setup: Downloading DE440S ephemeris...")
        urllib.request.urlretrieve(DE440S_URL, ephemeris_file)

    return SPK.open(ephemeris_file)


def find_path_bfs(start: int, end: int, graph: dict[int, list[int]]) -> list[int]:
    """
    Breadth-first search to find a path between two nodes in a graph

    Parameters
    ----------
    start : int
        The starting node
    end : int
        The ending node
    graph : dict[int, list[int]]
        The graph as an adjacency list

    Returns
    -------
    list[int]
        The path from start to end
    """
    queue = [(start, [])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in set(graph[vertex]) - set(path):
            if next == end:
                return path + [vertex, next]
            else:
                queue.append((next, path + [vertex]))

    raise ValueError(f"No path found between {start} and {end}")


def resolve_spk_path(kernel: SPK, observer: int, target: int) -> list[int]:
    """
    Determine the path to compute the kernel

    Parameters
    ----------
    kernel : SPK
        The SPK kernel to use
    observer : int
        The observer body ID
    target : int
        The target body ID

    Returns
    -------
    list[int]
        The path from observer to target
    """
    adj_list = defaultdict(list)

    for bo, bt in kernel.pairs.keys():
        adj_list[bo].append(bt)
        adj_list[bt].append(bo)

    assert observer in adj_list, f"Observer body {observer} not found in the SPK kernel"
    assert target in adj_list, f"Target body {target} not found in the SPK kernel"

    return find_path_bfs(observer, target, adj_list)


def path_to_named_string(path: list[int]) -> str:
    """
    Convert a path to a human-readable string

    Parameters
    ----------
    path : list[int]
        The path to convert

    Returns
    -------
    str
        The human-readable string
    """
    return " -> ".join(
        [f"{target_names[body_id].title()} ({body_id})" for body_id in path]
    )


def compute_kernel_at_times(
    kernel: SPK,
    observer: int,
    target: int,
    jd_samples: ArrayLike,
) -> Array:
    """
    Evaluate kernel at given times

    Automatically resolves the correct set of segments to use for the interpolation

    Parameters
    ----------
    kernel : SPK
        The SPK kernel to use
    observer : int
        The observer body ID
    target : int
        The target body ID
    jd_samples : ArrayLike
        The Julian dates to evaluate the kernel at
        NOTE: this array must be mutable, so JAX arrays will not work!

    Returns
    -------
    ArrayLike
        The cartesian state vectors of the target body at the given times,
        of shape (6, len(jd_samples))
    """
    assert not isinstance(jd_samples, jnp.ndarray), "jd_samples must be a numpy array"

    path = resolve_spk_path(kernel, observer, target)

    result = jnp.zeros((6, len(jd_samples)))

    # jplephem gives km/day, we want km/s
    ONE_DAY = 86400

    # the kernel only stores "forward pairs",
    # so we try to evaluate the segment in the forward direction first,
    # and if it fails, we try the reverse direction, with a sign flip
    # this is the EAFP way to do it
    for i in range(0, len(path) - 1):
        try:
            segment: Segment = kernel[path[i], path[i + 1]]
            pos, vel = segment.compute_and_differentiate(jd_samples)
            result += jnp.concat([pos, vel / ONE_DAY])
        except KeyError:
            segment: Segment = kernel[path[i + 1], path[i]]
            pos, vel = segment.compute_and_differentiate(jd_samples)
            result -= jnp.concat([pos, vel / ONE_DAY])

    return result


def lookup_body_id(body_name: str) -> int:
    """
    Lookup the body ID from the body name

    Parameters
    ----------
    body_name : str
        The body name

    Returns
    -------
    int
        The body ID, if found
    """

    for body_id, name in target_name_pairs:
        if name.lower() == body_name.lower():
            return body_id

    raise ValueError(f"Body name {body_name} not found in the SPICE kernel")


def generate_ephem_arrays(
    observer: int, target: int, epoch: float, t_span: ArrayLike, num_samples: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Top-level interface for generating SPICE ephemeris (interpolant) arrays.
    Given a jd epoch and a time span in seconds, a pair of interpolant
    arrays will be generated, measuring the position of the target body
    relative to the observer body.

    If you need help getting the epoch jd from a Gregorian date, you can
    use the `jplephem.calendar.compute_julian_date` function.

    Parameters
    ----------
    observer : int
        The observer body ID
    target : int
        The target body ID
    epoch : float
        The Julian date epoch
    t_span : ArrayLike
        The time span in seconds

    Returns
    -------
    t, states : tuple[NDArray, NDArray]
        The time array (in elapsed seconds) and the state array, usually in km.
        Shape is (6, num_samples) for the state array.
    """

    kernel = _ensure_ephemeris()

    assert len(t_span) == 2, "t_span must be a 2-tuple"

    # log compute path
    path = resolve_spk_path(kernel, observer, target)
    logger.debug(f"Ephemeris compute path: {path_to_named_string(path)}")

    # have to use numpy here because jplephem mutates the input
    t_samples = np.linspace(*t_span, num_samples)
    jd_samples = epoch + t_samples / 86400

    state_samples = compute_kernel_at_times(kernel, observer, target, jd_samples)

    return t_samples, state_samples
