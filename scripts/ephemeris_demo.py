from jplephem.calendar import compute_julian_date
from matplotlib import pyplot as plt

from qlipper.sim.ephemeris import generate_ephem_arrays, lookup_body_id

barycenter = lookup_body_id("solar system barycenter")

epoch = compute_julian_date(2022, 3, 15)

t_span = (0, 86400 * 365)

bodies_to_plot = ["sun", "earth", "moon"]
colours = ["yellow", "blue", "gray"]

for body, colour in zip(bodies_to_plot, colours):
    b_id = lookup_body_id(body)
    _, y = generate_ephem_arrays(barycenter, b_id, epoch, t_span, 1000)

    plt.plot(y[0, :], y[1, :], label=body, color=colour)

    # put a dot at the end of the trajectory
    plt.scatter(y[0, -1], y[1, -1], color=colour)

plt.title("Solar System Barycenter Ephemeris")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.show()
