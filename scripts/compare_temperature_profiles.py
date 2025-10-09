import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from matplotlib import colormaps
from matplotlib.lines import Line2D

from composite_ltes import MushModel, SharpFrontModel, root_dir

plt.rcParams.update({"font.size": 14})
cmap = colormaps["plasma"]
fig, ax = plt.subplots()

# Load data from COMSOL
file = "sharp"
data = pd.read_csv(root_dir() / "data" / f"cell_data_{file}_u.csv")
times = [float(x) for x in data.columns[2:].to_numpy()]
t_end = 1.1 * times[-1] # slightly beyond last COMSOL time

# Run both reduced models to compare
models = [MushModel(), SharpFrontModel()]
ys = [0.5, 1.0]
colors = [cmap(0), cmap(0.8)]

for model in models:
    t_sim = np.linspace(0, t_end, 50)
    simulation = pybamm.Simulation(model)
    solution = simulation.solve(
        [0, t_end], t_interp=t_sim
    )
    for y, color in zip(ys, colors):
        if isinstance(model, MushModel):
            label = "Mush" if y == 0.5 else None
            ax.plot(
                t_sim,
                solution["PCM temperature"](t=t_sim, x=0.1, y=y).squeeze(),
                label=label,
                color=color,
                linestyle="-",
            )
        elif isinstance(model, SharpFrontModel):
            label = "Sharp" if y == 0.5 else None
            ax.plot(
                t_sim,
                solution["Composite temperature"](t=t_sim, y=y).squeeze(),
                label=label,
                color=color,
                linestyle="--",
            )

# Plot model comparison
for y, color in zip(ys, colors):
    row = data[(data.iloc[:, 0] == 0.1) & (data.iloc[:, 1] == y)]
    temperatures = row.iloc[0, 2:].to_numpy()
    ax.plot(times, temperatures, "x", color=color, label=f"COMSOL z={y}")

ax.set_xlabel("Time")
ax.set_ylabel("Temperature")
ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))

custom_handles = [
    Line2D([0], [0], color="k", linestyle="-", label="Mush"),
    Line2D([0], [0], color="k", linestyle="--", label="Sharp-front"),
    Line2D([0], [0], color="k", marker="x", linestyle="None", markersize=8, label="Microscale"),
    Line2D([0], [0], color=colors[0], marker="s", linestyle="None", markersize=8, label="@ y=0.5"),
    Line2D([0], [0], color=colors[1], marker="s", linestyle="None", markersize=8, label="@ y=1.0"),
]

ax.legend(handles=custom_handles)

fig.tight_layout()
fig.savefig(root_dir() / "figures" / f"temperature_profile_{file}.png", dpi=300)

# plt.show()
