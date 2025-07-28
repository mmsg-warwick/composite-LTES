import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

from composite_ltes import root_dir

# Load the data from CSV files
data_sharp = pd.read_csv(root_dir() / "data" / "melting_times_sharp.csv")
data_mush = pd.read_csv(root_dir() / "data" / "melting_times_mush.csv")
data_comsol = pd.read_csv(root_dir() / "data" / "melting_times_comsol.csv")

# Define the plotting settings
plt.rcParams.update({"font.size": 14})
cmap = colormaps["plasma"]

# Calculate theoretical melting times for sharp-front and mush models
t_mush = (8e-2 / 0.876901) ** 2 / 2
t_sharp = (1 / 0.855015) ** 2 / 2

# Plot the melting times
fig, ax = plt.subplots()
ax.loglog(
    data_sharp["Kappa"], data_sharp["Melting time"], color=cmap(0), label="Sharp-front"
)
ax.loglog(data_mush["Kappa"], data_mush["Melting time"], color=cmap(0.8), label="Mush")
ax.loglog(data_comsol["Kappa"], data_comsol["Melting time"], "xk", label="Microscale")
ax.loglog(
    [1e-2, data_sharp["Kappa"].iloc[-1]],
    [t_sharp, t_sharp],
    linestyle="--",
    color="grey",
    label="Theoretical limit",
)
ax.loglog(
    [data_mush["Kappa"].iloc[0], 1e-1],
    [t_mush, t_mush],
    linestyle="--",
    color="grey",
)

ax.axvline(x=0.04, color="black", linestyle=":")
ax.text(
    0.05,
    ax.get_ylim()[0] * 1.2,
    r"$\kappa = \theta$",
    color="black",
    # rotation=90,
    va="bottom",
    ha="left",
)

ax.set_xlabel(r"$\kappa$")
ax.set_ylabel("Melting time")
ax.legend()
fig.tight_layout()

# Save the figure
fig.savefig(root_dir() / "figures" / "melting_times.png", dpi=300)

plt.show()
