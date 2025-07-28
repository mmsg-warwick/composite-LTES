import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from matplotlib import colormaps
from scipy.interpolate import griddata

from composite_ltes import MushModel, SharpFrontModel, root_dir


def plot_COMSOL_data(file, model):
    # Load data
    data = pd.read_csv(root_dir() / "data" / f"cell_data_{file}_H.csv")

    # Extract x, y, and time columns
    x_comsol = data["X"].to_numpy()
    y_comsol = data["Y"].to_numpy()
    time_columns = data.columns[2:]
    t_end = float(time_columns[-1])

    # Run model
    simulation = pybamm.Simulation(model)
    solution = simulation.solve(
        [0, t_end], t_interp=np.linspace(0, t_end, len(time_columns))
    )

    # Create a regular grid for interpolation
    x = np.linspace(
        x_comsol.min(), x_comsol.max(), 100
    )  # Adjust the resolution as needed
    y = np.linspace(y_comsol.min(), y_comsol.max(), 100)
    x_grid, y_grid = np.meshgrid(x, y)

    # Choose which contour level to plot (it's scaled with latent heat)s
    contour_level = 1

    # Define plotting settings
    cmap = colormaps["plasma"]
    colors = cmap(np.linspace(0, 0.9, len(time_columns)))
    plt.rcParams.update({"font.size": 14})

    # Create figures
    fig, ax = plt.subplots()  # Main contour plot (multiple times)
    fig2, ax2 = plt.subplots()  # Contour plot for each time step

    # Interpolate data for each time step and plot
    for time, color in zip(time_columns, colors):
        # Clear ax2 for each time step
        ax2.clear()

        # Plot the PyBaMM solution
        if isinstance(model, MushModel):
            ax.contour(
                x,
                y,
                solution["PCM enthalpy"](t=float(time), x=x, y=y).T,
                levels=[contour_level],
                colors="black",
                alpha=0.5,
                linestyles="--",
            )
            ax2.contour(
                x,
                y,
                solution["PCM enthalpy"](t=float(time), x=x, y=y).T,
                levels=[contour_level],
                colors="black",
                alpha=0.5,
                linestyles="--",
            )
        elif isinstance(model, SharpFrontModel):
            param = simulation.parameter_values
            # H0 = param["kappa"] * (1 - param["Copper volume fraction"])
            H0 = 1 - param["Copper volume fraction"]
            H_sol = solution["Composite enthalpy"](t=float(time), y=y)
            ax.contour(
                [x[0], x[-1]],
                y,
                np.vstack((H_sol, H_sol)).T,
                levels=[contour_level * H0],
                colors="black",
                alpha=0.5,
                linestyles="--",
            )
            ax2.contour(
                [x[0], x[-1]],
                y,
                np.vstack((H_sol, H_sol)).T,
                levels=[contour_level * H0],
                colors="black",
                alpha=0.5,
                linestyles="--",
            )
        else:
            raise ValueError("Model {} not recognised")

        # Interpolate the data to create a smooth function H(x, y)
        H = data[time].to_numpy()
        H_grid = griddata((x_comsol, y_comsol), H, (x_grid, y_grid), method="linear")

        # Plot the interpolated data
        ax.contour(x_grid, y_grid, H_grid, levels=[contour_level], colors=[color])
        ax2.contour(x_grid, y_grid, H_grid, levels=[contour_level], colors=[color])

        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        fig2.savefig(
            root_dir() / "figures" / f"contour_{file}" / f"contour_{file}_{time}.png",
            dpi=300,
        )  # Save the individual contour plot

    # Add labels and legend
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    # Save figure
    fig.savefig(root_dir() / "figures" / f"contour_{file}.png", dpi=300)


# Loop over files and plot
models = [("mush", MushModel()), ("sharp", SharpFrontModel())]

for file, model in models:
    plot_COMSOL_data(file, model)
