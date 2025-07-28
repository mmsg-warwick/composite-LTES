import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from composite_ltes import SharpFrontModel, MushModel, root_dir

# Uncomment the following line to set the logging level to DEBUG
# pybamm.set_logging_level("DEBUG")

# Initialise the sharp-front model
model = SharpFrontModel()

# Set kappa to be a parameter to pass on solving
parameter_values = model.default_parameter_values
parameter_values.update({"kappa": "[input]"})

# Add an event to terminate when the melting is complete
model.events.append(
    pybamm.Event(
        "100% SoC",
        1 - model.variables["SoC"],
        pybamm.EventType.TERMINATION,
    )
)

# Create a simulation object
simulation = pybamm.Simulation(model, parameter_values=parameter_values)

# Define the range of kappa values for the sharp-front model and loop over
kappas_sharp = np.logspace(-3, 3, 50)
times_sharp = []

for kappa in kappas_sharp:
    print(f"Solving sharp model for kappa={kappa:2e}")
    solution = simulation.solve(
        [0, 0.7],
        inputs={"kappa": kappa},
    )

    times_sharp.append(solution.t[-1])
    print(solution.solve_time)

# Save the results to a CSV file
pd.DataFrame(
    {
        "Kappa": kappas_sharp,
        "Melting time": times_sharp,
    }
).to_csv(root_dir() / "data" / "melting_times_sharp.csv", index=False)

# Initialise the homogenised model
model = MushModel()

# Set kappa to be a parameter to pass on solving
parameter_values = model.default_parameter_values
parameter_values.update({"kappa": "[input]"})

# Add an event to terminate when the melting is complete
model.events.append(
    pybamm.Event(
        "100% SoC",
        1 - 1e-3 - model.variables["SoC"],
        pybamm.EventType.TERMINATION,
    )
)

# Create a simulation object
simulation = pybamm.Simulation(model, parameter_values=parameter_values)

# Define the range of kappa values for the homogenised model and loop over
kappas_mush = np.logspace(-5, 0, 50)
times_mush = []

for kappa in kappas_mush:
    print(f"Solving mush model for kappa={kappa:.2e}")
    solution = simulation.solve(
        [0, 1.5],
        t_interp=np.linspace(0, 1.5, 100),
        inputs={"kappa": kappa}
    )
    times_mush.append(solution.t[-1])
    print(solution.solve_time)

# Save the results to a CSV file
pd.DataFrame(
    {
        "Kappa": kappas_mush,
        "Melting time": times_mush,
    }
).to_csv(root_dir() / "data" / "melting_times_mush.csv", index=False)