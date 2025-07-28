"""
Copyright (c) 2025 Ferran Brosa Planella. All rights reserved.

composite-LTES: A project to simulate composite latent thermal energy storage systems.
"""
__version__ = "0.1.0"

import pybamm

from .models import MushModel, SharpFrontModel
from .utils import root_dir

__all__ = [
    "__version__",
    "pybamm",
    "models",
    "utils",
]
