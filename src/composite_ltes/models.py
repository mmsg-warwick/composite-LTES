#
# 1D model for composite LTES
#
import pybamm


class MushModel(pybamm.models.base_model.BaseModel):
    def __init__(self, name="Mush model"):
        super().__init__(name=name)

        ######################
        # Variables
        ######################
        self.x = pybamm.SpatialVariable(
            "x",
            domains={"primary": "PCM", "secondary": "copper"},
            coord_sys="cartesian",
        )
        self.y = pybamm.SpatialVariable(
            "y", domains={"primary": "copper"}, coord_sys="cartesian"
        )
        T = pybamm.Variable(
            "PCM temperature", domains={"primary": "PCM", "secondary": "copper"}
        )
        H = pybamm.Variable(
            "PCM enthalpy", domains={"primary": "PCM", "secondary": "copper"}
        )
        T_c = pybamm.Variable("Copper temperature", domains={"primary": "copper"})

        ######################
        # Parameters
        ######################
        self.St = pybamm.Parameter("Stefan number")
        self.T_init = pybamm.Parameter("Initial temperature")
        self.T_b = pybamm.Parameter("Boundary temperature")
        self.theta = pybamm.Parameter("theta")
        self.r = pybamm.Parameter("r")
        self.kappa = pybamm.Parameter("kappa")
        self.epsilon = pybamm.Parameter("epsilon")
        self.delta = self.theta / self.epsilon

        ######################
        # Equations
        ######################
        q = (
            self.kappa
            * pybamm.BoundaryGradient(T, "left")
            / (self.epsilon * self.delta)
        )
        dTcdt = (pybamm.div(pybamm.grad(T_c)) + q) / (self.kappa * self.r * self.St)
        dHdt = pybamm.div(pybamm.grad(T))
        self.rhs = {T_c: dTcdt, H: dHdt}
        self.algebraic = {T: T - self.H2T(H)}

        self.boundary_conditions = {
            T: {
                "left": (T_c, "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
            T_c: {
                "left": (self.T_b, "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
        }

        self.initial_conditions = {
            T: self.T_init,
            H: self.T2H(self.T_init),
            T_c: self.T_init,
        }

        ######################
        # Output variables
        ######################
        phase = H >= 1 / 2
        ones_xy = pybamm.FullBroadcast(
            pybamm.Scalar(1),
            broadcast_domains={"primary": "PCM", "secondary": "copper"},
        )
        SoC = pybamm.Integral(pybamm.Integral(phase, self.x), self.y) / pybamm.Integral(
            pybamm.Integral(ones_xy, self.x), self.y
        )

        self.variables = {
            "PCM temperature": T,
            "Copper temperature": T_c,
            "PCM enthalpy": H,
            "PCM temperature at centre": pybamm.boundary_value(T, "right"),
            "x": self.x,
            "y": self.y,
            "t": pybamm.t,
            "SoC": SoC,
        }

    def H2T(self, H):
        """Convert enthalpy to temperature"""
        solid = H / self.St
        liquid = (H - 1) / self.St
        return pybamm.minimum(solid, 0) + pybamm.maximum(liquid, 0)

    def T2H(self, T):
        """Convert temperature to enthalpy"""
        if T == 0:
            msg = "Enthalpy is not uniquely defined at melting temperature"
            raise ValueError(msg)
        H_l = self.St * T + 1
        H_s = self.St * T
        return H_l * (T > 0) + H_s * (T < 0)

    @property
    def default_geometry(self):
        return pybamm.Geometry(
            {
                "PCM": {self.x: {"min": self.theta / 2, "max": self.epsilon / 2}},
                "copper": {self.y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
            }
        )

    @property
    def default_submesh_types(self):
        return {
            "PCM": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "copper": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

    @property
    def default_var_pts(self):
        return {self.x: 40, self.y: 20}

    @property
    def default_spatial_methods(self):
        return {
            "PCM": pybamm.FiniteVolume(),
            "copper": pybamm.FiniteVolume(),
        }

    @property
    def default_solver(self):
        return pybamm.IDAKLUSolver()

    @property
    def default_quick_plot_variables(self):
        return [
            "PCM temperature",
            "Copper temperature",
            "PCM enthalpy",
            "PCM temperature at centre",
            "SoC",
        ]

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(
            {
                "Stefan number": 1,
                "Initial temperature": 0,
                "Boundary temperature": 1,
                "theta": 0.04,
                "epsilon": 0.2,
                "r": 1,
                "kappa": 1e-3,
            }
        )


class SharpFrontModel(pybamm.models.base_model.BaseModel):
    def __init__(self, name="Sharp-front model"):
        super().__init__(name=name)

        ######################
        # Variables
        ######################
        self.y = pybamm.SpatialVariable(
            "y", domains={"primary": "composite"}, coord_sys="cartesian"
        )
        T = pybamm.Variable("Composite temperature", domains={"primary": "composite"})
        H = pybamm.Variable("Composite enthalpy", domains={"primary": "composite"})

        ######################
        # Parameters
        ######################
        self.St = pybamm.Parameter("Stefan number")
        self.T_init = pybamm.Parameter("Initial temperature")
        self.T_b = pybamm.Parameter("Boundary temperature")
        self.r = pybamm.Parameter("r")
        self.kappa = pybamm.Parameter("kappa")
        self.V_C = pybamm.Parameter("Copper volume fraction")
        self.D = (1 - self.V_C) + self.V_C / self.kappa

        ######################
        # Equations
        ######################
        dHdt = pybamm.div(self.D * pybamm.grad(T))
        self.rhs = {H: dHdt}
        self.algebraic = {T: T - self.H2T(H)}

        self.boundary_conditions = {
            T: {
                "left": (self.T_b, "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
        }

        self.initial_conditions = {
            T: self.T_init,
            H: self.T2H(self.T_init),
        }

        ######################
        # Output variables
        ######################
        phase = H >= (1 - self.V_C) / 2

        ones_y = pybamm.FullBroadcast(
            pybamm.Scalar(1), broadcast_domains={"primary": "composite"}
        )
        SoC = pybamm.Integral(phase, self.y) / pybamm.Integral(ones_y, self.y)

        self.variables = {
            "Composite temperature": T,
            "Composite enthalpy": H,
            "Composite temperature at centre": pybamm.boundary_value(T, "right"),
            "Composite enthalpy at centre": pybamm.boundary_value(H, "right"),
            "SoC": SoC,
            "y": self.y,
        }

    def H2T(self, H):
        """Convert enthalpy to temperature"""
        a = self.St * (self.r * self.V_C + (1 - self.V_C))
        b = 1 - self.V_C
        solid = H / a
        liquid = (H - b) / a
        return pybamm.minimum(solid, 0) + pybamm.maximum(liquid, 0)

    def T2H(self, T):
        """Convert temperature to enthalpy"""
        if T == 0:
            msg = "Enthalpy is not uniquely defined at melting temperature"
            raise ValueError(msg)
        a = self.St * (self.r * self.V_C + (1 - self.V_C))
        b = 1 - self.V_C
        H_l = a * T + b
        H_s = a * T
        return H_l * (T > 0) + H_s * (T < 0)

    @property
    def default_geometry(self):
        return pybamm.Geometry(
            {
                "composite": {
                    self.y: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
                },
            }
        )

    @property
    def default_submesh_types(self):
        return {
            "composite": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

    @property
    def default_var_pts(self):
        return {self.y: 50}

    @property
    def default_spatial_methods(self):
        return {
            "composite": pybamm.FiniteVolume(),
        }

    @property
    def default_solver(self):
        return pybamm.IDAKLUSolver()

    @property
    def default_quick_plot_variables(self):
        return [
            "Composite temperature",
            "Composite enthalpy",
            "Composite temperature at centre",
            "Composite enthalpy at centre",
            "SoC",
        ]

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(
            {
                "Stefan number": 1,
                "Initial temperature": 0,
                "Boundary temperature": 1,
                "r": 1,
                "kappa": 0.1,
                "Copper volume fraction": 0.2,
            }
        )
