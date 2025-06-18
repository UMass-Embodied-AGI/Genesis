import taichi as ti

from ..rigid_entity import RigidJoint
import numpy as np


def default_solver_params(n=6, substep_dt=0.01):
    """
    Default solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power). Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    Note that timeconst here will not be used in the current workflow. Instead, it will be computed using 2 * substep_dt.
    """

    solver_params = np.array([2 * substep_dt, 1.0e00, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e00])
    return np.repeat(solver_params[None], n, axis=0)

@ti.data_oriented
class AvatarJoint(RigidJoint):
    """AvatarJoint resembles RigidJoint in rigid_solver, but is only used for collision checking."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dofs_sol_params = default_solver_params(self.n_dofs)
    pass
