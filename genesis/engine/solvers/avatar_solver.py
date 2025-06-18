import numpy as np
import taichi as ti

from genesis.engine.entities import AvatarEntity
from genesis.engine.states.solvers import AvatarSolverState

from .base_solver import Solver
from .rigid.rigid_solver import RigidSolver


@ti.data_oriented
class AvatarSolver(RigidSolver):
    """
    Avatar, similar to Rigid, maintains a kinematic tree but does not consider actual physics.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        Solver.__init__(self, scene, sim, options)

        # options
        self._enable_collision = options.enable_collision
        self._enable_self_collision = options.enable_self_collision
        self._max_collision_pairs   = options.max_collision_pairs

        self._use_hibernation = options.use_hibernation
        self._hibernation_thresh_vel = options.hibernation_thresh_vel
        self._hibernation_thresh_acc = options.hibernation_thresh_acc

        self._options               = options

    def _init_mass_mat(self):
        self.entity_max_dofs = max([entity.n_dofs for entity in self._entities])

    def _init_invweight(self):
        pass

    def update_body(self):
        self._kernel_forward_kinematics()
        self._kernel_update_geoms()

    def substep(self):
        self._kernel_step()

    def _init_constraint_solver(self):
        self.constraint_solver = None

    @ti.kernel
    def _kernel_step(self):
        self._func_integrate()
        self._func_forward_kinematics()
        self._func_update_geoms()
        if self._enable_collision:
            self._func_detect_collision()

    @ti.kernel
    def _kernel_forward_kinematics_links_geoms(self):
        self._func_forward_kinematics()
        self._func_update_geoms()

    @ti.func
    def _func_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    def set_state(self, f, state, envs_idx):
        if self.is_active():
            self._kernel_set_state(state.qpos, state.dofs_vel, state.links_pos, state.links_quat)
            self._kernel_forward_kinematics_links_geoms()
            self.collider.reset()

    def get_state(self, f):
        if self.is_active():
            state = AvatarSolverState(self.scene)
            self._kernel_get_state(state.qpos, state.dofs_vel, state.links_pos, state.links_quat)
        else:
            state = None
        return state

    def detect_collision(self):
        batch_idx = 0
        n_collision = self.collider.n_contacts.to_numpy()[batch_idx]
        if n_collision > 0:
            collision_pairs = np.empty((n_collision, 2), dtype=np.int32)
            collision_pairs[:, 0] = self.collider.contact_data.geom_a.to_numpy()[:n_collision, batch_idx]
            collision_pairs[:, 1] = self.collider.contact_data.geom_b.to_numpy()[:n_collision, batch_idx]
            return collision_pairs
        else:
            return np.array([])