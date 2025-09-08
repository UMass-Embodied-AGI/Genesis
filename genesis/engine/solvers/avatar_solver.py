import numpy as np
import gstaichi as ti
import genesis as gs
from genesis.engine.states.solvers import AvatarSolverState

from .rigid.rigid_solver_decomp import RigidSolver
from .base_solver import Solver
# Minimum ratio between simulation timestep `_substep_dt` and time constant of constraints
TIME_CONSTANT_SAFETY_FACTOR = 2.0

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
        self._enable_multi_contact = options.enable_multi_contact
        self._enable_mujoco_compatibility = options.enable_mujoco_compatibility
        self._enable_joint_limit = options.enable_joint_limit
        self._enable_self_collision = options.enable_self_collision
        self._enable_adjacent_collision = options.enable_adjacent_collision
        self._disable_constraint = options.disable_constraint
        self._max_collision_pairs = options.max_collision_pairs
        self._integrator = options.integrator
        self._box_box_detection = options.box_box_detection

        self._use_contact_island = options.use_contact_island
        self._use_hibernation = options.use_hibernation and options.use_contact_island
        if options.use_hibernation and not options.use_contact_island:
            gs.logger.warning(
                "`use_hibernation` is set to False because `use_contact_island=False`. Please set "
                "`use_contact_island=True` if you want to use hibernation"
            )

        self._hibernation_thresh_vel = options.hibernation_thresh_vel
        self._hibernation_thresh_acc = options.hibernation_thresh_acc

        self._sol_min_timeconst = TIME_CONSTANT_SAFETY_FACTOR * self._substep_dt
        self._sol_global_timeconst = options.constraint_timeconst
        if options.contact_resolve_time is not None:
            self._sol_global_timeconst = options.contact_resolve_time
            gs.logger.warning(
                "Rigid option 'contact_resolve_time' is deprecated and will be remove in future release. Please use "
                "'constraint_timeconst' instead."
            )
        self._options = options

    def _init_invweight(self):
        pass

    def update_body(self):
        self._kernel_forward_kinematics()
        self._kernel_update_geoms()

    def substep(self):
        self._kernel_step()
        # constraint force
        self._func_constraint_force()

    @ti.kernel
    def _kernel_step(self):
        self._func_integrate()
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            self._func_forward_kinematics(i_b)
            self._func_update_geoms(i_b)

    @ti.kernel
    def _kernel_forward_kinematics_links_geoms(self, envs_idx: ti.types.ndarray()):
        for i_b in envs_idx:
            self._func_forward_kinematics(i_b)
            self._func_update_geoms(i_b)

    @ti.func
    def _func_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    def get_state(self, f):
        if self.is_active():
            state = AvatarSolverState(self.scene)
            self._kernel_get_state(
                state.qpos,
                state.dofs_vel,
                state.links_pos,
                state.links_quat,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
        else:
            state = None
        return state

    def print_contact_data(self):
        batch_idx = 0
        n_contacts = self.collider._collider_state.n_contacts[batch_idx]
        print("collision_pairs:")
        if n_contacts > 0:
            contact_data = self.collider._collider_state.contact_data.to_numpy()
            links_a = contact_data["link_a"][:n_contacts, batch_idx]
            links_b = contact_data["link_b"][:n_contacts, batch_idx]
            link_pairs = np.vstack([links_a, links_b]).T
            print(link_pairs)
