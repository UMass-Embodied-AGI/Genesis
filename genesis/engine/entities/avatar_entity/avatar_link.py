import numpy as np
import taichi as ti

import genesis as gs
import trimesh
from genesis.repr_base import RBC
from genesis.utils import geom as gu
from .avatar_geom import AvatarGeom, AvatarVisGeom


@ti.data_oriented
class AvatarLink(RBC):
    def __init__(
        self,
        entity,
        name,
        idx,
        joint_start,
        n_joints,
        idx_offset_q,
        idx_offset_dof,
        idx_offset_geom,
        idx_offset_cell,
        idx_offset_vert,
        idx_offset_face,
        idx_offset_edge,
        idx_offset_vgeom,
        vvert_start,
        vface_start,
        n_qs,
        n_dofs,
        pos,
        quat,
        inertial_pos,
        inertial_quat,
        inertial_i,
        inertial_mass,
        parent_idx,
        invweight,
        joint_type,
        joint_pos,
        joint_quat,
        dofs_motion_ang,
        dofs_motion_vel,
        dofs_limit,
        dofs_invweight,
        dofs_stiffness,
        dofs_damping,
        dofs_armature,
        dofs_kp,
        dofs_kv,
        dofs_force_range,
        init_q,
        root_idx
    ):
        self._name   = name
        self._entity = entity
        self._solver = entity.solver
        self._entity_idx_in_solver = entity.idx

        self._uid             = gs.UID()
        self._idx            = idx
        self._parent_idx     = parent_idx
        self._child_idxs     = list()
        self._idx_offset_q   = idx_offset_q
        self._idx_offset_dof = idx_offset_dof
        self._n_qs            = n_qs
        self._n_dofs         = n_dofs
        self._invweight      = invweight
        self._joint_type     = joint_type

        self._joint_start    = joint_start
        self._n_joints       = n_joints
        self._root_idx = root_idx

        # if self._joint_type is gs.JOINT_TYPE.FREE:
        #     self._n_dofs_controllable = 0
        # else:
        #     self._n_dofs_controllable = self._n_dofs

        self._n_dofs_controllable = self._n_dofs

        self._idx_offset_geom  = idx_offset_geom
        self._idx_offset_cell  = idx_offset_cell
        self._idx_offset_vert  = idx_offset_vert
        self._idx_offset_face  = idx_offset_face
        self._idx_offset_edge  = idx_offset_edge
        self._idx_offset_vgeom = idx_offset_vgeom
        self._vvert_start = vvert_start
        self._vface_start = vface_start

        self._pos           = pos
        self._quat          = quat
        self._inertial_pos  = inertial_pos
        self._inertial_quat = inertial_quat
        self._inertial_mass = inertial_mass
        self._joint_pos     = joint_pos
        self._joint_quat    = joint_quat

        # will be process later
        self._inertial_i = inertial_i

        self._dofs_motion_ang  = dofs_motion_ang
        self._dofs_motion_vel  = dofs_motion_vel
        self._dofs_limit       = dofs_limit
        self._dofs_invweight   = dofs_invweight
        self._dofs_stiffness   = dofs_stiffness
        self._dofs_damping     = dofs_damping
        self._dofs_armature    = dofs_armature
        self._dofs_kp          = dofs_kp
        self._dofs_kv          = dofs_kv
        self._dofs_force_range = dofs_force_range

        self._init_q = init_q

        self._geoms = gs.List()
        self._vgeoms = gs.List()

    def _build(self):
        for geom in self._geoms:
            geom._build()

        for vgeom in self._vgeoms:
            vgeom._build()

        self._init_mesh = self._get_init_composed_mesh()

        # find root link and check if link is fixed
        solver_links = self._solver.links
        link = self
        is_fixed = self._joint_type is gs.JOINT_TYPE.FIXED
        while link.parent_idx > -1:
            link = solver_links[link.parent_idx]
            if link.joint_type is not gs.JOINT_TYPE.FIXED:
                is_fixed = False
        self.root_idx = gs.np_int(link.idx)
        self.is_fixed = gs.np_int(is_fixed)

        if self._inertial_mass is None:
            if len(self._geoms) == 0 and len(self._vgeoms) == 0:
                self._inertial_mass = 0.0
            else:
                if self._init_mesh.is_watertight:
                    self._inertial_mass = self._init_mesh.volume * self.entity.material.rho
                else: # TODO: handle non-watertight mesh
                    self._inertial_mass = 1.0

        if self._invweight is None:
            if self._inertial_mass > 0:
                self._invweight = 1.0 / self.inertial_mass
            else:
                self._invweight = np.inf

        # compute inertia using all geoms/vgeoms
        if self._inertial_i is None:
            if len(self._geoms) == 0 and len(self._vgeoms) == 0: # use sphere inertia with radius 0.1
                self._inertial_i = 0.4 * self._inertial_mass * 0.1**2 * np.eye(3)

            else:
                init_verts = []
                init_faces = []
                vert_offset = 0
                if len(self._geoms) > 0:
                    for geom in self._geoms:
                        init_verts.append(gu.transform_by_trans_quat(
                            geom.init_verts, geom.init_pos, geom.init_quat
                        ))
                        init_faces.append(geom.init_faces + vert_offset)
                        vert_offset += geom.init_verts.shape[0]
                elif len(self._vgeoms) > 0:  # use vgeom if there's no geom
                    for vgeom in self._vgeoms:
                        init_verts.append(gu.transform_by_trans_quat(
                            vgeom.init_vverts, vgeom.init_pos, vgeom.init_quat
                        ))
                        init_faces.append(vgeom.init_vfaces + vert_offset)
                        vert_offset += vgeom.init_vverts.shape[0]
                init_verts = np.concatenate(init_verts)
                init_faces = np.concatenate(init_faces)
                mesh = trimesh.Trimesh(init_verts, init_faces)
                # TODO: check if this is correct. This is correct if the inertia frame is w.r.t to link frame
                T_inertia = gu.trans_quat_to_T(self._inertial_pos, self._inertial_quat)
                if np.isinf(self._inertial_mass) or mesh.mass == 0:
                    self._inertial_i = np.diag([np.inf] * 3)
                else:
                    self._inertial_i = mesh.moment_inertia_frame(T_inertia) / mesh.mass * self._inertial_mass

        self._inertial_i = np.array(self._inertial_i, dtype=gs.np_float)

        # override invweight if fixed
        if is_fixed:
            self._invweight = 0.0


    def _get_init_composed_mesh(self):
        if len(self._geoms) == 0 and len(self._vgeoms) == 0:
            return None
        else:
            init_verts = []
            init_faces = []
            vert_offset = 0
            if len(self._geoms) > 0:
                for geom in self._geoms:
                    init_verts.append(gu.transform_by_trans_quat(
                        geom.init_verts, geom.init_pos, geom.init_quat
                    ))
                    init_faces.append(geom.init_faces + vert_offset)
                    vert_offset += geom.init_verts.shape[0]
            elif len(self._vgeoms) > 0:  # use vgeom if there's no geom
                for vgeom in self._vgeoms:
                    init_verts.append(gu.transform_by_trans_quat(
                        vgeom.init_vverts, vgeom.init_pos, vgeom.init_quat
                    ))
                    init_faces.append(vgeom.init_vfaces + vert_offset)
                    vert_offset += vgeom.init_vverts.shape[0]
            init_verts = np.concatenate(init_verts)
            init_faces = np.concatenate(init_faces)
            return trimesh.Trimesh(init_verts, init_faces)

    def _add_geom(
        self, mesh, init_pos, init_quat, type, friction, sol_params, center_init=None, needs_coup=False, data=None
    ):
        geom = AvatarGeom(
            link=self,
            idx=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            mesh=mesh,
            init_pos=init_pos,
            init_quat=init_quat,
            type=type,
            friction=friction,
            sol_params=sol_params,
            center_init=center_init,
            needs_coup=needs_coup,
            data=data,
        )
        self._geoms.append(geom)

    def add_vgeom(self, init_vverts, init_vfaces, init_vnormals, init_pos, init_quat, type, init_matrix=None, vert_joints=None, vert_invbind=None, vert_weights=None, skin_joints=None, nodes=None, data=None, color=None, uvs=None, image=None, surface=None):
        vgeom = AvatarVisGeom(
            link             = self,
            idx              = self.n_vgeoms + self._idx_offset_vgeom,
            vvert_start = self.n_vverts + self._vvert_start,
            vface_start = self.n_vfaces + self._vface_start,
            init_vverts      = init_vverts,
            init_vfaces      = init_vfaces,
            init_vnormals    = init_vnormals,
            init_pos         = init_pos,
            init_quat        = init_quat,
            init_matrix      = init_matrix,
            vert_joints      = vert_joints,
            vert_invbind     = vert_invbind,
            vert_weights     = vert_weights,
            skin_joints      = skin_joints,
            nodes            = nodes,
            type             = type,
            color            = color,
            uvs              = uvs,
            image            = image,
            surface          = surface,
        )
        self._vgeoms.append(vgeom)

    def add_geom(self, mesh, init_pos, init_quat, type, friction, sol_params, center_init=None, needs_coup=False, data=None):
        geom = AvatarGeom(
            link        = self,
            idx         = self.n_geoms + self._geom_start,
            cell_start  = self.n_cells + self._cell_start,
            vert_start  = self.n_verts + self._vert_start,
            face_start  = self.n_faces + self._face_start,
            edge_start  = self.n_edges + self._edge_start,
            mesh        = mesh,
            init_pos    = init_pos,
            init_quat   = init_quat,
            type        = type,
            friction    = friction,
            sol_params  = sol_params,
            center_init = center_init,
            needs_coup  = needs_coup,
            data        = data,
        )
        self._geoms.append(geom)
    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    def get_pos(self):
        arr = np.zeros((self._solver._B, 3), dtype=gs.np_float)
        self.kernel_get_pos(arr)
        if self._solver.n_envs == 0:
            arr = arr.squeeze(0)
        return arr
    @ti.kernel
    def kernel_get_pos(self, arr: ti.types.ndarray()):
        for i, b in ti.ndrange(3, self._solver._B):
            arr[b, i] = self._solver.links_state[self._idx, b].pos[i]

    def get_quat(self):
        arr = np.zeros((self._solver._B, 4), dtype=gs.np_float)
        self.kernel_get_quat(arr)
        if self._solver.n_envs == 0:
            arr = arr.squeeze(0)
        return arr
    @ti.kernel
    def kernel_get_quat(self, arr: ti.types.ndarray()):
        for i, b in ti.ndrange(4, self._solver._B):
            arr[b, i] = self._solver.links_state[self._idx, b].quat[i]

    def get_verts(self):
        arr = np.zeros((self._solver._B, self.n_verts, 3), dtype=gs.np_float)
        self.kernel_get_verts(arr)
        if self._solver.n_envs == 0:
            arr = arr.squeeze(0)
        return arr
    @ti.kernel
    def kernel_get_verts(self, arr: ti.types.ndarray()):
        for i_g_, i_b in ti.ndrange(self.n_geoms, self._solver._B):
            i_g = i_g_ + self._idx_offset_geom
            self._solver.func_update_verts_for_geom(i_g, i_b)

        for i, j, b in ti.ndrange(self.n_verts, 3, self._solver._B):
            idx_vert = i + self._idx_offset_vert
            arr[b, i, j] = self._solver.verts_state[idx_vert, b].pos[j]

    def get_AABB(self):
        verts = self.get_verts()
        AABB = np.concatenate([verts.min(axis=-2, keepdims=True), verts.max(axis=-2, keepdims=True)], axis=-2)
        return AABB

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def name(self):
        return self._name

    @property
    def entity(self):
        return self._entity

    @property
    def solver(self):
        return self._solver

    @property
    def idx(self):
        return self._idx

    @property
    def parent_idx(self):
        return self._parent_idx

    @property
    def child_idxs(self):
        return self._child_idxs

    @property
    def idx_local(self):
        return self._idx - self._entity._idx_offset_link

    @property
    def parent_idx_local(self):
        if self._parent_idx >= 0:
            return self._parent_idx - self._entity._idx_offset_link
        else:
            return self._parent_idx

    @property
    def child_idxs_local(self):
        return [idx - self._entity._idx_offset_link if idx >= 0 else idx for idx in self._child_idxs]

    @property
    def is_leaf(self):
        return len(self._child_idxs) == 0

    @property
    def init_q(self):
        return self._init_q

    @property
    def n_qs(self):
        return self._n_qs

    @property
    def n_dofs(self):
        return self._n_dofs

    @property
    def n_dofs_controllable(self):
        return self._n_dofs_controllable

    @property
    def invweight(self):
        return self._invweight

    @property
    def joint_type(self):
        return self._joint_type

    @property
    def pos(self):
        return self._pos

    @property
    def quat(self):
        return self._quat

    @property
    def inertial_pos(self):
        return self._inertial_pos

    @property
    def inertial_quat(self):
        return self._inertial_quat

    @property
    def inertial_mass(self):
        return self._inertial_mass

    @property
    def joint_pos(self):
        return self._joint_pos

    @property
    def joint_quat(self):
        return self._joint_quat

    @property
    def inertial_i(self):
        return self._inertial_i

    @property
    def geoms(self):
        return self._geoms

    @property
    def vgeoms(self):
        return self._vgeoms

    @property
    def n_geoms(self):
        return len(self._geoms)

    @property
    def n_vgeoms(self):
        return len(self._vgeoms)

    @property
    def n_cells(self):
        return sum([geom.n_cells for geom in self._geoms])

    @property
    def n_verts(self):
        return sum([geom.n_verts for geom in self._geoms])

    @property
    def n_vverts(self):
        return sum([vgeom.n_vverts for vgeom in self._vgeoms])

    @property
    def n_faces(self):
        return sum([geom.n_faces for geom in self._geoms])

    @property
    def n_vfaces(self):
        return sum([vgeom.n_vfaces for vgeom in self._vgeoms])

    @property
    def n_edges(self):
        return sum([geom.n_edges for geom in self._geoms])

    @property
    def q_start(self):
        return self._idx_offset_q

    @property
    def dof_start(self):
        return self._idx_offset_dof

    @property
    def q_end(self):
        return self._n_qs + self.q_start

    @property
    def joints(self):
        """
        The sequence of joints that connects the link to its parent link.
        """
        return self._solver.joints[self.joint_start : self.joint_end]

    @property
    def n_joints(self):
        """
        Number of the joints that connects the link to its parent link.
        """
        return self._n_joints

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def dof_start(self):
        """The index of the link's first degree of freedom (DOF) in the scene."""
        if len(self.joints) == 0:
            return -1
        return self.joints[0].dof_start
    @property
    def dof_end(self):
        return self._n_dofs + self.dof_start

    @property
    def dofs_motion_ang(self):
        return self._dofs_motion_ang

    @property
    def dofs_motion_vel(self):
        return self._dofs_motion_vel

    @property
    def dofs_limit(self):
        return self._dofs_limit

    @property
    def dofs_invweight(self):
        return self._dofs_invweight

    @property
    def dofs_stiffness(self):
        return self._dofs_stiffness

    @property
    def dofs_damping(self):
        return self._dofs_damping

    @property
    def dofs_armature(self):
        return self._dofs_armature

    @property
    def dofs_kp(self):
        return self._dofs_kp

    @property
    def dofs_kv(self):
        return self._dofs_kv

    @property
    def dofs_force_range(self):
        return self._dofs_force_range

    @property
    def joint_start(self):
        """
        The start index of the link's joints in the RigidSolver.
        """
        return self._joint_start

    @property
    def joint_end(self):
        """
        The end index of the link's joints in the RigidSolver.
        """
        return self._joint_start + self.n_joints
    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{(self._repr_type())}: {self._id}, name: '{self._name}', idx: {self._idx} (from entity {self._entity.id})"
