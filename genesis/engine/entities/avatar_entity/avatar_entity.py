import torch
import copy
import numpy as np
import taichi as ti
import genesis as gs
from genesis.utils import geom as gu
from genesis.utils import mesh as mu

from scipy.spatial.transform import Rotation

from ..rigid_entity import RigidEntity
from .avatar_joint import AvatarJoint
from .avatar_link import AvatarLink


def transform_quat_by_quat(u, v):
    return np.array([
        u[3] * v[0] + u[0] * v[3] + u[1] * v[2] - u[2] * v[1],
        u[3] * v[1] - u[0] * v[2] + u[1] * v[3] + u[2] * v[0],
        u[3] * v[2] + u[0] * v[1] - u[1] * v[0] + u[2] * v[3],
        u[3] * v[3] - u[0] * v[0] - u[1] * v[1] - u[2] * v[2],
    ])

def parse_matrix(node):
    if node.matrix is not None:
        matrix = np.array(node.matrix, dtype=float).reshape((4, 4))
        assert 0, "why use matrix?"
    else:
        matrix = np.identity(4)
        if node.translation is not None:
            matrix[3, :3] = node.translation
        if node.rotation is not None:
            rotation_matrix = np.identity(4)
            rotation_matrix[:3, :3] = Rotation.from_quat(node.rotation).as_matrix().T
            matrix = rotation_matrix @ matrix
        if node.scale is not None:
            scale = np.array(node.scale, dtype=float)
            scale_matrix = np.diag(np.append(scale, 1))
            matrix = scale_matrix @ matrix
    return matrix

def forward_kinematics(nodes, node_index, root_matrix=np.identity(4), root_id = None, matrix_list=list()):
    node = nodes[node_index]
    if root_id is None:
        matrix = parse_matrix(node) @ root_matrix
    else:
        matrix = root_matrix

    if node_index == root_id:
        root_id = None
    
    matrix_list.append([node_index, matrix])

    for sub_node_index in node.children:
        nodes[sub_node_index].parent = node_index
        forward_kinematics(nodes, sub_node_index, matrix, root_id, matrix_list=matrix_list)

MAPPING = [0, 1, 2, 3, 4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

@ti.data_oriented
class AvatarEntity(RigidEntity):

    def calculate_joint_T(self, base_translation, global_mat, global_mat_inv, skin_base_rot, node_trans):   
        skin_root_id = self._skin_joints[0]
        root = self._nodes[1][skin_root_id]
        if len(self._skin_joints) > 65:
            root.translation = \
                self._init_nodes[1][skin_root_id].translation + base_translation
        else:
            root.translation = \
                self._init_nodes[1][skin_root_id].translation + base_translation * 100
        
        rotation = np.matmul(np.matmul(global_mat_inv[0], gu.quat_to_R(skin_base_rot)), global_mat[0])
        rotation = gu.R_to_quat(rotation)[[1,2,3,0]]
        root.rotation = \
            transform_quat_by_quat(
                self._init_nodes[1][skin_root_id].rotation if self._init_nodes[1][skin_root_id].rotation is not None else [0,0,0,1],
                rotation
            )
        result = node_trans @ parse_matrix(root) @ parse_matrix(self._nodes[1][self._nodes[0]])
        return result

    def calculate_real_pos(self, rotation, base_translation, root_id = None):
        if len(self._skin_joints) > 65:
            self._nodes[1][self._skin_joints[0]].translation = \
                self._init_nodes[1][self._skin_joints[0]].translation + base_translation
            for i,r in enumerate(rotation[:65]):
                joint_id = self._skin_joints[MAPPING[i]]
                self._nodes[1][joint_id].rotation = \
                    transform_quat_by_quat(
                        self._init_nodes[1][joint_id].rotation if self._init_nodes[1][joint_id].rotation is not None else [0,0,0,1],
                        r
                    )
        else:
            self._nodes[1][self._skin_joints[0]].translation = \
                self._init_nodes[1][self._skin_joints[0]].translation + base_translation * 100
            for i,j in enumerate(self._skin_joints[:]):
                self._nodes[1][j].rotation = \
                    transform_quat_by_quat(
                        self._init_nodes[1][j].rotation if self._init_nodes[1][j].rotation is not None else [0,0,0,1],
                        rotation[i]
                    )
        matrix_list = list()
        forward_kinematics(self._nodes[1], self._nodes[0], root_id=root_id, matrix_list=matrix_list)
        self.transforms = np.asarray([it[1] for it in sorted(matrix_list, key=lambda x:x[0])])
        return self.transforms
    
    def update_mesh(self, base_translation, global_mat, global_mat_inv, skin_base_rot, node_trans):
        self.global_transforms = self.calculate_joint_T(base_translation, global_mat, global_mat_inv, skin_base_rot, node_trans)
        skin_mat = self._vert_invbind_cuda @ torch.from_numpy(self.global_transforms[self._skin_joints]).to(torch.float64).to(gs.device)
        skin_mat = torch.einsum(
            "ij,ijmn->imn", 
            self._vert_weights_cuda,
            skin_mat[self._vert_joints_cuda],
        )
        result = (self._points_homo_cuda @ skin_mat @ self._init_matrix_cuda)[:,0,:3].cpu().numpy()
        prev = 0
        for vgeom in self.links[0]._vgeoms:
            nxt = prev + vgeom._points_homo_cuda.shape[0]
            vgeom._vverts = result[prev : nxt]
            prev = nxt
    
    def fabrik(self, node_index, root_node_index, target_pos, max_iterations=10, tolerance=1e-3):
        chain = []
        current_index = node_index
        nodes = self._nodes[1]
        while current_index != root_node_index:
            chain.append(current_index)
            current_index = nodes[current_index].parent
        chain.append(root_node_index)
        chain.reverse()

        positions = [self.transforms[idx][-1,:3].copy() for idx in chain]
        bone_lengths = [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(chain) - 1)]
        
        root_pos = positions[0].copy()
        for _ in range(max_iterations):
            positions[-1] = target_pos.copy() 
            for i in range(len(positions) - 2, -1, -1):
                direction = (positions[i] - positions[i + 1]) / np.linalg.norm(positions[i] - positions[i + 1])
                positions[i] = positions[i + 1] + bone_lengths[i] * direction
            
            positions[0] = root_pos  
            for i in range(1, len(positions)):
                direction = (positions[i] - positions[i - 1]) / np.linalg.norm(positions[i] - positions[i - 1])
                positions[i] = positions[i - 1] + bone_lengths[i - 1] * direction
            
            if np.linalg.norm(positions[-1] - target_pos) < tolerance:
                break

        def get_rotation_between_vectors(v1, v2):
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            cos_theta = np.dot(v1, v2)
            angle = np.arccos(cos_theta)
            axis = np.cross(v1, v2)
            axis = axis / np.linalg.norm(axis)
            rotation = Rotation.from_rotvec(angle * axis)
            return rotation.as_quat()
        
        matrix = self.transforms[nodes[chain[0]].parent]
        for i in range(1, len(chain)):
            pos = np.linalg.inv(matrix.T) @ np.append(positions[i], 1.0)
            vec = pos[:3] - nodes[chain[i-1]].translation
            nodes[chain[i-1]].rotation = get_rotation_between_vectors(nodes[chain[i]].translation, vec)
            matrix = parse_matrix(nodes[chain[i-1]]) @ matrix

        matrix_list = list()
        forward_kinematics(self._nodes[1], self._nodes[0], root_id=self._skin_joints[0], matrix_list=matrix_list)
        node_trans = np.asarray([it[1] for it in sorted(matrix_list, key=lambda x:x[0])])
        return node_trans

    def ik_solve(
        self,
        hand_id,
        root_id,
        target_pos,
        max_iterations=10,
        tolerance=1e-3
    ): 
        hand_idx = self.node_findup[hand_id]
        root_idx = self.node_findup[root_id]
        target_pos[2] -= self._morph.pos[2]
        target_pos[0] *= -1
        target_pos[:2] = target_pos[:2][::-1]
        return self.fabrik(hand_idx, root_idx, target_pos, max_iterations, tolerance)
        

    def get_node_translation(self, name):
        return self.global_transforms[self.node_findup[name]]
    
    def get_global_translation(self, name):
        '''
        Get the global translation and rotation matrix of a specified joint in the world coordinate system.

        Parameters:
        name (str): The name of the joint, such as "LeftHand", "RightEye".

        Returns:
        tuple: A tuple containing the translation vector and rotation matrix of the joint in the world coordinate system.
            - The translation is a 3D vector (x, y, z).
            - The rotation matrix is a 3x3 matrix representing the joint's rotation.
        '''
        matrix4x4T = self.get_node_translation(name).copy()
        
        pos = matrix4x4T[-1, :3]
        pos[:2] = pos[:2][::-1]
        pos[0] *= -1
        pos[2] += self._morph.pos[2]
        rot = matrix4x4T[:3, :3]
        rot = (rot / np.linalg.norm(rot, axis=1, keepdims=True)).T

        return pos, rot

    def _load_mesh(self, morph, surface):
        if morph.fixed:
            joint_type = gs.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_q = np.zeros(0)

        else:
            joint_type = gs.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_q = np.concatenate([morph.pos, morph.quat], dtype=gs.np_float)

         # mesh has 1 single link
        link = self.add_link(
            name            = 'baselink',
            n_qs             = n_qs,
            n_dofs          = n_dofs,
            pos             = np.array(morph.pos, dtype=gs.np_float),
            quat            = np.array(morph.quat, dtype=gs.np_float),
            inertial_pos    = gu.zero_pos(),
            inertial_quat   = gu.identity_quat(),
            inertial_i      = None,
            inertial_mass   = None,
            parent_idx      = -1,
            invweight       = None,
            joint_type      = joint_type,
            joint_pos       = gu.zero_pos(),
            joint_quat      = gu.identity_quat(),
            dofs_motion_ang = np.eye(6, 3, -3),
            dofs_motion_vel = np.eye(6, 3),
            dofs_limit      = np.tile([-np.inf, np.inf], (6, 1)),
            dofs_invweight  = np.ones(6),
            dofs_stiffness  = np.zeros(6),
            dofs_damping    = np.zeros(6),
            dofs_armature   = np.zeros(6),
            dofs_kp         = np.zeros((6,), dtype=gs.np_float),
            dofs_kv         = np.zeros((6,), dtype=gs.np_float),
            dofs_force_range= np.tile([-np.inf, np.inf], (6, 1)),
            init_q          = init_q,
            joint_start    = self._joint_start + self.n_joints,
            n_joints        = 1,
            root_idx       = None,
        )

        self._add_joint(
            name             = 'joint_baselink',
            n_qs             = n_qs,
            n_dofs           = n_dofs,
            type             = joint_type,
            link_idx         = link.idx,
            pos              = gu.zero_pos(),
            quat             = gu.identity_quat(),
            dofs_motion_ang = np.eye(6, 3, -3),
            dofs_motion_vel = np.eye(6, 3),
            dofs_limit      = np.tile([-np.inf, np.inf], (6, 1)),
            dofs_invweight  = np.ones(6),
            dofs_stiffness  = np.zeros(6),
            dofs_damping    = np.zeros(6),
            dofs_armature   = np.zeros(6),
            dofs_kp         = np.zeros((6,), dtype=gs.np_float),
            dofs_kv         = np.zeros((6,), dtype=gs.np_float),
            dofs_force_range= np.tile([-np.inf, np.inf], (6, 1)),
            init_q           = init_q,
        )

        vm_infos, m_infos = mu.parse_visual_and_col_mesh_avatar(morph, surface)
        m_infos  = m_infos[0:300]  if len(m_infos)  > 100 else m_infos
        vm_infos = vm_infos[0:300] if len(vm_infos) > 100 else vm_infos

        if 'nodes' in vm_infos[0]:
            self._init_nodes = copy.deepcopy(vm_infos[0]['nodes'])
            self._nodes = copy.deepcopy(self._init_nodes)
            self.node_findup = {n.name.split(":")[-1]:i for i,n in enumerate(self._nodes[1])}

        if morph.visualization:
            
            self._vert_joints_cuda = []
            self._vert_weights_cuda = []
            self._points_homo_cuda = []
            for vm_info in vm_infos:
                link.add_vgeom(
                    init_vverts   = vm_info['vverts'],
                    init_vfaces   = vm_info['vfaces'],
                    init_vnormals = vm_info['vnormals'],
                    init_pos      = gu.zero_pos(),
                    init_quat     = gu.identity_quat(),
                    type          = gs.GEOM_TYPE.MESH,
                    init_matrix   = vm_info['init_matrix'] if 'init_matrix' in vm_info else None,
                    vert_joints   = vm_info['vert_joints'] if 'vert_joints' in vm_info else None,
                    vert_invbind  = vm_info['vert_invbind'] if 'vert_invbind' in vm_info else None,
                    vert_weights  = vm_info['vert_weights'] if 'vert_weights' in vm_info else None,
                    skin_joints   = vm_info['skin_joints'] if 'skin_joints' in vm_info else None,
                    nodes         = vm_info['nodes'] if 'nodes' in vm_info else None,
                    color         = vm_info['color'],
                    uvs           = vm_info['uvs'],
                    image         = vm_info['image'],
                    surface       = surface,
                )
                vgeom = link._vgeoms[-1]
                self._vert_joints_cuda.append(vgeom._vert_joints_cuda)
                self._vert_weights_cuda.append(vgeom._vert_weights_cuda)
                self._points_homo_cuda.append(vgeom._points_homo_cuda)
            self._vert_joints_cuda = torch.cat(self._vert_joints_cuda, dim=0)
            self._vert_weights_cuda = torch.cat(self._vert_weights_cuda, dim=0)
            self._points_homo_cuda = torch.cat(self._points_homo_cuda, dim=0)
            self._vert_invbind_cuda = link._vgeoms[-1]._vert_invbind_cuda
            self._init_matrix_cuda = link._vgeoms[-1]._init_matrix_cuda
            self._skin_joints = link._vgeoms[-1]._skin_joints

        if morph.collision:
            for m_info in m_infos:
                link.add_geom(
                    init_verts   = m_info['verts'],
                    init_faces   = m_info['faces'],
                    init_edges   = m_info['edges'],
                    init_normals = m_info['normals'],
                    init_pos     = gu.zero_pos(),
                    init_quat    = gu.identity_quat(),
                    type         = gs.GEOM_TYPE.MESH,
                    needs_coup   = self.material.needs_coup,
                    is_convex    = m_info['is_convex'],
                )

    def add_link(self, name, n_qs, n_dofs, pos, quat, inertial_pos, inertial_quat, inertial_i, inertial_mass, parent_idx, invweight, joint_type, joint_pos, joint_quat, dofs_motion_ang, dofs_motion_vel, dofs_limit, dofs_invweight, dofs_stiffness, dofs_damping, dofs_armature, dofs_kp, dofs_kv, dofs_force_range, init_q,
            joint_start,
            n_joints,
            root_idx ):
        link = AvatarLink(
            entity           = self,
            name             = name,
            idx              = self.n_links + self._link_start,
            idx_offset_q     = self.n_qs + self._q_start,
            idx_offset_dof   = self.n_dofs + self._dof_start,
            idx_offset_geom  = self.n_geoms + self._geom_start,
            idx_offset_cell  = self.n_cells + self._cell_start,
            idx_offset_vert  = self.n_verts + self._vert_start,
            idx_offset_face  = self.n_faces + self._face_start,
            idx_offset_edge  = self.n_edges + self._edge_start,
            idx_offset_vgeom = self.n_vgeoms + self._vgeom_start,
            vvert_start = self.n_vverts + self._vvert_start,
            vface_start = self.n_vfaces + self._vface_start,
            n_qs              = n_qs,
            n_dofs           = n_dofs,
            pos              = pos,
            quat             = quat,
            inertial_pos     = inertial_pos,
            inertial_quat    = inertial_quat,
            inertial_i       = inertial_i,
            inertial_mass    = inertial_mass,
            parent_idx       = parent_idx,
            invweight        = invweight,
            joint_type       = joint_type,
            joint_pos        = joint_pos,
            joint_quat       = joint_quat,
            dofs_motion_ang  = dofs_motion_ang,
            dofs_motion_vel  = dofs_motion_vel,
            dofs_limit       = dofs_limit,
            dofs_invweight   = dofs_invweight,
            dofs_stiffness   = dofs_stiffness,
            dofs_damping     = dofs_damping,
            dofs_armature    = dofs_armature,
            dofs_kp          = dofs_kp,
            dofs_kv          = dofs_kv,
            dofs_force_range = dofs_force_range,
            init_q           = init_q,
            joint_start      = joint_start,
            n_joints         = n_joints,
            root_idx         = root_idx,
        )
        self._links.append(link)
        return link

    def _add_joint(
        self,
        name: str,
        n_qs: int,
        n_dofs: int,
        type: str,
        link_idx: int,
        pos,
        quat,
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
    ) -> AvatarJoint:
        """
        Add a new joint (AvatarJoint) to the entity.

        Parameters
        ----------
        name : str
            Name of the joint.
        n_qs : int
            Number of configuration variables (generalized coordinates).
        n_dofs : int
            Number of degrees of freedom for the joint.
        type : str
            Type of the joint (e.g., "revolute", "prismatic").
        pos : array-like
            Position of the joint frame.
        quat : array-like
            Orientation (quaternion) of the joint frame.
        dofs_motion_ang : array-like
            Angular motions allowed for each DOF.
        dofs_motion_vel : array-like
            Velocity directions for each DOF.
        dofs_limit : array-like
            Limits for each DOF (e.g., min/max).
        dofs_invweight : array-like
            Inverse weight for each DOF.
        dofs_stiffness : array-like
            Stiffness values for each DOF.
        dofs_damping : array-like
            Damping values for each DOF.
        dofs_armature : array-like
            Armature inertia values.
        dofs_kp : array-like
            Proportional gains for control.
        dofs_kv : array-like
            Derivative gains for control.
        dofs_force_range : array-like
            Allowed force/torque range for each DOF.
        init_q : array-like
            Initial configuration (position/orientation) for the joint.

        Returns
        -------
        joint : AvatarJoint
            The created AvatarJoint instance.
        """
        joint = AvatarJoint(
            entity           = self,
            name             = name,
            idx              = self.n_joints + self._joint_start,
            link_idx         = link_idx,
            q_start          = self.n_qs + self._q_start,
            dof_start        = self.n_dofs + self._dof_start,
            n_qs             = n_qs,
            n_dofs           = n_dofs,
            type             = type,
            pos              = pos,
            quat             = quat,
            dofs_motion_ang  = dofs_motion_ang,
            dofs_motion_vel  = dofs_motion_vel,
            dofs_limit       = dofs_limit,
            dofs_invweight   = dofs_invweight,
            dofs_stiffness   = dofs_stiffness,
            dofs_damping     = dofs_damping,
            dofs_armature    = dofs_armature,
            dofs_kp          = dofs_kp,
            dofs_kv          = dofs_kv,
            dofs_force_range = dofs_force_range,
            init_qpos        = init_q,
            sol_params = gu.default_solver_params()
        )
        self._joints.append([joint])
        return joint

    def init_jac_and_IK(self) -> None:
        # TODO: Avatar should also support IK
        pass
