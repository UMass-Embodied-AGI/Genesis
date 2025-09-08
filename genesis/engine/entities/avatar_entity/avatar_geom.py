import copy
import torch
import numpy as np
import taichi as ti
import genesis as gs
import trimesh
import genesis.utils.mesh as mesh_utils

import gstaichi as ti

from ..rigid_entity import RigidGeom, RigidVisGeom
@ti.data_oriented
class AvatarGeom(RigidGeom):
    '''AvatarGeom resembles RigidGeom in rigid_solver, but is only used for collision checking.
    '''
    def get_mesh(self):
        '''
        Reconstruct mesh using vverts and vfaces.
        '''
        mesh = trimesh.Trimesh(vertices=self._init_vverts, faces=self._init_vfaces, vertex_normals=self._init_vnormals, process=False)
        mesh.visual = mesh_utils.surface_uvs_to_trimesh_visual(self._surface, self._uvs, len(mesh.vertices))
        return mesh
    
@ti.data_oriented
class AvatarVisGeom(RigidVisGeom):
    def __init__(
        self,
        link,
        idx,
        vvert_start,
        vface_start,
        init_vverts,
        init_vfaces,
        init_vnormals,
        init_matrix,
        vert_joints,
        vert_invbind,
        vert_weights,
        skin_joints,
        nodes,
        type,
        init_pos,
        init_quat,
        color       = None,
        uvs         = None,
        image       = None,
        surface     = None,
    ):
        self._link     = link
        self._entity   = link.entity
        self._material = link.entity.material
        self._solver   = link.entity.solver

        self._uid   = gs.UID()
        self._idx  = idx
        self._type = type
        
        self._vvert_start = vvert_start
        self._vface_start = vface_start

        self._init_pos  = init_pos
        self._init_quat = init_quat

        # enforce contiguous just in case, as trimesh sometimes returns arrays with unpredicatable behavior...
        self._init_vverts   = np.ascontiguousarray(np.array(init_vverts, dtype=gs.np_float))
        self._init_vfaces   = np.ascontiguousarray(np.array(init_vfaces, dtype=gs.np_int))
        self._init_vnormals = np.ascontiguousarray(np.array(init_vnormals, dtype=gs.np_float))

        # skinned vert info
        if vert_joints is not None:
            self._init_matrix  = np.ascontiguousarray(np.array(init_matrix, dtype=gs.np_float))
            self._vert_joints  = np.ascontiguousarray(np.array(vert_joints, dtype=gs.np_int))
            self._vert_invbind = np.ascontiguousarray(np.array(vert_invbind, dtype=gs.np_float))
            self._vert_weights = np.ascontiguousarray(np.array(vert_weights, dtype=gs.np_float))
            self._skin_joints  = np.ascontiguousarray(np.array(skin_joints, dtype=gs.np_int))
            self._nodes = nodes

            self._init_matrix_inv = np.asarray(np.matrix(self._init_matrix).I)
            self._vverts = self._init_vverts
            self._points_homo = (np.append(
                self._init_vverts, np.ones((self._init_vverts.shape[0], 1)), axis=-1
            )@ self._init_matrix_inv)[:,None]

            self._vert_joints_cuda = torch.from_numpy(self._vert_joints).to(torch.int32).to(gs.device)
            self._init_matrix_cuda = torch.from_numpy(self._init_matrix).to(torch.float64).to(gs.device)
            self._vert_invbind_cuda = torch.from_numpy(self._vert_invbind).to(torch.float64).to(gs.device)
            self._vert_weights_cuda = torch.from_numpy(self._vert_weights).to(torch.float64).to(gs.device)
            self._points_homo_cuda = torch.from_numpy(self._points_homo).to(torch.float64).to(gs.device)
        else:
            self._vert_joints  = None
            self._vert_invbind = None
            self._vert_weights = None
            self._skin_joints  = None
            self._nodes = None
            self._vverts = self._init_vverts

        self._init_nodes = copy.deepcopy(self._nodes)
        self.global_transforms = None

        '''
        Surface's own attributes has a higher priority than intrinsic attributes of the parsed assets.
        If surface doesn't come with its own attributes, we update using assets' attributes.
        '''
        if surface is None:
            self._surface = gs.surfaces.Default()
        else:
            self._surface = surface.copy()


        def create_texture(image, factor, encoding):
            if image is not None:
                return gs.textures.ImageTexture(image_array=image, image_color=factor, encoding=encoding) 
            elif factor is not None:
                return gs.textures.ColorTexture(color=factor)
            else:
                return None

        diffuse_texture = create_texture(image[..., :3].copy(order='C') if image is not None else image, None, 'srgb')
        self._surface.update_texture(diffuse_texture)
        texture = self._surface.get_texture()

        self._uvs = None
        if isinstance(texture, gs.textures.ColorTexture):
            self._uvs = None

        elif isinstance(texture, gs.textures.ImageTexture):
            if uvs is None:
                gs.raise_exception('Asset does not have original texture, thus missing uv info (or failed to load). Custom texture map is not allowed.')
            self._uvs = np.ascontiguousarray(uvs)

    def get_trimesh(self):
        '''
        Reconstruct mesh using vverts and vfaces.
        '''
        mesh = trimesh.Trimesh(vertices=self._init_vverts, faces=self._init_vfaces, vertex_normals=self._init_vnormals, process=False)
        mesh.visual = mesh_utils.surface_uvs_to_trimesh_visual(self._surface, self._uvs, len(mesh.vertices))
        return mesh

    @property
    def vverts(self):
        return self._vverts

