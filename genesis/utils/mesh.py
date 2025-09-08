import hashlib
import json
import math
import os
import pickle as pkl
from functools import lru_cache
from pathlib import Path
from urllib import request
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image
import OpenEXR
import Imath

import coacd
import igl

import genesis as gs

from tqdm import tqdm

from . import geom as gu
from .misc import (
    get_assets_dir,
    get_cvx_cache_dir,
    get_exr_cache_dir,
    get_gsd_cache_dir,
    get_ptc_cache_dir,
    get_remesh_cache_dir,
    get_src_dir,
    get_usd_cache_dir,
    get_tet_cache_dir,
)

MESH_REPAIR_ERROR_THRESHOLD = 0.01

class MeshInfo:
    def __init__(self, surface=None):
        self.surface = surface
        self.verts = list()
        self.faces = list()
        self.normals = list()
        self.uvs = list()
        self.n_points = 0
        self.n_members = 0
        self.uvs_exist = False

    def append(self, verts, faces, normals, uvs):
        faces += self.n_points
        self.verts.append(verts)
        self.faces.append(faces)
        self.normals.append(normals)
        self.uvs.append(uvs)
        self.n_points += verts.shape[0]
        self.n_members += 1
        if uvs is not None:
            self.uvs_exist = True

    def export_mesh(self, scale):
        if self.uvs:
            for i, (uvs, verts) in enumerate(zip(self.uvs, self.verts)):
                if uvs is None:
                    self.uvs[i] = np.zeros((len(verts), 2), dtype=gs.np_float)
            uvs = np.concatenate(self.uvs, axis=0)
        else:
            uvs = None

        verts = np.concatenate(self.verts, axis=0)
        faces = np.concatenate(self.faces, axis=0)
        normals = np.concatenate(self.normals, axis=0)

        mesh = gs.Mesh.from_attrs(
            verts=verts,
            faces=faces,
            normals=normals,
            surface=self.surface,
            uvs=uvs,
            scale=scale,
        )
        return mesh

    def set_property(self, surface=None, metadata=None):
        self.surface = surface
        self.metadata = metadata

class MeshInfoGroup:
    def __init__(self):
        self.infos = dict()

    def get(self, name):
        first_created = False
        mesh_info = self.infos.get(name)
        if mesh_info is None:
            mesh_info = self.infos.setdefault(name, MeshInfo())
            first_created = True
        return mesh_info, first_created

    def append(self, name, verts, faces, normals, uvs, surface):
        if name not in self.infos:
            self.infos[name] = MeshInfo(surface)
        self.infos[name].append(verts, faces, normals, uvs)

    def export_meshes(self, scale):
        return [mesh_info.export_mesh(scale) for mesh_info in self.infos.values()]


def get_asset_path(file):
    return os.path.join(get_src_dir(), "assets", file)


def get_gsd_path(verts, faces, sdf_cell_size, sdf_min_res, sdf_max_res):
    hashkey = get_hashkey(
        verts.tobytes(),
        faces.tobytes(),
        str(sdf_cell_size).encode(),
        str(sdf_min_res).encode(),
        str(sdf_max_res).encode(),
    )
    return os.path.join(get_gsd_cache_dir(), f"{hashkey}.gsd")


def get_cvx_path(verts, faces, coacd_options):
    hashkey = get_hashkey(verts.tobytes(), faces.tobytes(), str(coacd_options.__dict__).encode())
    return os.path.join(get_cvx_cache_dir(), f"{hashkey}.cvx")


def get_ptc_path(verts, faces, p_size, sampler):
    hashkey = get_hashkey(verts.tobytes(), faces.tobytes(), str(p_size).encode(), sampler.encode())
    return os.path.join(get_ptc_cache_dir(), f"{hashkey}.ptc")


def get_tet_path(verts, faces, tet_cfg):
    hashkey = get_hashkey(verts.tobytes(), faces.tobytes(), str(tet_cfg).encode())
    return os.path.join(get_tet_cache_dir(), f"{hashkey}.tet")


def get_remesh_path(verts, faces, edge_len_abs, edge_len_ratio, fix):
    hashkey = get_hashkey(
        verts.tobytes(), faces.tobytes(), str(edge_len_abs).encode(), str(edge_len_ratio).encode(), str(fix).encode()
    )
    return os.path.join(get_remesh_cache_dir(), f"{hashkey}.rm")


def get_exr_path(file_path):
    hashkey = get_file_hashkey(file_path)
    return os.path.join(get_exr_cache_dir(), f"{hashkey}.exr")


def get_usd_zip_path(file_path):
    hashkey = get_file_hashkey(file_path)
    return os.path.join(get_usd_cache_dir(), "zip", hashkey)


def get_usd_bake_path(file_path):
    hashkey = get_file_hashkey(file_path)
    return os.path.join(get_usd_cache_dir(), "bake", hashkey)


def get_file_hashkey(file):
    file_obj = Path(file)
    return get_hashkey(file_obj.resolve().as_posix().encode(), str(file_obj.stat().st_size).encode())


def get_hashkey(*args):
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(arg)
    hasher.update(gs.__version__.encode())
    return hasher.hexdigest()


def load_mesh(file):
    if isinstance(file, str):
        return trimesh.load(file, force="mesh", skip_texture=True)
    else:
        return file


def normalize_mesh(mesh):
    """
    Normalize mesh to [-0.5, 0.5].
    """
    scale = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2.0

    normalized_mesh = mesh.copy()
    normalized_mesh.vertices -= center
    normalized_mesh.vertices /= scale
    return normalized_mesh


def scale_mesh(mesh, scale):
    scale = np.array(scale)
    return trimesh.Trimesh(
        vertices=mesh.vertices * scale,
        faces=mesh.faces,
    )


def cleanup_mesh(mesh):
    """
    Retain only mesh's vertices, faces, and normals.
    """
    return trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_normals=mesh.vertex_normals,
        face_normals=mesh.face_normals,
    )


def compute_sdf_data(mesh, res):
    """
    Convert mesh to sdf voxels and a transformation matrix from mesh frame to voxel frame.
    """
    voxels_radius = 0.6
    x = np.linspace(-voxels_radius, voxels_radius, res)
    y = np.linspace(-voxels_radius, voxels_radius, res)
    z = np.linspace(-voxels_radius, voxels_radius, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

    voxels, *_ = igl.signed_distance(query_points, mesh.vertices, mesh.faces)
    voxels = voxels.reshape((res, res, res)).astype(gs.np_float, copy=False)

    T_mesh_to_sdf = np.eye(4, dtype=gs.np_float)
    T_mesh_to_sdf[:3, :3] *= (res - 1) / (voxels_radius * 2)
    T_mesh_to_sdf[:3, 3] = (res - 1) / 2

    sdf_data = {
        "voxels": voxels,
        "T_mesh_to_sdf": T_mesh_to_sdf,
    }
    return sdf_data


def voxelize_mesh(mesh, res):
    return mesh.voxelized(pitch=1.0 / res).fill()


def surface_uvs_to_trimesh_visual(surface, uvs=None, n_verts=None):
    texture = surface.get_rgba()
    texture.check_dim(3)
    # texture = surface.get_texture()

    if isinstance(texture, gs.textures.ImageTexture):
        if uvs is not None:
            uvs = uvs.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
            assert texture.image_array.dtype == np.uint8
            visual = trimesh.visual.TextureVisuals(
                uv=uvs,
                material=trimesh.visual.material.SimpleMaterial(
                    texture.image_array, diffuse=(1.0, 1.0, 1.0, 1.0)
                ),
            )
        else:
            # fall back to color texture
            visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(texture.mean_color(), [n_verts, 1]))

    elif isinstance(texture, gs.textures.ColorTexture):
        if n_verts is None:
            gs.raise_exception("n_verts is required for color texture.")
        visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(np.array(texture.color), [n_verts, 1]))

    else:
        gs.raise_exception("Cannot get texture when generating trimesh visual.")

    return visual


def convex_decompose(mesh, coacd_options):
    # compute file name via hashing for caching
    cvx_path = get_cvx_path(mesh.vertices, mesh.faces, coacd_options)

    # loading pre-computed cache if available
    is_cached_loaded = False
    if os.path.exists(cvx_path):
        gs.logger.debug("Convex decomposition file (.cvx) found in cache.")
        try:
            with open(cvx_path, "rb") as file:
                mesh_parts = pkl.load(file)
            is_cached_loaded = True
        except (EOFError, ModuleNotFoundError, pkl.UnpicklingError):
            gs.logger.info("Ignoring corrupted cache.")

    if not is_cached_loaded:
        with gs.logger.timer("Running convex decomposition."):
            mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            args = coacd_options
            result = coacd.run_coacd(
                mesh,
                threshold=args.threshold,
                max_convex_hull=args.max_convex_hull,
                preprocess_mode=args.preprocess_mode,
                preprocess_resolution=args.preprocess_resolution,
                resolution=args.resolution,
                mcts_nodes=args.mcts_nodes,
                mcts_iterations=args.mcts_iterations,
                mcts_max_depth=args.mcts_max_depth,
                pca=args.pca,
                merge=args.merge,
                decimate=args.decimate,
                max_ch_vertex=args.max_ch_vertex,
                extrude=args.extrude,
                extrude_margin=args.extrude_margin,
                apx_mode=args.apx_mode,
                seed=args.seed,
            )
            mesh_parts = []
            for vs, fs in result:
                mesh_parts.append(trimesh.Trimesh(vs, fs))

            os.makedirs(os.path.dirname(cvx_path), exist_ok=True)
            with open(cvx_path, "wb") as file:
                pkl.dump(mesh_parts, file)

    return mesh_parts


def postprocess_collision_geoms(
    g_infos, decimate, decimate_face_num, decimate_aggressiveness, convexify, decompose_error_threshold, coacd_options
):
    # Early return if there is no geometry to process
    if not g_infos:
        return []

    # Try the repair meshes that seems to be "broken" but not beyond repair.
    # Note that this procedure is only applied if the estimated volume is significantly different before and after
    # repair, to avoid altering the original mesh without actual benefit. Moreover, only duplicate faces are removed,
    # which is less aggressive than `Trimesh.process(validate=True)`.
    for g_info in g_infos:
        mesh = g_info["mesh"]
        tmesh = mesh.trimesh
        if g_info["type"] != gs.GEOM_TYPE.MESH:
            continue
        if tmesh.is_winding_consistent and not tmesh.is_watertight:
            tmesh_repaired = tmesh.copy()
            tmesh_repaired.update_faces(tmesh_repaired.unique_faces())
            if abs(tmesh_repaired.volume) < gs.EPS:
                continue
            if abs(abs(tmesh.volume / tmesh_repaired.volume) - 1.0) > MESH_REPAIR_ERROR_THRESHOLD:
                gs.logger.info(
                    "Collision mesh is not watertight and has ill-defined volume. It will be repaired by removing "
                    "duplicate faces."
                )
                tmesh.update_faces(tmesh.unique_faces())
                tmesh._cache.clear()
                tmesh.visual._cache.clear()

    # Check if all the geometries can be convexify without decomposition
    must_decompose = False
    if convexify:
        for g_info in g_infos:
            mesh = g_info["mesh"]
            tmesh = mesh.trimesh

            # Skip geometries that do not corresponds to mesh or have no enclosed volume
            if g_info["type"] != gs.GEOM_TYPE.MESH:
                continue
            cmesh = trimesh.convex.convex_hull(tmesh)
            if abs(cmesh.volume) < gs.EPS:
                continue

            # Fix mesh temporarily to make volume computation more reliable
            if not tmesh.is_winding_consistent:
                tmesh = tmesh.copy()
                tmesh.process(validate=True)

            # Compute volume approximation error between true geometry and its convex hull conservatively
            if not tmesh.is_winding_consistent:
                volume_err = float("inf")
                must_decompose = not math.isinf(decompose_error_threshold)
            elif tmesh.volume > gs.EPS:
                volume_err = cmesh.volume / abs(tmesh.volume) - 1.0
                if volume_err > decompose_error_threshold:
                    must_decompose = True

    # Check whether merging the geometries is possible, i.e.
    # * They are all meshes
    # * They belong to the same collision group (same contype and conaffinity)
    # * Their physical properties are the same (friction coef and contact solver parameters)
    if must_decompose and len(g_infos) > 1:
        is_merged = all(g_info["type"] == gs.GEOM_TYPE.MESH for g_info in g_infos)
        for name in ("contype", "conaffinity", "friction", "sol_params"):
            if not is_merged:
                break
            values = np.stack([g_info.get(name, float("nan")) for g_info in g_infos], axis=0)
            diffs = np.diff(values, axis=0)
            if not (np.isnan(diffs).all(axis=0) | (np.abs(diffs) < gs.EPS).all(axis=0)).all():
                is_merged = False

        # Must apply geometry transform before merge concatenation
        if is_merged:
            tmeshes = []
            for g_info in g_infos:
                mesh = g_info["mesh"]
                tmesh = mesh.trimesh.copy()
                pos = g_info.get("pos", gu.zero_pos())
                quat = g_info.get("quat", gu.identity_quat())
                tmesh.apply_transform(gs.utils.geom.trans_quat_to_T(pos, quat))
                tmeshes.append(tmesh)
            tmesh = trimesh.util.concatenate(tmeshes)
            mesh = gs.Mesh.from_trimesh(mesh=tmesh, surface=gs.surfaces.Collision(), metadata={"merged": True})
            g_infos = [{**g_infos[0], **dict(mesh=mesh, pos=gu.zero_pos(), quat=gu.identity_quat())}]

        # Try again to convexify then apply convex decomposition if not possible
        if is_merged:
            return postprocess_collision_geoms(
                g_infos,
                decimate,
                decimate_face_num,
                decimate_aggressiveness,
                convexify,
                decompose_error_threshold,
                coacd_options,
            )

    if must_decompose:
        if math.isinf(volume_err):
            gs.logger.info(
                "Collision mesh has inconsistent winding and 'decompose_error_threshold' != float('inf'). "
                "Falling back to more expensive convex decomposition (see FileMorph options)."
            )
        else:
            gs.logger.info(
                f"Convex hull is not accurate enough for collision detection ({volume_err:.3f}). Falling back to more "
                "expensive convex decomposition (see FileMorph options)."
            )
        _g_infos = []
        for g_info in g_infos:
            mesh = g_info["mesh"]
            tmesh = mesh.trimesh
            if g_info["type"] != gs.GEOM_TYPE.MESH:
                volume_err = 0.0
            if not tmesh.is_winding_consistent:
                volume_err = float("inf")
            elif abs(tmesh.volume) < gs.EPS:
                volume_err = 0.0
            else:
                cmesh = trimesh.convex.convex_hull(tmesh)
                volume_err = cmesh.volume / abs(tmesh.volume) - 1.0
            if volume_err > decompose_error_threshold:  # Note that 'inf' is not larger than 'inf'
                tmeshes = convex_decompose(tmesh, coacd_options)
                meshes = [
                    gs.Mesh.from_trimesh(
                        tmesh, surface=gs.surfaces.Collision(), metadata={**mesh.metadata, "decomposed": True}
                    )
                    for tmesh in tmeshes
                ]
                _g_infos += [{**g_info, **dict(mesh=mesh)} for mesh in meshes]
            else:
                _g_infos.append(g_info)
        g_infos = _g_infos

    # Process of meshes sequentially
    _g_infos = []
    for g_info in g_infos:
        mesh = g_info["mesh"]
        tmesh = mesh.trimesh

        num_faces = len(tmesh.faces)
        if not decimate and num_faces > 5000:
            gs.logger.warning(
                f"At least one of the meshes contain many faces ({num_faces}). Consider setting "
                "'morph.decimate=True' to speed up collision detection and improve numerical stability."
            )
        if decimate and decimate_face_num < 100:
            gs.logger.warning(
                "`decimate_face_num` should be greater than 100 to ensure sufficient geometry details are preserved."
            )

        must_decimate = num_faces > decimate_face_num or tmesh.is_watertight
        if not must_decimate:
            gs.logger.debug(
                "Collision mesh is not watertight. Decimate would be unreliable. Skipping as mesh is already low-poly."
            )

        mesh = gs.Mesh.from_trimesh(
            mesh=tmesh,
            convexify=convexify,
            decimate=decimate and must_decimate,
            decimate_face_num=decimate_face_num,
            decimate_aggressiveness=decimate_aggressiveness,
            surface=gs.surfaces.Collision(),
            metadata=mesh.metadata.copy(),
        )
        _g_infos.append({**g_info, **dict(mesh=mesh)})

    return _g_infos


def parse_mesh_trimesh(path, group_by_material, scale, surface):
    meshes = []
    for _, mesh in trimesh.load(path, force="scene", group_material=group_by_material, process=False).geometry.items():
        meshes.append(gs.Mesh.from_trimesh(mesh=mesh, scale=scale, surface=surface, metadata={"mesh_path": path}))
    return meshes


def trimesh_to_mesh(mesh, scale, surface):
    return gs.Mesh.from_trimesh(mesh=mesh, scale=scale, surface=surface)


def adjust_alpha_cutoff(alpha_cutoff, alpha_mode):
    if alpha_mode == 0:  # OPAQUE
        return 0.0
    if alpha_mode == 1:  # MASK
        return alpha_cutoff
    return None  # BLEND


def PIL_to_array(image):
    return np.array(image)


def tonemapped(image):
    exposure = 0.5
    return (np.clip(np.power(image / 255 * np.power(2, exposure), 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def create_texture(image, factor, encoding):
    if image is not None:
        return gs.textures.ImageTexture(image_array=image, image_color=factor, encoding=encoding)
    elif factor is not None:
        return gs.textures.ColorTexture(color=factor)
    else:
        return None

def opacity_from_texture(color_texture, alpha_cutoff=None):
    opacity_texture = color_texture.check_dim(3)
    if opacity_texture is not None:
        if alpha_cutoff is not None:
            opacity_texture.apply_cutoff(alpha_cutoff)
        if isinstance(opacity_texture, gs.textures.ImageTexture) and opacity_texture.image_array.max() == opacity_texture.image_array.min():
            alpha = opacity_texture.image_array.max() / 255.0
            opacity_texture = create_texture(None, (alpha,), 'linear')
    return opacity_texture

def apply_transform(transform, positions, normals=None):
    transformed_positions = (np.column_stack([positions, np.ones(len(positions))]) @ transform)[:, :3]

    transformed_normals = normals
    if normals is not None:
        rot_mat = transform[:3, :3]
        if np.abs(3.0 - np.trace(rot_mat)) > gs.EPS**2:  # has rotation or scaling
            transformed_normals = normals @ rot_mat
            scale = np.linalg.norm(rot_mat, axis=1, keepdims=True)
            if np.any(np.abs(scale - 1.0) > gs.EPS):  # has scale
                transformed_normals /= np.linalg.norm(transformed_normals, axis=1, keepdims=True)

    return transformed_positions, transformed_normals

def parse_tree(nodes, node_index):
    node = nodes[node_index]
    if node.matrix is not None:
        matrix = np.array(node.matrix, dtype=float).reshape((4, 4))
    else:
        matrix = np.identity(4, dtype=float)
        if node.translation is not None:
            translation = np.array(node.translation, dtype=float)
            translation_matrix = np.identity(4, dtype=float)
            translation_matrix[3, :3] = translation
            matrix = translation_matrix @ matrix
        if node.rotation is not None:
            rotation = np.array(node.rotation, dtype=float)     # xyzw
            rotation_matrix = np.identity(4, dtype=float)
            rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
            rotation_matrix[:3, :3] = trimesh.transformations.quaternion_matrix(rotation)[:3, :3].T
            matrix = rotation_matrix @ matrix
        if node.scale is not None:
            scale = np.array(node.scale, dtype=float)
            scale_matrix = np.diag(np.append(scale, 1))
            matrix = scale_matrix @ matrix
    mesh_list = list()
    if node.mesh is not None:
        mesh_list.append([node.mesh, np.identity(4, dtype=float)])
    for sub_node_index in node.children:
        sub_mesh_list = parse_tree(nodes, sub_node_index)
        mesh_list.extend(sub_mesh_list)
    for i in range(len(mesh_list)):
        mesh_list[i][1] = mesh_list[i][1] @ matrix
    return mesh_list

    def get_bufferview_data(buffer_view):
        buffer = glb.buffers[buffer_view.buffer]
        return glb.get_data_from_buffer_uri(buffer.uri)

    def get_data_from_accessor(accessor_index):
        accessor = glb.accessors[accessor_index]
        buffer_view = glb.bufferViews[accessor.bufferView]
        buffer_data = get_bufferview_data(buffer_view)

        data_type, data_ctype, count = accessor.type, accessor.componentType, accessor.count
        dtype = ctype_to_numpy[data_ctype][1]
        itemsize = np.dtype(dtype).itemsize
        buffer_byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        num_components = type_to_count[data_type][0]

        byte_stride = buffer_view.byteStride if buffer_view.byteStride else num_components * itemsize
        # Extract data considering byteStride
        if byte_stride == num_components * itemsize:
            # Data is tightly packed
            byte_length = count * num_components * itemsize
            data = buffer_data[buffer_byte_offset : buffer_byte_offset + byte_length]
            array = np.frombuffer(data, dtype=dtype)
            if num_components > 1:
                array = array.reshape((count, num_components))
        else:
            # Data is interleaved
            array = np.zeros((count, num_components), dtype=dtype)
            for i in range(count):
                start = buffer_byte_offset + i * byte_stride
                end = start + num_components * itemsize
                data_slice = buffer_data[start:end]
                array[i] = np.frombuffer(data_slice, dtype=dtype, count=num_components)

        return array.reshape([count] + type_to_count[data_type][1])

    glb.convert_images(pygltflib.ImageFormat.DATAURI)

    scene = glb.scenes[glb.scene]
    mesh_list = list()
    for node_index in scene.nodes:
        root_mesh_list = parse_tree(node_index)
        mesh_list.extend(root_mesh_list)

    temp_infos = dict()
    for i in range(len(mesh_list)):
        mesh = glb.meshes[mesh_list[i][0]]
        matrix = mesh_list[i][1]
        for primitive in mesh.primitives:
            if group_by_material:
                group_idx = primitive.material
            else:
                group_idx = i

            uvs0, uvs1 = None, None
            if "KHR_draco_mesh_compression" in primitive.extensions:
                import DracoPy

                KHR_index = primitive.extensions["KHR_draco_mesh_compression"]["bufferView"]
                mesh_buffer_view = glb.bufferViews[KHR_index]
                mesh_data = get_bufferview_data(mesh_buffer_view)
                mesh = DracoPy.decode(
                    mesh_data[mesh_buffer_view.byteOffset : mesh_buffer_view.byteOffset + mesh_buffer_view.byteLength]
                )
                points = mesh.points
                triangles = mesh.faces
                normals = mesh.normals if len(mesh.normals) > 0 else None
                uvs0 = mesh.tex_coord if len(mesh.tex_coord) > 0 else None

            else:
                # "primitive.attributes" records accessor indices in "glb.accessors", like:
                #      Attributes(POSITION=2, NORMAL=1, TANGENT=None, TEXCOORD_0=None, TEXCOORD_1=None,
                #                 COLOR_0=None, JOINTS_0=None, WEIGHTS_0=None)
                # parse vertices

                points = get_data_from_accessor(primitive.attributes.POSITION).astype(float)

                if primitive.indices is None:
                    indices = np.arange(points.shape[0], dtype=np.uint32)
                else:
                    indices = get_data_from_accessor(primitive.indices).astype(np.int32)

                mode = primitive.mode if primitive.mode is not None else 4

                if mode == 4:  # TRIANGLES
                    triangles = indices.reshape(-1, 3)
                elif mode == 5:  # TRIANGLE_STRIP
                    triangles = []
                    for i in range(len(indices) - 2):
                        if i % 2 == 0:
                            triangles.append([indices[i], indices[i + 1], indices[i + 2]])
                        else:
                            triangles.append([indices[i], indices[i + 2], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                elif mode == 6:  # TRIANGLE_FAN
                    triangles = []
                    for i in range(1, len(indices) - 1):
                        triangles.append([indices[0], indices[i], indices[i + 1]])
                    triangles = np.array(triangles, dtype=np.uint32)
                else:
                    gs.logger.warning(f"Primitive mode {mode} not supported.")
                    continue  # Skip unsupported modes

                # parse normals
                if primitive.attributes.NORMAL:
                    normals = get_data_from_accessor(primitive.attributes.NORMAL).astype(float)
                else:
                    normals = None

                # parse uvs
                if primitive.attributes.TEXCOORD_0:
                    uvs0 = get_data_from_accessor(primitive.attributes.TEXCOORD_0).astype(float)
                if primitive.attributes.TEXCOORD_1:
                    uvs1 = get_data_from_accessor(primitive.attributes.TEXCOORD_1).astype(float)

            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals
            points, normals = apply_transform(matrix, points, normals)

            if group_idx not in temp_infos.keys():
                temp_infos[group_idx] = {
                    "mat_index": primitive.material,
                    "points": [points],
                    "triangles": [triangles],
                    "normals": [normals],
                    "uvs0": [uvs0],
                    "uvs1": [uvs1],
                    "n_points": len(points),
                }

            else:
                triangles += temp_infos[group_idx]["n_points"]
                temp_infos[group_idx]["points"].append(points)
                temp_infos[group_idx]["triangles"].append(triangles)
                temp_infos[group_idx]["normals"].append(normals)
                temp_infos[group_idx]["uvs0"].append(uvs0)
                temp_infos[group_idx]["uvs1"].append(uvs1)
                temp_infos[group_idx]["n_points"] += len(points)

    meshes = list()
    for group_idx in temp_infos.keys():
        # parse images
        color_texture = None
        opacity_texture = None
        roughness_texture = None
        metallic_texture = None
        normal_texture = None
        emissive_texture = None

        alpha_cutoff = None
        double_sided = None
        ior = None
        uvs_used = 0

        if temp_infos[group_idx]["mat_index"] is not None:
            material = glb.materials[temp_infos[group_idx]["mat_index"]]
            double_sided = material.doubleSided

            # parse normal map
            if material.normalTexture is not None:
                texture = glb.textures[material.normalTexture.index]
                uvs_used = material.normalTexture.texCoord
                image_index = texture.source
                image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                normal_texture = create_texture(np.array(image), None, "linear")

            # TODO: Parse occlusion
            if material.occlusionTexture is not None:
                texture = glb.textures[material.normalTexture.index]
                uvs_used = material.normalTexture.texCoord
                image_index = texture.source
                image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                occlusion_texture = create_texture(np.array(image), None, "linear")

            # parse alpha mode
            if material.alphaMode == "OPAQUE":
                alpha_cutoff = 0.0
            elif material.alphaMode == "MASK":
                alpha_cutoff = material.alphaCutoff
            else:
                alpha_cutoff = None

            # parse pbr roughness and metallic
            if material.pbrMetallicRoughness is not None:
                pbr_texture = material.pbrMetallicRoughness

                # parse metallic and roughness
                roughness_image = None
                metallic_image = None
                if pbr_texture.metallicRoughnessTexture is not None:
                    texture = glb.textures[pbr_texture.metallicRoughnessTexture.index]
                    uvs_used = pbr_texture.metallicRoughnessTexture.texCoord
                    image_index = texture.source
                    image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                    bands = image.split()
                    if len(bands) == 1:
                        roughness_image = np.array(bands[0])
                    else:
                        roughness_image = np.array(bands[1])  # G for roughness
                        metallic_image = np.array(bands[2])  # B for metallic
                        # metallic_image = np.array(bands[0])     # R for metallic????

                metallic_factor = None
                if pbr_texture.metallicFactor is not None:
                    metallic_factor = (pbr_texture.metallicFactor,)

                roughness_factor = None
                if pbr_texture.roughnessFactor is not None:
                    roughness_factor = (pbr_texture.roughnessFactor,)

                metallic_texture = create_texture(metallic_image, metallic_factor, "linear")
                roughness_texture = create_texture(roughness_image, roughness_factor, "linear")

                # Check if material has a base color texture
                color_image = None
                if pbr_texture.baseColorTexture is not None:
                    texture = glb.textures[pbr_texture.baseColorTexture.index]
                    uvs_used = pbr_texture.baseColorTexture.texCoord
                    image_index = texture.source
                    image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                    color_image = np.array(image.convert("RGBA"))

                # parse color
                color_factor = None
                if pbr_texture.baseColorFactor is not None:
                    color_factor = np.array(pbr_texture.baseColorFactor, dtype=float)

                color_texture = create_texture(color_image, color_factor, "srgb")

            elif "KHR_materials_pbrSpecularGlossiness" in material.extensions:
                extension_material = material.extensions["KHR_materials_pbrSpecularGlossiness"]
                color_image = None
                if "diffuseTexture" in extension_material:
                    texture = extension_material["diffuseTexture"]
                    uvs_used = texture["texCoord"]
                    image = Image.open(uri_to_PIL(glb.images[texture["index"]].uri))
                    color_image = np.array(image.convert("RGBA"))

                color_factor = None
                if "diffuseFactor" in extension_material:
                    color_factor = np.array(extension_material["diffuseFactor"], dtype=float)

                color_texture = create_texture(color_image, color_factor, "srgb")

            if color_texture is not None:
                opacity_texture = color_texture.check_dim(3)
                if opacity_texture is not None:
                    opacity_texture.apply_cutoff(alpha_cutoff)

            # TODO: Parse them!
            if "KHR_materials_specular" in material.extensions:
                extension_material = material.extensions["KHR_materials_specular"]
                if "specularColorFactor" in extension_material:
                    specular_color = np.array(extension_material["specularColorFactor"], dtype=float)

            if "KHR_materials_transmission" in material.extensions:
                extension_material = material.extensions["KHR_materials_transmission"]
                specular_transmission = extension_material["transmissionFactor"]  # e.g. 1

            if "KHR_materials_ior" in material.extensions:
                extension_material = material.extensions["KHR_materials_ior"]
                ior = extension_material["ior"]  # e.g. 1.4500000476837158

            if "KHR_materials_unlit" in material.extensions:
                # No unlit material implemented in renderers. Use emissive texture.
                if color_texture is not None:
                    emissive_texture = color_texture
                    color_texture = None
            else:
                # parse emissive
                emissive_image = None
                if material.emissiveTexture is not None:
                    texture = glb.textures[material.emissiveTexture.index]
                    uvs_used = material.emissiveTexture.texCoord
                    image_index = texture.source
                    image = Image.open(uri_to_PIL(glb.images[image_index].uri))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    emissive_image = np.array(image)

                emissive_factor = None
                if material.emissiveFactor is not None:
                    emissive_factor = np.array(material.emissiveFactor, dtype=float)

                if emissive_factor is not None and np.any(emissive_factor > 0.0):
                    emissive_texture = create_texture(emissive_image, emissive_factor, "srgb")

        # repair uv
        group_uvs = temp_infos[group_idx]["uvs1"] if uvs_used == 1 else temp_infos[group_idx]["uvs0"]
        group_points = temp_infos[group_idx]["points"]
        member_count = len(group_points)
        group_uv_exist = False

        for i in range(member_count):
            if group_uvs[i] is not None:
                group_uv_exist = True

        if group_uv_exist:
            for i in range(member_count):
                num_points = group_points[i].shape[0]
                if group_uvs[i] is None:
                    group_uvs[i] = np.zeros((num_points, 2), dtype=float)
            uvs = np.concatenate(group_uvs)
        else:
            uvs = None

        # build other group properties
        verts = np.concatenate(temp_infos[group_idx]["points"])
        normals = np.concatenate(temp_infos[group_idx]["normals"])
        faces = np.concatenate(temp_infos[group_idx]["triangles"])

        group_surface = surface.copy()
        group_surface.update_texture(
            color_texture=color_texture,
            opacity_texture=opacity_texture,
            roughness_texture=roughness_texture,
            metallic_texture=metallic_texture,
            normal_texture=normal_texture,
            emissive_texture=emissive_texture,
            ior=ior,
            double_sided=double_sided,
        )

        meshes.append(
            gs.Mesh.from_attrs(
                verts=verts,
                faces=faces,
                normals=normals,
                surface=group_surface,
                uvs=uvs,
                scale=scale,
            )
        )

    return meshes


def PIL_to_array(image):
    return np.array(image)


def uri_to_PIL(data_uri):
    with request.urlopen(data_uri) as response:
        data = response.read()
    return BytesIO(data)


def tonemapped(image):
    exposure = 0.5
    return (np.clip(np.power(image / 255 * np.power(2, exposure), 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def create_texture(image, factor, encoding):
    if image is not None:
        return gs.textures.ImageTexture(image_array=image, image_color=factor, encoding=encoding)
    elif factor is not None:
        return gs.textures.ColorTexture(color=factor)
    else:
        return None

def opacity_from_texture(color_texture, alpha_cutoff=None):
    opacity_texture = color_texture.check_dim(3)
    if opacity_texture is not None:
        if alpha_cutoff is not None:
            opacity_texture.apply_cutoff(alpha_cutoff)
        if isinstance(opacity_texture, gs.textures.ImageTexture) and opacity_texture.image_array.max() == opacity_texture.image_array.min():
            alpha = opacity_texture.image_array.max() / 255.0
            opacity_texture = create_texture(None, (alpha,), 'linear')
    return opacity_texture


def apply_transform(transform, positions, normals=None):
    transformed_positions = (np.column_stack([positions, np.ones(len(positions))]) @ transform)[:, :3]

    transformed_normals = normals
    if normals is not None:
        rot_mat = transform[:3, :3]
        if np.abs(3.0 - np.trace(rot_mat)) > gs.EPS**2:  # has rotation or scaling
            transformed_normals = normals @ rot_mat
            scale = np.linalg.norm(rot_mat, axis=1, keepdims=True)
            if np.any(np.abs(scale - 1.0) > gs.EPS):  # has scale
                transformed_normals /= np.linalg.norm(transformed_normals, axis=1, keepdims=True)

    return transformed_positions, transformed_normals

def create_frame(
    origin_radius=0.012, axis_radius=0.005, axis_length=1.0, head_radius=0.01, head_length=0.03, sections=12
):
    origin = create_sphere(radius=origin_radius, subdivisions=2)

    x = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.7, 0.0, 0.0, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )
    y = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.0, 0.7, 0.0, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )
    z = create_arrow(
        length=axis_length,
        radius=axis_radius,
        l_ratio=head_length / axis_length,
        r_ratio=head_radius / axis_radius,
        body_color=(0.0, 0.0, 0.7, 1.0),
        head_color=(0.7, 0.7, 0.7, 1.0),
        sections=sections,
    )

    x.vertices = gu.transform_by_R(x.vertices, gu.euler_to_R((0.0, 90.0, 0.0)))
    y.vertices = gu.transform_by_R(y.vertices, gu.euler_to_R((-90.0, 0.0, 0.0)))

    return trimesh.util.concatenate([origin, x, y, z])


def create_camera_frustum(camera, color):
    # camera
    camera_mesh = trimesh.load(os.path.join(get_src_dir(), "assets", "meshes", "camera/camera.obj"))

    # frustum
    near_half_height = camera.near * np.tan(np.deg2rad(camera.fov / 2))
    near_half_width = near_half_height * camera.aspect_ratio
    far_half_height = camera.far * np.tan(np.deg2rad(camera.fov / 2))
    far_half_width = far_half_height * camera.aspect_ratio

    # Define the vertices of the frustum
    vertices = np.array(
        [
            [0, 0, 0],  # apex
            [-near_half_width, -near_half_height, -camera.near],  # near bottom left
            [near_half_width, -near_half_height, -camera.near],  # near bottom right
            [near_half_width, near_half_height, -camera.near],  # near top right
            [-near_half_width, near_half_height, -camera.near],  # near top left
            [-far_half_width, -far_half_height, -camera.far],  # far bottom left
            [far_half_width, -far_half_height, -camera.far],  # far bottom right
            [far_half_width, far_half_height, -camera.far],  # far top right
            [-far_half_width, far_half_height, -camera.far],  # far top left
        ]
    )

    # Define the faces of the frustum
    faces = np.array(
        [
            # # near face
            # [1, 2, 3, 4],
            # # far face
            # [5, 6, 7, 8],
            # side face
            [2, 1, 5, 6],
            [3, 2, 6, 7],
            [4, 3, 7, 8],
            [1, 4, 8, 5],
        ]
    )

    # Create the frustum mesh
    frustum_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    frustum_mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(np.asarray(color, dtype=np.float32), (len(frustum_mesh.vertices), 1))
    )
    return trimesh.util.concatenate([camera_mesh, frustum_mesh])


def create_tets_mesh(n_tets=1, halfsize=1.0, quats=None, randomize_halfsize=True):
    """
    Create artistic tet-based mesh for rendering particles as tets.
    """
    # create tet-based particles given positions
    vert_per_tet = 12
    face_per_tet = 20
    if quats is None:
        quats = np.tile(gu.random_quaternion(n_tets), [1, vert_per_tet]).reshape(-1, 4)

    if randomize_halfsize:
        halfsize = (
            np.tile(np.random.uniform(0.3, 1.9, size=(n_tets, 1)), [1, vert_per_tet * 3]).reshape(-1, 3) * halfsize
        )
        halfsize = (
            np.tile(np.random.uniform(0.3, 1.9, size=(n_tets * 4, 1)), [1, vert_per_tet // 4 * 3]).reshape(-1, 3)
            * halfsize
        )
        # halfsize = np.random.uniform(0.2, 1.9, size=(n_tets * vert_per_tet, 3)) * halfsize

    vertices = (
        np.tile(
            np.array(
                [
                    [0.91835, 0.836701, 0.91835],
                    [0.91835, 0.91835, 0.836701],
                    [0.836701, 0.91835, 0.91835],
                    [-0.836701, 0.91835, -0.91835],
                    [-0.91835, 0.836701, -0.91835],
                    [-0.91835, 0.91835, -0.836701],
                    [-0.836701, -0.91835, 0.91835],
                    [-0.91835, -0.836701, 0.91835],
                    [-0.91835, -0.91835, 0.836701],
                    [0.91835, -0.836701, -0.91835],
                    [0.91835, -0.91835, -0.836701],
                    [0.836701, -0.91835, -0.91835],
                ]
            ),
            [n_tets, 1],
        )
        * halfsize
    )
    vertices = gu.transform_by_quat(vertices, quats)

    faces = np.tile(
        np.array(
            [
                [0, 6, 10],
                [11, 8, 4],
                [2, 5, 7],
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [0, 10, 9],
                [9, 1, 0],
                [1, 3, 5],
                [5, 2, 1],
                [2, 7, 6],
                [6, 0, 2],
                [4, 8, 7],
                [7, 5, 4],
                [8, 11, 10],
                [10, 6, 8],
                [3, 9, 11],
                [11, 4, 3],
                [1, 9, 3],
            ]
        ),
        [n_tets, 1],
    )
    faces_offset = np.tile(np.arange(0, n_tets).reshape(-1, 1) * vert_per_tet, [1, face_per_tet * 3]).reshape(
        n_tets * face_per_tet, 3
    )
    faces += faces_offset

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def transform_tets_mesh_verts(vertices, positions, zs=None):
    vert_per_tet = 12
    assert len(vertices) == len(positions) * vert_per_tet
    vertices = vertices.reshape(-1, vert_per_tet, 3)
    if zs is not None:
        assert len(zs) == len(positions)
        vertices = gu.transform_by_R(vertices, gu.z_up_to_R(zs))
    return (vertices + positions[:, np.newaxis]).reshape((-1, 3))


@lru_cache(maxsize=32)
def _create_unit_sphere_impl(subdivisions):
    mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=subdivisions)
    vertices, faces, face_normals = mesh.vertices.copy(), mesh.faces.copy(), mesh.face_normals.copy()
    for data in (vertices, faces, face_normals):
        data.flags.writeable = False
    return vertices, faces, face_normals


def create_sphere(radius, subdivisions=3, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces, face_normals = _create_unit_sphere_impl(subdivisions=subdivisions)
    vertices = vertices * radius
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile((np.asarray(color) * 255).astype(np.uint8), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache["face_normals"] = face_normals
    return mesh


@lru_cache(maxsize=32)
def _create_unit_cylinder_impl(sections):
    mesh = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=sections)
    vertices, faces, face_normals = mesh.vertices.copy(), mesh.faces.copy(), mesh.face_normals.copy()
    for data in (vertices, faces, face_normals):
        data.flags.writeable = False
    return vertices, faces, face_normals


def create_cylinder(radius, height, sections=None, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces, face_normals = _create_unit_cylinder_impl(sections=sections)
    vertices = vertices * (radius, radius, height)
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile((np.asarray(color) * 255).astype(np.uint8), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache["face_normals"] = face_normals
    return mesh


@lru_cache(maxsize=32)
def _create_unit_cone_impl(sections):
    mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=sections)
    vertices, faces, face_normals = mesh.vertices.copy(), mesh.faces.copy(), mesh.face_normals.copy()
    for data in (vertices, faces, face_normals):
        data.flags.writeable = False
    return vertices, faces, face_normals


def create_cone(radius, height, sections=None, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces, face_normals = _create_unit_cone_impl(sections=sections)
    vertices = vertices * (radius, radius, height)
    face_normals = face_normals / (radius, radius, height)
    face_normals /= np.linalg.norm(face_normals, axis=-1, keepdims=True)
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile((np.asarray(color) * 255).astype(np.uint8), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache["face_normals"] = face_normals
    return mesh


def create_arrow(
    length=1.0,
    radius=0.02,
    l_ratio=0.25,
    r_ratio=1.5,
    body_color=(1.0, 1.0, 1.0, 1.0),
    head_color=(1.0, 1.0, 1.0, 1.0),
    sections=12,
):
    r_head = radius * r_ratio
    r_body = radius
    l_head = length * l_ratio
    l_body = length - l_head

    head = create_cone(r_head, l_head, sections=sections, color=head_color)
    body = create_cylinder(r_body, l_body, sections=sections, color=body_color)
    face_normals = np.vstack((body._cache["face_normals"], head._cache["face_normals"]))
    face_normals.flags.writeable = False
    head._data["vertices"] += np.array([0.0, 0.0, l_body])
    body._data["vertices"] += np.array([0.0, 0.0, l_body / 2])

    vertices = np.vstack((body.vertices, head.vertices))
    faces = np.vstack((body.faces, head.faces + len(body.vertices)))
    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.vstack((body.visual.vertex_colors, head.visual.vertex_colors))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache["face_normals"] = face_normals
    return mesh


def create_line(start, end, radius=0.002, color=(1.0, 1.0, 1.0, 1.0), sections=12):
    vec = end - start
    length = np.linalg.norm(vec)
    mesh = create_cylinder(radius, length, sections, color)  # along z-axis
    mesh._data["vertices"][:, -1] += length / 2.0
    mesh.vertices = gu.transform_by_trans_R(mesh._data["vertices"], start, gu.z_up_to_R(vec))
    return mesh


@lru_cache(maxsize=1)
def _create_unit_box_impl():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    vertices, faces, face_normals = mesh.vertices.copy(), mesh.faces.copy(), mesh.face_normals.copy()
    for data in (vertices, faces, face_normals):
        data.flags.writeable = False
    return vertices, faces, face_normals


def create_box(extents=None, color=(1.0, 1.0, 1.0, 1.0), bounds=None, wireframe=False, wireframe_radius=0.002):
    if bounds is not None:
        bounds = np.asarray(bounds)
        extents = bounds[1] - bounds[0]
        pos = bounds.mean(axis=0)
    elif extents is not None:
        extents = np.asarray(extents)
        pos = np.zeros(3)
    else:
        gs.raise_exception("Neither `extents` nor `bounds` is provided.")

    if wireframe:
        box_vertices = np.asarray(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        box_vertices = box_vertices * extents + pos
        box_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

        n_verts = 0
        vertices, faces, face_normals = [], [], []
        for v_start, v_end in box_edges:
            p_start, p_end = box_vertices[v_start], box_vertices[v_end]
            vec = p_end - p_start
            length = np.linalg.norm(vec)

            line_vertices, line_faces, line_face_normals = _create_unit_cylinder_impl(sections=12)
            line_vertices = line_vertices * (wireframe_radius, wireframe_radius, length)
            line_vertices[:, -1] += length / 2.0
            line_vertices = gu.transform_by_trans_R(line_vertices, p_start, gu.z_up_to_R(vec))

            vertices.append(line_vertices)
            faces.append(line_faces + n_verts)
            face_normals.append(line_face_normals)
            n_verts += len(line_vertices)

        for vertex in box_vertices:
            sphere_vertices, sphere_faces, sphere_face_normals = _create_unit_sphere_impl(subdivisions=3)

            vertices.append(sphere_vertices * wireframe_radius + vertex)
            faces.append(sphere_faces + n_verts)
            face_normals.append(sphere_face_normals)
            n_verts += len(sphere_vertices)

        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        face_normals = np.concatenate(face_normals)
    else:
        vertices, faces, face_normals = _create_unit_box_impl()
        vertices = vertices * extents + pos

    visual = trimesh.visual.ColorVisuals()
    visual._data["vertex_colors"] = np.tile((np.asarray(color) * 255).astype(np.uint8), (len(vertices), 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh._cache.id_set()
    mesh._cache.cache["face_normals"] = face_normals

    return mesh


def create_plane(normal=(0.0, 0.0, 1.0), plane_size=(1e3, 1e3), tile_size=(1, 1), color=None):
def convexify(verts, faces, normals):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    if mesh.vertices.shape[0] > 3:
        mesh = trimesh.convex.convex_hull(mesh)
    return mesh.vertices.copy(), mesh.faces.copy(), mesh.vertex_normals.copy()

def decimate(verts, faces, normals, target_face_num):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    mesh = mesh.simplify_quadratic_decimation(target_face_num)
    return mesh.vertices.copy(), mesh.faces.copy(), mesh.vertex_normals.copy()

def get_unique_edges(vert: np.ndarray, face: np.ndarray) -> np.ndarray:
    '''
    Returns unique edges.

    Args:
    vert: (n_vert, 3) vertices
    face: (n_face, n_vert) face index array

    Returns:
    edges: 2-tuples of vertex indexes for each edge
    '''
    # return np.array([[0, 1]])

    r_face = np.roll(face, 1, axis=1)
    edges = np.concatenate(np.array([face, r_face]).T)

    # do a first pass to remove duplicates
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]

    return edges

ctype_to_numpy = {
    5120: (1, np.int8),     # BYTE
    5121: (1, np.uint8),    # UNSIGNED_BYTE
    5122: (2, np.int16),    # SHORT
    5123: (2, np.uint16),   # UNSIGNED_SHORT
    5124: (4, np.int32),    # INT
    5125: (4, np.uint32),   # UNSIGNED_INT
    5126: (4, np.float32),  # FLOAT
}

type_to_count = {
    "SCALAR": (1, []),
    "VEC2": (2, [2]),
    "VEC3": (3, [3]),
    "VEC4": (4, [4]),
    "MAT2": (4, [2, 2]),
    "MAT3": (9, [3, 3]),
    "MAT4": (16, [4, 4]),
}

def create_plane(size=1000, color=None, normal=(0, 0, 1)):
    thickness = 1e-2  # for safety
    mesh = trimesh.creation.box(extents=[plane_size[0], plane_size[1], thickness])
    mesh.vertices[:, 2] -= thickness / 2
    mesh.vertices = gu.transform_by_R(mesh.vertices, gu.z_up_to_R(np.asarray(normal, dtype=np.float32)))

    half_x, half_y = (plane_size[0] * 0.5, plane_size[1] * 0.5)
    verts = np.array(
        [
            [-half_x, -half_y, 0.0],
            [half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, -half_y, 0.0],
            [half_x, half_y, 0.0],
            [-half_x, half_y, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.arange(6, dtype=np.int32).reshape(-1, 3)
    vmesh = trimesh.Trimesh(verts, faces, process=False)
    vmesh.vertices[:, 2] -= thickness / 2
    vmesh.vertices = gu.transform_by_R(vmesh.vertices, gu.z_up_to_R(np.asarray(normal, dtype=np.float32)))
    if color is None:  # use checkerboard texture
        n_tile_x, n_tile_y = plane_size[0] / tile_size[0], plane_size[1] / tile_size[1]
        vmesh.visual = trimesh.visual.TextureVisuals(
            uv=np.array(
                [
                    [0, 0],
                    [n_tile_x, 0],
                    [n_tile_x, n_tile_y],
                    [0, 0],
                    [n_tile_x, n_tile_y],
                    [0, n_tile_y],
                ],
                dtype=np.float32,
            ),
            material=trimesh.visual.material.SimpleMaterial(
                image=Image.open(os.path.join(get_assets_dir(), "textures/checker.png")),
            ),
        )
    else:
        vmesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(np.asarray(color, dtype=np.float32), (len(vmesh.vertices), 1))
        )

    return vmesh, mesh


def generate_tetgen_config_from_morph(morph):
    if not isinstance(morph, gs.options.morphs.TetGenMixin):
        raise TypeError(
            f"Expected an instance of a class that inherits from TetGenMixin, but got an instance of {type(morph).name}."
        )
    return dict(
        order=morph.order,
        mindihedral=morph.mindihedral,
        minratio=morph.minratio,
        nobisect=morph.nobisect,
        quality=morph.quality,
        maxvolume=morph.maxvolume,
        verbose=morph.verbose,
    )


def make_tetgen_switches(cfg):
    """Build a TetGen switches string from a config dict."""
    flags = ["p"]

    if cfg.get("quality", True):
        r = cfg.get("minratio", 1.1)
        di = cfg.get("mindihedral", 10)
        flags.append(f"q{r}/{di}")

    a = cfg.get("maxvolume", -1.0)
    if a > 0:
        flags.append(f"a{a}")

    o = cfg.get("order", 1)
    if o != 1:
        flags.append(f"o{o}")

    if cfg.get("nobisect", False):
        flags.append("Y")

    v = cfg.get("verbose", 0)
    if v > 0:
        flags.append("V" * v)

    return "".join(flags)


def tetrahedralize_mesh(mesh, tet_cfg):
    # Importing pyvista and tetgen are very slow and not used very often. Let's delay import.
    import pyvista as pv
    import tetgen

    pv_obj = pv.PolyData(
        mesh.vertices, np.concatenate([np.full((mesh.faces.shape[0], 1), mesh.faces.shape[1]), mesh.faces], axis=1)
    )
    tet = tetgen.TetGen(pv_obj)
    # Build and apply the switches string directly, since
    # the Python wrapper sometimes ignores certain kwargs
    # (e.g. maxvolume). See: https://github.com/pyvista/tetgen/issues/24
    switches = make_tetgen_switches(tet_cfg)
    verts, elems = tet.tetrahedralize(switches=switches)
    # visualize_tet(tet, pv_obj, show_surface=False, plot_cell_qual=False)
    return verts, elems


def visualize_tet(tet, pv_data, show_surface=True, plot_cell_qual=False):
    grid = tet.grid
    if show_surface:
        grid.plot(show_edges=True)
    else:
        # get cell centroids
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = grid.points[cells].mean(axis=1)

        # extract cells below the 0 xy plane
        cell_ind = (cell_center[:, 2] < 0.0).nonzero(as_tuple=False)
        subgrid = grid.extract_cells(cell_ind)

        # advanced plotting
        if plot_cell_qual:
            cell_qual = subgrid.compute_cell_quality()["CellQuality"]
            subgrid.plot(
                scalars=cell_qual, stitle="Quality", cmap="bwr", clim=[0, 1], flip_scalars=True, show_edges=True
            )
        else:
            # Importing pyvista is very slow and not used very often. Let's delay import.
            import pyvista as pv

            plotter = pv.Plotter()
            plotter.add_mesh(subgrid, "lightgrey", lighting=True, show_edges=True)
            plotter.add_mesh(pv_data, "r", "wireframe")
            plotter.add_legend([[" Input Mesh ", "r"], [" Tessellated Mesh ", "black"]])
            plotter.show()


def check_exr_compression(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    exr_header = exr_file.header()
    if exr_header["compression"].v > Imath.Compression.PIZ_COMPRESSION:
        new_exr_path = get_exr_path(exr_path)
        if os.path.exists(new_exr_path):
            gs.logger.info(f"Assets of fixed compression detected and used: {new_exr_path}.")
        else:
            gs.logger.warning(
                f"EXR image {exr_path}'s compression type {exr_header['compression']} is not supported. "
                f"Converting to compression type ZIP_COMPRESSION and saving to {new_exr_path}."
            )

            channel_data = {channel: exr_file.channel(channel) for channel in exr_header["channels"]}
            exr_header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

            os.makedirs(os.path.dirname(new_exr_path), exist_ok=True)
            new_exr_file = OpenEXR.OutputFile(new_exr_path, exr_header)
            new_exr_file.writePixels(channel_data)
            new_exr_file.close()

        exr_path = new_exr_path

    exr_file.close()



def parse_visual_and_col_mesh_avatar(morph, surface):
    path = get_asset_path(morph.file)
    assert path.endswith('glb'), 'Only glb file is supported for avatar mesh.'
    vm_infos = parse_mesh_avatar(path, morph.group_by_material, morph.scale)

    # compute collision mesh
    m_infos = list()
    if morph.collision:
        if morph.merge_submeshes_for_collision:
            meshes = []
            for vm_info in vm_infos:
                meshes.append(trimesh.Trimesh(vm_info['vverts'], vm_info['vfaces'], vertex_normals=vm_info['vnormals'], process=False))
            mesh = trimesh.util.concatenate(meshes)
            verts, faces, normals = mesh.vertices, mesh.faces, mesh.vertex_normals

            m_info = dict()
            if morph.convexify:
                verts, faces, normals = convexify(verts, faces, normals)
                m_info['is_convex'] = True

            if morph.decimate:
                verts, faces, normals = decimate(verts, faces, normals, morph.decimate_face_num)
                m_info['is_convex'] = False

            edges = get_unique_edges(verts, faces)

            m_info['verts']   = verts
            m_info['faces']   = faces
            m_info['edges']   = edges
            m_info['normals'] = normals

            m_infos.append(m_info)

        else:
            for vm_info in vm_infos:
                m_info = dict()
                verts, faces, normals = vm_info['vverts'], vm_info['vfaces'], vm_info['vnormals']

                if morph.is_convex:
                    verts, faces, normals = convexify(verts, faces, normals)
                    m_info['is_convex'] = True

                if morph.decimate:
                    verts, faces, normals = decimate(verts, faces, normals, morph.decimate_face_num)
                    m_info['is_convex'] = False

                edges = get_unique_edges(verts, faces)

                m_info['verts']   = verts
                m_info['faces']   = faces
                m_info['edges']   = edges
                m_info['normals'] = normals

                m_infos.append(m_info)

    return vm_infos, m_infos


def parse_mesh_avatar(path, group_by_material, scale):
    glb = pygltflib.GLTF2().load(path)
    assert glb is not None

    def parse_tree(node_index):
        node = glb.nodes[node_index]
        if node.matrix is not None:
            matrix = np.array(node.matrix, dtype=float).reshape((4, 4))
        else:
            matrix = np.identity(4, dtype=float)
            if node.translation is not None:
                translation = np.array(node.translation, dtype=float)
                translation_matrix = np.identity(4, dtype=float)
                translation_matrix[3, :3] = translation
                matrix = translation_matrix @ matrix
            if node.rotation is not None:
                rotation = np.array(node.rotation, dtype=float)     # xyzw
                rotation_matrix = np.identity(4, dtype=float)
                rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
                rotation_matrix[:3, :3] = trimesh.transformations.quaternion_matrix(rotation)[:3, :3].T
                matrix = rotation_matrix @ matrix
            if node.scale is not None:
                scale = np.array(node.scale, dtype=float)
                scale_matrix = np.diag(np.append(scale, 1))
                matrix = scale_matrix @ matrix
        mesh_list = list()
        if node.mesh is not None:
            mesh_list.append([node.mesh, np.identity(4, dtype=float)])
        for sub_node_index in node.children:
            sub_mesh_list = parse_tree(sub_node_index)
            mesh_list.extend(sub_mesh_list)
        for i in range(len(mesh_list)):
            mesh_list[i][1] = mesh_list[i][1] @ matrix
        return mesh_list

    def get_bufferview_data(buffer_view):
        buffer = glb.buffers[buffer_view.buffer]
        return glb.get_data_from_buffer_uri(buffer.uri)

    def get_data_from_accessor(accessor_index):
        accessor = glb.accessors[accessor_index]
        buffer_view = glb.bufferViews[accessor.bufferView]
        buffer_data = get_bufferview_data(buffer_view)
        data_type, data_ctype, data_count = accessor.type, accessor.componentType, accessor.count
        data_length = data_count * type_to_count[data_type][0] * ctype_to_numpy[data_ctype][0]
        data_start = buffer_view.byteOffset + accessor.byteOffset
        return np.frombuffer(
            buffer_data[data_start: data_start + data_length],
            dtype=ctype_to_numpy[data_ctype][1],
            count=data_count * type_to_count[data_type][0]
        ).reshape([data_count] + type_to_count[data_type][1])

    glb.convert_images(pygltflib.ImageFormat.DATAURI)

    # binary_blob = glb.binary_blob()
    scene = glb.scenes[glb.scene]

    # for n in glb.nodes:
    #     # set scale to None for speed up
    #     n.scale = None
    #     if n.translation is not None:
    #         if "mixamo" in path:
    #             n.translation = list(map(lambda x:x*0.01, n.translation))

    mesh_list = list()
    for node_index in scene.nodes:
        root_mesh_list = parse_tree(node_index)
        mesh_list.extend(root_mesh_list)

    temp_infos = dict()
    primitive_id = 0
    for i in tqdm(range(len(mesh_list))):
        mesh = glb.meshes[mesh_list[i][0]]
        matrix = mesh_list[i][1]
        for primitive in mesh.primitives:
            if group_by_material:
                group_idx = primitive.material
            else:
                group_idx = primitive_id

            primitive_id += 1

            if 'KHR_draco_mesh_compression' in primitive.extensions:
                KHR_index = primitive.extensions['KHR_draco_mesh_compression']['bufferView']
                mesh_buffer_view = glb.bufferViews[KHR_index]
                mesh_data = get_bufferview_data(mesh_buffer_view)
                mesh = DracoPy.decode(mesh_data[
                                      mesh_buffer_view.byteOffset:
                                      mesh_buffer_view.byteOffset + mesh_buffer_view.byteLength
                                      ])
                points = mesh.points
                triangles = mesh.faces
                normals = mesh.normals if len(mesh.normals) > 0 else None
                uvs = mesh.tex_coord if len(mesh.tex_coord) > 0 else None

            else:
                # "primitive.attributes" records accessor indices in "glb.accessors", like:
                #      Attributes(POSITION=2, NORMAL=1, TANGENT=None, TEXCOORD_0=None, TEXCOORD_1=None,
                #                 COLOR_0=None, JOINTS_0=None, WEIGHTS_0=None)
                # parse vertices
                points = get_data_from_accessor(primitive.attributes.POSITION).astype(float)

                # parse faces
                triangles = get_data_from_accessor(primitive.indices).astype(np.int32).reshape(-1, 3)

                # parse normals
                if primitive.attributes.NORMAL:
                    normals = get_data_from_accessor(primitive.attributes.NORMAL).astype(float)
                else:
                    normals = None

                # parse uvs
                if primitive.attributes.TEXCOORD_0:
                    # and glb.materials[primitive.material].pbrMetallicRoughness.baseColorTexture is not None:
                    # Roughness and metallic also need uvs
                    uvs = get_data_from_accessor(primitive.attributes.TEXCOORD_0).astype(float)
                else:
                    uvs = None

            # skinned vertices

            # 1. parse joints
            # - these arrays define the joints that affect each vertice
            # - maximum 4 joints for each vertice
            joints = get_data_from_accessor(primitive.attributes.JOINTS_0).reshape(-1, 4).astype(int)

            # 2. parse weights
            # - new transform matrix for each point is weighted sum of all related joints
            weights = get_data_from_accessor(primitive.attributes.WEIGHTS_0).reshape(-1, 4).astype(float)

            # 3. get skin parameters
            # - joint order are defined in the skin_joints
            skin = glb.skins[0]

            # 5. parse global transform of joint nodes
            global_transforms = np.asarray(
                [it[1] for it in sorted(
                    parse_global_transform(glb.nodes, scene.nodes[-1]),
                    key=lambda x:x[0]
                )]
            )
            globaljoint_mat = global_transforms[skin.joints]

            # 6. get inverse transform
            # invbindjoint_mat = np.array(
            #     [np.matrix(globaljoint_mat[i]).I for i in range(globaljoint_mat.shape[0])]
            # )
            invbindjoint_mat = get_data_from_accessor(skin.inverseBindMatrices).reshape(-1, 4, 4).astype(float)

            # 7. calculate skin mat for each points
            skin_mat = invbindjoint_mat @ globaljoint_mat
            skin_mat = (
                    weights * skin_mat[joints].transpose(2,3,0,1)
            ).transpose(2,3,0,1).sum(axis=1)

            # 8. transform points
            points_homo = np.append(points, np.ones((points.shape[0], 1)), axis=-1)[:,None]
            points = (points_homo @ skin_mat)[:,0,:3]
            normals_homo = np.append(normals, np.ones((normals.shape[0], 1)), axis=-1)[:,None]
            normals = (normals_homo @ skin_mat)[:,0,:3]

            matrix /= np.abs(matrix).max(axis=-1)
            # matrix = matrix * np.abs(np.diag(1 / matrix.diagonal()))

            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals
            points, normals = apply_transform(matrix, points, normals)

            if group_idx not in temp_infos.keys():
                temp_infos[group_idx] = {
                    'mat_index' : primitive.material,
                    'points'    : [points],
                    'triangles' : [triangles],
                    'normals'   : [normals],
                    'uvs'       : [uvs],
                    'n_points'  : len(points),
                    'init_matrix'  : [matrix],
                    'vert_joints'  : [joints],
                    'vert_invbind' : [invbindjoint_mat],
                    'vert_weights' : [weights],
                    'skin_joints'  : [np.asarray(skin.joints)],
                    'nodes'        : [[scene.nodes[-1], glb.nodes]]
                }
            else:
                triangles += temp_infos[group_idx]['n_points']
                temp_infos[group_idx]['points'].append(points)
                temp_infos[group_idx]['triangles'].append(triangles)
                temp_infos[group_idx]['normals'].append(normals)
                temp_infos[group_idx]['uvs'].append(uvs)
                temp_infos[group_idx]['n_points'] += len(points)
                temp_infos[group_idx]['init_matrix'].append(matrix)
                temp_infos[group_idx]['vert_joints'].append(joints)
                temp_infos[group_idx]['vert_invbind'].append(invbindjoint_mat)
                temp_infos[group_idx]['vert_weights'].append(weights)
                temp_infos[group_idx]['skin_joints'].append(np.asarray(skin.joints))
                temp_infos[group_idx]['nodes'].append([scene.nodes[-1], glb.nodes])

    infos = list()
    for group_idx in temp_infos.keys():
        # parse images
        color           = None
        color_image     = None
        # metallic_image  = None
        # roughness_image = None

        if temp_infos[group_idx]['mat_index'] is not None:
            material = glb.materials[temp_infos[group_idx]['mat_index']]

            if material.pbrMetallicRoughness is not None:
                # Check if material has a base color texture
                if material.pbrMetallicRoughness.baseColorTexture is not None:
                    texture = glb.textures[material.pbrMetallicRoughness.baseColorTexture.index]
                    image_index = texture.source
                    image_data = uri_to_PIL(glb.images[image_index].uri)
                    color_image = Image.open(image_data).convert("RGBA")

                # parse color
                if material.pbrMetallicRoughness.baseColorFactor is not None:
                    color_factor = material.pbrMetallicRoughness.baseColorFactor
                    color_factor = np.array(color_factor, dtype=float)

                    if color_image is not None:
                        bands = color_image.split()
                        color_image = Image.merge("RGBA", [
                            Image.eval(bands[i], lambda x: x * color_factor[i]) for i in range(4)
                        ])
                    else:
                        color = color_factor

        elif 'KHR_materials_pbrSpecularGlossiness' in material.extensions:
            extension_material = material.extensions['KHR_materials_pbrSpecularGlossiness']
            if 'diffuseTexture' in extension_material:
                color_image = Image.open(uri_to_PIL(glb.images[extension_material['diffuseTexture']['index']].uri)).convert("RGBA")

                if 'diffuseFactor' in extension_material:
                    color_factor = np.array(extension_material['diffuseFactor'], dtype=float)
                    if color_image is not None:
                        bands = color_image.split()
                        color_image = Image.merge("RGBA", [
                            Image.eval(bands[i], lambda x: x * color_factor[i]) for i in range(4)
                        ])
                    else:
                        color = color_factor

        if color_image is not None:
            image = np.array(color_image)
        else:
            image = None

        # repair uv
        group_uvs = temp_infos[group_idx]['uvs']
        group_points = temp_infos[group_idx]['points']
        member_count = len(group_points)
        group_uv_exist = False

        for i in range(member_count):
            if group_uvs[i] is not None:
                group_uv_exist = True

        if group_uv_exist:
            for i in range(member_count):
                num_points = group_points[i].shape[0]
                if group_uvs[i] is None:
                    group_uvs[i] = np.zeros((num_points, 2), dtype=float)
            uvs = np.concatenate(group_uvs)
        else:
            uvs = None

        # build other group properties
        vverts   = np.concatenate(temp_infos[group_idx]['points'])
        vnormals = np.concatenate(temp_infos[group_idx]['normals'])
        vfaces   = np.concatenate(temp_infos[group_idx]['triangles'])

        vverts *= scale

        init_matrix  = np.concatenate(temp_infos[group_idx]['init_matrix'])
        vert_joints  = np.concatenate(temp_infos[group_idx]['vert_joints'])
        vert_invbind = np.concatenate(temp_infos[group_idx]['vert_invbind'])
        vert_weights = np.concatenate(temp_infos[group_idx]['vert_weights'])
        skin_joints  = np.concatenate(temp_infos[group_idx]['skin_joints'])
        nodes  = temp_infos[group_idx]['nodes'][0]

        return_info = {
            'vverts'   : vverts,
            'vfaces'   : vfaces,
            'vnormals' : vnormals,
            'color'    : color,
            'image'    : image,
            'uvs'      : uvs,
            'init_matrix'  : init_matrix,
            'vert_joints'  : vert_joints,
            'vert_invbind' : vert_invbind,
            'vert_weights' : vert_weights,
            'skin_joints'  : skin_joints,
            'nodes'        : nodes
            }
        infos.append(return_info)

    return infos


def parse_global_transform(nodes, node_index, root_matrix=None):
    matrix_list = list()
    node = nodes[node_index]
    matrix = parse_matrix(nodes, node_index)

    if root_matrix is not None:
        matrix = matrix @ root_matrix

    matrix_list.append([node_index, matrix])

    for sub_node_index in node.children:
        sub_matrix_list = parse_global_transform(nodes, sub_node_index, matrix)
        matrix_list.extend(sub_matrix_list)

    return matrix_list

def parse_matrix(nodes, node_index):
    node = nodes[node_index]
    if node.matrix is not None:
        matrix = np.array(node.matrix, dtype=float).reshape((4, 4))
    else:
        matrix = np.identity(4, dtype=float)
        scale_matrix, rotation_matrix, translation_matrix = \
            np.identity(4), np.identity(4), np.identity(4)
        if node.translation is not None:
            translation = np.array(node.translation, dtype=float)
            translation_matrix = np.identity(4, dtype=float)
            translation_matrix[3, :3] = translation
        if node.rotation is not None:
            rotation = np.array(node.rotation, dtype=float)     # xyzw
            rotation_matrix = np.identity(4, dtype=float)
            rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
            rotation_matrix[:3, :3] = trimesh.transformations.quaternion_matrix(rotation)[:3, :3].T
        if node.scale is not None:
            scale = np.array(node.scale, dtype=float)
            scale_matrix = np.diag(np.append(scale, 1))
        if node.name == "mixamorig:Hips":
            matrix = scale_matrix @ rotation_matrix @ translation_matrix
        else:
            matrix = scale_matrix @ rotation_matrix @ translation_matrix
    return matrix