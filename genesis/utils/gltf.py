from io import BytesIO
from urllib import request
import numpy as np
import pygltflib
import trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image

import genesis as gs

from tqdm import tqdm
from . import mesh as mu


ctype_to_numpy = {
    5120: (1, np.int8),  # BYTE
    5121: (1, np.uint8),  # UNSIGNED_BYTE
    5122: (2, np.int16),  # SHORT
    5123: (2, np.uint16),  # UNSIGNED_SHORT
    5124: (4, np.int32),  # INT
    5125: (4, np.uint32),  # UNSIGNED_INT
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

alpha_modes = {
    "OPAQUE": 0,
    "MASK": 1,
    "BLEND": 2,
}

def uri_to_PIL(data_uri):
    with request.urlopen(data_uri) as response:
        data = response.read()
    return BytesIO(data)

def get_glb_bufferview_data(glb, buffer_view):
    buffer = glb.buffers[buffer_view.buffer]
    return glb.get_data_from_buffer_uri(buffer.uri)


def get_glb_data_from_accessor(glb, accessor_index):
    accessor = glb.accessors[accessor_index]
    buffer_view = glb.bufferViews[accessor.bufferView]
    buffer_data = get_glb_bufferview_data(glb, buffer_view)

    data_type, data_ctype, count = accessor.type, accessor.componentType, accessor.count
    num_components = type_to_count[data_type][0]
    dtype = ctype_to_numpy[data_ctype]
    itemsize = np.dtype(dtype).itemsize

    # Extract data considering byteStride
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    byte_stride = buffer_view.byteStride

    if not byte_stride or byte_stride == num_components * itemsize:
        # Data is tightly packed
        byte_length = count * num_components * itemsize
        data = buffer_data[byte_offset : byte_offset + byte_length]
        array = np.frombuffer(data, dtype=dtype)

    else:
        # Data is interleaved
        array = np.zeros((count, num_components), dtype=dtype)
        for i in range(count):
            start = byte_offset + i * byte_stride
            end = start + num_components * itemsize
            data_slice = buffer_data[start:end]
            array[i] = np.frombuffer(data_slice, dtype=dtype, count=num_components)

    return array.reshape([count, *type_to_count[data_type][1]])

def get_glb_image(glb, image_index, image_type=None):
    if image_index is not None:
        image = Image.open(uri_to_PIL(glb.images[image_index].uri))
        if image_type is not None:
            image = image.convert(image_type)
        return np.array(image)
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

def parse_glb_material(glb, material_index, surface):
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

    material = glb.materials[material_index]
    double_sided = material.doubleSided

    # parse normal map
    if material.normalTexture is not None:
        texture = glb.textures[material.normalTexture.index]
        if material.normalTexture.texCoord is not None:
            uvs_used = material.normalTexture.texCoord
        normal_image = get_glb_image(glb, texture.source)
        if normal_image is not None:
            normal_texture = mu.create_texture(normal_image, None, "linear")

    # TODO: Parse occlusion
    if material.occlusionTexture is not None:
        texture = glb.textures[material.occlusionTexture.index]
        if material.occlusionTexture.texCoord is not None:
            uvs_used = material.occlusionTexture.texCoord
        occlusion_image = get_glb_image(glb, texture.source)
        if occlusion_image is not None:
            occlusion_texture = mu.create_texture(occlusion_image, None, "linear")

    # parse alpha mode
    alpha_cutoff = mu.adjust_alpha_cutoff(alpha_cutoff, alpha_modes[material.alphaMode])

    # parse pbr roughness and metallic
    if material.pbrMetallicRoughness is not None:
        pbr_texture = material.pbrMetallicRoughness

        # parse metallic and roughness
        roughness_image = None
        metallic_image = None
        if pbr_texture.metallicRoughnessTexture is not None:
            texture = glb.textures[pbr_texture.metallicRoughnessTexture.index]
            if pbr_texture.metallicRoughnessTexture.texCoord is not None:
                uvs_used = pbr_texture.metallicRoughnessTexture.texCoord

            combined_image = get_glb_image(glb, texture.source)
            if combined_image is not None:
                if combined_image.ndim == 2:
                    roughness_image = combined_image
                else:
                    roughness_image = combined_image[:, :, 1]  # G for roughness
                    metallic_image = combined_image[:, :, 2]  # B for metallic
                    # metallic_image = np.array(bands[0])     # R for metallic????

        metallic_factor = None
        if pbr_texture.metallicFactor is not None:
            metallic_factor = (pbr_texture.metallicFactor,)

        roughness_factor = None
        if pbr_texture.roughnessFactor is not None:
            roughness_factor = (pbr_texture.roughnessFactor,)

        metallic_texture = mu.create_texture(metallic_image, metallic_factor, "linear")
        roughness_texture = mu.create_texture(roughness_image, roughness_factor, "linear")

        # Check if material has a base color texture
        color_image = None
        if pbr_texture.baseColorTexture is not None:
            texture = glb.textures[pbr_texture.baseColorTexture.index]
            if pbr_texture.baseColorTexture.texCoord is not None:
                uvs_used = pbr_texture.baseColorTexture.texCoord
            if "KHR_texture_basisu" in texture.extensions:
                gs.logger.warning(
                    f"Mesh file `{glb.path}` uses 'KHR_texture_basisu' extension for supercompression of texture "
                    "images, which is unsupported. Ignoring texture."
                )
            color_image = get_glb_image(glb, texture.source, "RGBA")

        # parse color
        color_factor = None
        if pbr_texture.baseColorFactor is not None:
            color_factor = np.array(pbr_texture.baseColorFactor, dtype=np.float32)

        color_texture = mu.create_texture(color_image, color_factor, "srgb")

    elif "KHR_materials_pbrSpecularGlossiness" in material.extensions:
        extension_material = material.extensions["KHR_materials_pbrSpecularGlossiness"]
        color_image = None
        if "diffuseTexture" in extension_material:
            texture = extension_material["diffuseTexture"]
            if texture.get("texCoord") is not None:
                uvs_used = texture["texCoord"]
            color_image = get_glb_image(glb, texture.get("index"), "RGBA")

        color_factor = None
        if "diffuseFactor" in extension_material:
            color_factor = np.array(extension_material["diffuseFactor"], dtype=np.float32)

        color_texture = mu.create_texture(color_image, color_factor, "srgb")
        material.extensions.pop("KHR_materials_pbrSpecularGlossiness")

    if color_texture is not None:
        opacity_texture = opacity_from_texture(color_texture, alpha_cutoff)

    if "KHR_materials_unlit" in material.extensions:
        # No unlit material implemented in renderers. Use emissive texture.
        if color_texture is not None:
            emissive_texture = color_texture
            color_texture = None
        material.extensions.pop("KHR_materials_unlit")
    else:
        # parse emissive
        emissive_image = None
        if material.emissiveTexture is not None:
            texture = glb.textures[material.emissiveTexture.index]
            if material.emissiveTexture.texCoord is not None:
                uvs_used = material.emissiveTexture.texCoord
            image_index = texture.source
            image = Image.open(uri_to_PIL(glb.images[image_index].uri))
            if image.mode != 'RGB':
                img = image.convert('RGB')
            else:
                img = image
            emissive_image = np.asarray(img)

        emissive_factor = None
        if material.emissiveFactor is not None:
            emissive_factor = np.array(material.emissiveFactor, dtype=np.float32)

        if emissive_factor is not None and np.any(emissive_factor > 0.0):  # Make sure to check emissive
            emissive_texture = mu.create_texture(emissive_image, emissive_factor, "srgb")

    # TODO: Parse them!
    for extension_name, extension_material in material.extensions.items():
        if extension_name == "KHR_materials_specular":
            specular_weight = extension_material.get("specularFactor", 1.0)
            specular_color = np.array(extension_material.get("specularColorFactor", [1.0, 1.0, 1.0]), dtype=np.float32)

        elif extension_name == "KHR_materials_clearcoat":
            clearcoat_weight = extension_material.get("clearcoatFactor", 0.0)
            clearcoat_roughness_factor = (extension_material["clearcoatRoughnessFactor"],)

        elif extension_name == "KHR_materials_volume":
            attenuation_distance = extension_material["attenuationDistance"]

        elif extension_name == "KHR_materials_transmission":
            specular_trans_factor = extension_material.get("transmissionFactor", 0.0)  # e.g. 1

        elif extension_name == "KHR_materials_ior":
            ior = extension_material["ior"]  # e.g. 1.4500000476837158

    material_surface = surface.copy()
    # In project ViCo, we do not use base color texture, instead we use emissive texture.
    # To adapt to this, we set color_texture to emissive_texture if emissive_texture is not None.
    # This is a temporary solution. To support both color_texture and emissive_texture,
    # we need to modify the surface_uvs_to_trimesh_visual() method to make it support PBRMaterial instead of just SimpleMaterial.
    if emissive_texture is not None:
        color_texture = emissive_texture
    material_surface.update_texture(
        color_texture=color_texture,
        opacity_texture=opacity_texture,
        roughness_texture=roughness_texture,
        metallic_texture=metallic_texture,
        normal_texture=normal_texture,
        emissive_texture=emissive_texture,
        ior=ior,
        double_sided=double_sided,
    )

    return material_surface, uvs_used, material.name


def parse_glb_tree(glb, node_index):
    node = glb.nodes[node_index]
    non_identity = False
    if node.matrix is not None:
        transform = np.array(node.matrix, dtype=np.float32).reshape((4, 4))
        non_identity = True
    else:
        transform = np.identity(4, dtype=np.float32)
        if node.translation is not None:
            transform[:3, 3] = node.translation
            non_identity = True
        if node.rotation is not None:
            transform[:3, :3] = R.from_quat(node.rotation).as_matrix()  # xyzw
            non_identity = True
        if node.scale is not None:
            transform[:3, :3] *= node.scale
            non_identity = True
        transform = transform.T  # translation at bottom

    mesh_list = []
    for sub_node_index in node.children:
        sub_mesh_list = parse_glb_tree(glb, sub_node_index)
        mesh_list += sub_mesh_list
    if non_identity:
        for _, mesh_transform in mesh_list:
            mesh_transform @= transform

    if node.mesh is not None:
        mesh_list.append([node.mesh, transform])
    return mesh_list

def parse_mesh_glb(path, group_by_material, scale, surface):
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
                rotation = np.array(node.rotation, dtype=float)  # xyzw
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
                mesh = DracoPy.decode(mesh_data[
                                      mesh_buffer_view.byteOffset:
                                      mesh_buffer_view.byteOffset + mesh_buffer_view.byteLength
                                      ])
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
    gs.logger.debug(f"meshes loaded, start to load materials")
    for group_idx in tqdm(temp_infos.keys()):
        # parse images
        color_texture     = None
        opacity_texture   = None
        roughness_texture = None
        metallic_texture  = None
        normal_texture    = None
        emissive_texture  = None

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
                opacity_texture = opacity_from_texture(color_texture, alpha_cutoff)

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
                emissive_image = None
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

        # In project Ella, we do not use base color texture, instead we use emissive texture.
        # To adapt to this, we set color_texture to emissive_texture if emissive_texture is not None.
        # This is a temporary solution. To support both color_texture and emissive_texture,
        # we need to modify the surface_uvs_to_trimesh_visual() method to make it support PBRMaterial instead of just SimpleMaterial.
        if emissive_texture is not None:
            color_texture = emissive_texture

        group_surface = gs.surfaces.Default()

        group_surface.update_texture(
            color_texture     = color_texture,
            opacity_texture   = opacity_texture,
            roughness_texture = roughness_texture,
            metallic_texture  = metallic_texture,
            normal_texture    = normal_texture,
            emissive_texture  = emissive_texture,
            ior               = ior,
            double_sided      = double_sided,
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

def apply_transform(matrix, positions, normals=None):
    n = positions.shape[0]
    transformed_positions = (np.hstack([positions, np.ones((n, 1))]) @ matrix)[:, :3]
    if normals is not None:
        transformed_normals = (np.hstack([normals, np.zeros((n, 1))]) @ matrix)[:, :3]
    else:
        transformed_normals = None
    return transformed_positions, transformed_normals

def create_texture(image, factor, encoding):
    if image is not None:
        return gs.textures.ImageTexture(image_array=image, image_color=factor, encoding=encoding)
    elif factor is not None:
        return gs.textures.ColorTexture(color=factor)
    else:
        return None