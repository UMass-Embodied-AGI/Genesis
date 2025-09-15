import psutil
import genesis as gs
from . import mesh as mu
import os
import re

from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, UsdSemantics
from scipy.spatial.transform import Rotation as R
import trimesh
import numpy as np
from PIL import Image

cs_encode = {
    "raw": "linear",
    "sRGB": "srgb",
    "auto": None,
    "": None,
}


def log_memory_usage(label):
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"[MEM][{label}]: {mem:.2f} MB")

def make_tuple(value):
    if value is None:
        return None
    else:
        return (value,)

def create_flipped_texture(image, factor, encoding):
    """Create a texture and flip vertically if it's an image."""
    if image is not None:
        if image.dtype != np.uint8:
            if image.max() <= 1.0:        
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:                           
                image = image.astype(np.uint8)
        flipped_image = np.flipud(image)
        return gs.textures.ImageTexture(image_array=flipped_image, image_color=factor, encoding=encoding)
    elif factor is not None:
        return gs.textures.ColorTexture(color=factor)
    else:
        return None


def load_texture(texture):
    """Safely load a texture: if it's already a numpy array, return it directly."""
    if texture is None:
        return None
    if isinstance(texture, np.ndarray):
        arr = texture
    else:
        img = Image.open(texture.resolvedPath if hasattr(texture, "resolvedPath") else texture)
        arr = np.array(img)    

    h, w = arr.shape[:2]
    scale = max(h, w) / 256 if max(h, w) > 256 else 1.0
    if scale > 1.0: 
        new_w = int(w / scale)
        new_h = int(h / scale)
        img_resized = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)
        arr = np.array(img_resized)
    return arr

def get_input_attribute_value(shader, input_name, input_type=None):
    shader_input = shader.GetInput(input_name)
    
    if input_type != "value":
        shader_input_attr = shader_input.GetValueProducingAttribute()[0]
        if shader_input_attr.IsValid():
            return UsdShade.Shader(shader_input_attr.GetPrim()), shader_input_attr.GetBaseName()

    if input_type != "attribute":
        return shader_input.Get(), None
    return None, None

# --- parsing functions (parse_preview_surface, parse_gltf_surface, parse_omni_surface) ---

def parse_preview_surface(shader, output_name):
    shader_id = shader.GetShaderId()
    if shader_id == "UsdPreviewSurface":
        uvname = None

        def parse_component(component_name, component_encode):
            component, component_output = get_input_attribute_value(shader, component_name)
            if component_output is None:
                component_factor = component
                component_image = None
            else:
                component_image, component_overencode, component_uvname = parse_preview_surface(component, component_output)
                if component_overencode is not None:
                    component_encode = component_overencode
                component_factor = None

            component_texture = create_flipped_texture(component_image, component_factor, component_encode)
            return component_texture, component_uvname

        color_texture, color_uvname = parse_component("diffuseColor", "srgb")
        if color_uvname is not None:
            uvname = color_uvname

        opacity_texture, opacity_uvname = parse_component("opacity", "linear")
        if opacity_uvname is not None and uvname is None:
            uvname = opacity_uvname
        alpha_cutoff = get_input_attribute_value(shader, "opacityThreshold", "value")[0]
        opacity_texture.apply_cutoff(alpha_cutoff)

        emissive_texture, emissive_uvname = parse_component("emissiveColor", "srgb")
        if emissive_uvname is not None and uvname is None:
            uvname = emissive_uvname

        use_metallic = get_input_attribute_value(shader, "useSpecularWorkflow", "value")[0] == 0
        if use_metallic:
            metallic_texture, metallic_uvname = parse_component("metallic", "linear")
            if metallic_uvname is not None and uvname is None:
                uvname = metallic_uvname
        else:
            metallic_texture = None

        roughness_texture, roughness_uvname = parse_component("roughness", "linear")
        if roughness_uvname is not None and uvname is None:
            uvname = roughness_uvname

        normal_texture, normal_uvname = parse_component("normal", "linear")
        if normal_uvname is not None and uvname is None:
            uvname = normal_uvname

        ior = get_input_attribute_value(shader, "ior", "value")[0]

        return {
            "color_texture": color_texture,
            "opacity_texture": opacity_texture,
            "roughness_texture": roughness_texture,
            "metallic_texture": metallic_texture,
            "emissive_texture": emissive_texture,
            "normal_texture": normal_texture,
            "ior": ior,
        }, uvname

    elif shader_id == "UsdUVTexture":
        texture = get_input_attribute_value(shader, "file", "value")[0]
        texture_image = load_texture(texture)
        texture_encode = get_input_attribute_value(shader, "sourceColorSpace", "value")[0]
        texture_encode = cs_encode[texture_encode]

        texture_uvs_shader, texture_uvs_output = get_input_attribute_value(shader, "st", "attribute")
        texture_uvs_name = parse_preview_surface(texture_uvs_shader, texture_uvs_output)

        if output_name == "r":
            texture_image = texture_image[:, :, 0]
        elif output_name == "g":
            texture_image = texture_image[:, :, 1]
        elif output_name == "b":
            texture_image = texture_image[:, :, 2]
        elif output_name == "a":
            texture_image = texture_image[:, :, 3]
        elif output_name == "rgb":
            texture_image = texture_image[:, :, :3]
        else:
            gs.raise_exception(f"Invalid output channel for UsdUVTexture: {output_name}.")

        return texture_image, texture_encode, texture_uvs_name

    elif shader_id.startswith("UsdPrimvarReader"):
        primvar_name = get_input_attribute_value(shader, "varname", "value")[0]
        return primvar_name

def parse_gltf_surface(shader, source_type, output_name):
    shader_subid = shader.GetSourceAssetSubIdentifier(source_type)
    if shader_subid == "gltf_material":
        color_factor = get_input_attribute_value(shader, "base_color_factor", "value")[0]
        color_texture_shader, color_texture_output = get_input_attribute_value(shader, "base_color_texture", "attribute")
        if color_texture_shader is not None:
            color_image = parse_gltf_surface(color_texture_shader, source_type, color_texture_output)
        else:
            color_image = None
        color_texture = create_flipped_texture(color_image, color_factor, "srgb")
        
        opacity_factor = make_tuple(get_input_attribute_value(shader, "base_alpha", "value")[0])
        opacity_texture = create_flipped_texture(None, opacity_factor, "linear")
        alpha_cutoff = get_input_attribute_value(shader, "alpha_cutoff", "value")[0]
        alpha_mode = get_input_attribute_value(shader, "alpha_mode", "value")[0]
        alpha_cutoff = mu.adjust_alpha_cutoff(alpha_cutoff, alpha_mode)
        opacity_texture.apply_cutoff(alpha_cutoff)

        metallic_factor = make_tuple(get_input_attribute_value(shader, "metallic_factor", "value")[0])
        roughness_factor = make_tuple(get_input_attribute_value(shader, "roughness_factor", "value")[0])
        combined_texture_shader, combined_texture_output = get_input_attribute_value(shader, "metallic_roughness_texture", "attribute")
        if combined_texture_shader is not None:
            combined_image = parse_gltf_surface(combined_texture_shader, source_type, combined_texture_output)
            roughness_image = combined_image[:, :, 1]
            metallic_image = combined_image[:, :, 2]
        else:
            roughness_image, metallic_image = None, None
        metallic_texture = create_flipped_texture(metallic_image, metallic_factor, "linear")
        roughness_texture = create_flipped_texture(roughness_image, roughness_factor, "linear")

        emissive_factor = make_tuple(get_input_attribute_value(shader, "emissive_strength", "value")[0])
        emissive_texture = create_flipped_texture(None, emissive_factor, "srgb")

        occlusion_texture_shader, occlusion_texture_output = \
            get_input_attribute_value(shader, "occlusion_texture", "attribute")
        if occlusion_texture_shader is not None:
            occlusion_image = parse_gltf_surface(occlusion_texture_shader, source_type, occlusion_texture_output)

        return {
            "color_texture": color_texture,
            "opacity_texture": opacity_texture,
            "roughness_texture": roughness_texture,
            "metallic_texture": metallic_texture,
            "emissive_texture": emissive_texture,
        }, "st"

    elif shader_subid == "gltf_texture_lookup":
        texture = get_input_attribute_value(shader, "texture", "value")[0]
        texture_image = load_texture(texture)
        return texture_image

    else:
        raise Exception(f"Fail to parse gltf Shader {shader_subid}.")

def parse_omni_surface(shader, source_type, output_name):
    def parse_component(component_name, component_encode, adjust=None):
        component_usetex = get_input_attribute_value(shader, f"Is{component_name}Tex", "value")[0] == 1
        if component_usetex:
            component_tex_name = f"{component_name}_Tex"
            component_image = get_input_attribute_value(shader, component_tex_name, "value")[0]
            component_image = load_texture(component_image)
            component_cs = shader.GetInput(component_tex_name).GetAttr().GetColorSpace()
            component_overencode = cs_encode[component_cs]
            if component_overencode is not None:
                component_encode = component_overencode
            if adjust is not None:
                component_image = adjust(component_image)
            component_factor = None
        else:
            component_color_name = f"{component_name}_Color"
            component_factor = get_input_attribute_value(shader, component_color_name, "value")[0]
            if adjust is not None and component_factor is not None:
                component_factor = tuple([adjust(c) for c in component_factor])
            component_image = None

        component_texture = create_flipped_texture(component_image, component_factor, component_encode)
        return component_texture

    color_texture = parse_component("BaseColor", "srgb")
    opacity_texture = color_texture.check_dim(3) if color_texture else None
    emissive_intensity = get_input_attribute_value(shader, "EmissiveIntensity", "value")[0]
    emissive_texture = parse_component("Emissive", "srgb", lambda x: x * emissive_intensity) if emissive_intensity else None
    metallic_texture = parse_component("Metallic", "linear")
    normal_texture = parse_component("Normal", "linear")
    roughness_texture = parse_component("Gloss", "linear", lambda x: (2 / (x + 2 + 1e-6))**(1.0 / 4.0))
    return {
        "color_texture": color_texture,
        "opacity_texture": opacity_texture,
        "roughness_texture": roughness_texture,
        "metallic_texture": metallic_texture,
        "emissive_texture": emissive_texture,
        "normal_texture": normal_texture,
    }, "st"


def _read_mdl_file(mdl_path: str) -> str:
    """Read an MDL file and return its text content (UTF‑8 BOM tolerated)."""
    with open(mdl_path, "r", encoding="utf‑8‑sig") as f:
        return f.read()


def _extract_texture_and_color(mdl_text: str, comp: str):
    """
    Helper that finds either <comp>_Tex (texture_2d) or <comp>_Color (float4)
    and returns (image_path | None, color_tuple | None, encode).
    """
    tex_pat = re.compile(
        rf'uniform\s+texture_2d\s+{comp}_Tex\s*=\s*texture_2d\("([^"]+)"\s*,\s*::tex::gamma_(srgb|linear)\)',
        re.IGNORECASE,
    )
    flag_pat = re.compile(rf'float\s+Is{comp}Tex\s*=\s*([0-9.+-eE]+)')
    col_pat = re.compile(
        rf'float4\s+{comp}_Color\s*=\s*float4\(([0-9.,\s+-eE]+)\)',
        re.IGNORECASE,
    )

    tex_m = tex_pat.search(mdl_text)
    flag_m = flag_pat.search(mdl_text)
    col_m = col_pat.search(mdl_text)

    use_tex = tex_m and (not flag_m or float(flag_m.group(1)) > 0.5)

    if use_tex:
        img_path = tex_m.group(1).lstrip("./")  # keep relative, fix below
        encode = tex_m.group(2).lower()
        return img_path, None, encode
    else:
        if col_m:
            vals = [float(v) for v in col_m.group(1).split(",")]
            if len(vals) == 3:  # append alpha if missing
                vals.append(1.0)
            return None, tuple(vals), "srgb" if comp in ["BaseColor", "Emissive"] else "linear"
        return None, None, None

def parse_mdl_surface(shader, surface_output_name="surface"):
    """
    Convert an MDL shader prim into PBR texture dictionary like parse_omni_surface().
    """
    def _parse_slot(slot: str, encode_default: str, adjust_fn=None):
        """Read Color/Texture pair indicated by Is<slot>Tex flag."""
        use_tex_val = get_input_attribute_value(shader, f"Is{slot}Tex", "value")
        use_tex = use_tex_val[0] == 1 if use_tex_val else False
        if use_tex:
            tex_path = get_input_attribute_value(shader, f"{slot}_Tex", "value")[0]
            tex_img  = load_texture(tex_path)
            # MDL stores gamma info in attribute's ColorSpace
            cs       = shader.GetInput(f"{slot}_Tex").GetAttr().GetColorSpace()
            encode   = cs_encode.get(cs, encode_default)
            if adjust_fn:
                tex_img = adjust_fn(tex_img)
            factor   = None
        else:
            factor   = get_input_attribute_value(shader, f"{slot}_Color", "value")[0]
            if adjust_fn and factor is not None:
                factor = tuple(adjust_fn(c) for c in factor)
            tex_img  = None
            encode   = encode_default
        return create_flipped_texture(tex_img, factor, encode)

    # ----- BaseColor / Metallic / etc. -----
    color_tex      = _parse_slot("BaseColor", "srgb")
    opacity_tex    = color_tex.check_dim(3) if color_tex else None
    val = get_input_attribute_value(shader, "EmissiveIntensity", "value")
    emissive_int = val[0] if val else 0.0  
    emissive_tex   = _parse_slot("Emissive", "srgb",
                                 (lambda x: x * emissive_int) if emissive_int else None)
    metal_tex      = _parse_slot("Metallic", "linear")
    # Gloss → Roughness
    gloss_to_rough = lambda x: (2.0 / (x + 2.0 + 1e-6)) ** 0.25
    rough_tex      = _parse_slot("Gloss", "linear", gloss_to_rough)
    # Normal map stays in tangent space, linear
    normal_tex     = _parse_slot("Normal", "linear")

    return {
        "color_texture":     color_tex,
        "opacity_texture":   opacity_tex,
        "roughness_texture": rough_tex,
        "metallic_texture":  metal_tex,
        "emissive_texture":  emissive_tex,
        "normal_texture":    normal_tex,
    }, "st"


def parse_usd_material(material):
    surface_outputs = material.GetSurfaceOutputs()
    for surface_output in surface_outputs:
        if not surface_output.HasConnectedSource():
            continue
        surface_output_connectable, surface_output_name, _ = surface_output.GetConnectedSource()
        surface_shader = UsdShade.Shader(surface_output_connectable.GetPrim())
        surface_shader_implement = surface_shader.GetImplementationSource()

        if surface_shader_implement == "id":
            if surface_shader.GetShaderId() == "UsdPreviewSurface":
                return parse_preview_surface(surface_shader, surface_output_name)
            gs.logger.warning(f"Fail to parse Shader {surface_shader.GetPath()} with ID {surface_shader.GetShaderId()}.")
            continue

        elif surface_shader_implement == "sourceAsset":
            source_types = surface_shader.GetSourceTypes()
            for source_type in source_types:
                source_asset = surface_shader.GetSourceAsset(source_type).resolvedPath
                if source_asset.lower().endswith(".mdl"):
                    base_name = os.path.basename(source_asset)

                    if "gltf/pbr" in source_asset:
                        return parse_gltf_surface(surface_shader, source_type, surface_output_name)
                    try:
                        return parse_omni_surface(surface_shader, source_type, surface_output_name)
                    except Exception as e:
                        try:
                            return parse_mdl_surface(surface_shader)
                        except Exception as e:
                            gs.logger.warning(f"Fail to parse Shader {surface_shader.GetPath()} of asset {source_asset} with message: {e}.")

    
    return None, None

def parse_mesh_usd(path, group_by_material, scale, surface):
    stage = Usd.Stage.Open(path)
    xform_cache = UsdGeom.XformCache()
    usd_traverse = stage.Traverse()

    mesh_infos = mu.MeshInfoGroup()
    materials = dict()
    uv_names = dict()

    for prim in stage.Traverse():
        if prim.HasRelationship("material:binding"):
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(prim)

    for i, prim in enumerate(usd_traverse):
        if prim.IsA(UsdGeom.Mesh):
            path_str = str(prim.GetPath())
            avoid = ["__default_setting", "person"]
            if any(key in path_str for key in avoid):
                continue
            matrix = np.array(xform_cache.GetLocalToWorldTransform(prim))
            usd_mesh = UsdGeom.Mesh(prim)
            mesh_path = prim.GetPath().pathString

            if not usd_mesh.GetPointsAttr().HasValue():
                continue
            points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            faces_vertex_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
            points_faces_varying = False

            # parse normals
            normals = None
            normal_attr = usd_mesh.GetNormalsAttr()
            if normal_attr.HasValue():
                normals = np.array(normal_attr.Get(), dtype=np.float32)
                if normals.shape[0] != points.shape[0]:
                    if normals.shape[0] == faces.shape[0]:        # face varying
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of normals mismatch for mesh {mesh_path} in usd file {path}.")
            
            # parse materials
            prim_bindings = UsdShade.MaterialBindingAPI(prim)
            material = prim_bindings.ComputeBoundMaterial()[0]
            group_idx = ""
            if material.GetPrim().IsValid():
                material_spec = material.GetPrim().GetPrimStack()[-1]
                material_id = material_spec.layer.identifier + material_spec.path.pathString
                if material_id not in materials:
                    material_dict, uv_names[material_id] = parse_usd_material(material)
                    material_surface = surface.copy()
                    if material_dict is not None:
                        material_surface.update_texture(
                            color_texture=material_dict.get("color_texture"),
                            opacity_texture=material_dict.get("opacity_texture"),
                            roughness_texture=material_dict.get("roughness_texture"),
                            metallic_texture=material_dict.get("metallic_texture"),
                            normal_texture=material_dict.get("normal_texture"),
                            emissive_texture=material_dict.get("emissive_texture"),
                            ior=material_dict.get("ior"),
                        )
                    materials[material_id] = material_surface
                group_idx = material_id if group_by_material else i
                uv_name = uv_names[material_id]

            # parse uvs
            
            uv_attr = prim.GetAttribute(f"primvars:{uv_name}")
            uvs = None
            if uv_attr.HasValue():
                uvs = np.array(uv_attr.Get(), dtype=np.float32)
                if uvs.shape[0] != points.shape[0]:
                    if uvs.shape[0] == faces.shape[0]:
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of uvs mismatch for mesh {mesh_path} in usd file {path}.")

            # rearrange points and faces
            if points_faces_varying:
                points = points[faces]
                faces = np.arange(faces.shape[0])
            
            if np.max(faces_vertex_counts) > 3:
                triangles = list()
                bi = 0
                for fi in range(len(faces_vertex_counts)):
                    if faces_vertex_counts[fi] == 3:
                        triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                        bi += 3
                    elif faces_vertex_counts[fi] == 4:
                        triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                        triangles.append([faces[bi + 0], faces[bi + 2], faces[bi + 3]])
                        bi += 4
                triangles = np.array(triangles, dtype=np.int32)
            else:
                triangles = faces.reshape(-1, 3)

            if normals is None:
                normals = trimesh.Trimesh(points, triangles, process=False).vertex_normals
            points, normals = mu.apply_transform(matrix, points, normals)
            if np.linalg.det(matrix[:3, :3]) < 0:
                triangles = triangles[:, [0, 2, 1]]

            # print(
            #     group_idx,
            #     points.shape,
            #     triangles.shape,
            #     normals.shape if normals is not None else None,
            #     uvs.shape if uvs is not None else None
            # )
            mesh_infos.append(group_idx, points, triangles, normals, uvs, materials[material_id])

    # l = mesh_infos.export_meshes(scale=scale)
    # print(len(l), l[0].verts.shape, l[0].faces.shape,
    #     l[0].normals.shape if l[0].normals is not None else None,
    #     l[0].uvs.shape if l[0].uvs is not None else None)
    # with open("usda_geom_ella.txt", "a") as f:
    #     f.write(path + "\n")
    #     for ll in l:
    #         f.write(str(ll.verts.shape) + " ")
    #     f.write("\n")

    return mesh_infos.export_meshes(scale=scale)


def _extract_semantic(prim, extract_from_children=True):
    for attr in prim.GetAuthoredAttributes():
        if attr.GetBaseName() == "semanticData":
            if attr.HasAuthoredValue():
                return attr.Get()

    if extract_from_children:
        for child in prim.GetChildren():
            val = _extract_semantic(child, False)
            if val:
                return val
    return None


def parse_instance_usd(path):
    stage        = Usd.Stage.Open(path)
    xform_cache  = UsdGeom.XformCache()
    instance_lst = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        if len(prim.GetPrimStack()) <= 1:            
            continue
        matrix = np.array(xform_cache.GetLocalToWorldTransform(prim)).T
        instance_spec = prim.GetPrimStack()[-1]
        layer_id      = instance_spec.layer.identifier
        target   = prim.GetPrototype() if prim.IsInstance() else prim  
        semantic = _extract_semantic(target)
        if semantic and "/" in semantic:
            semantic = semantic.split("/", 1)[0]
        typ = 'object'
        if any([st in str(prim.GetPath()) for st in ['__default_setting', 'lights', 'Base']]):
            typ = 'structure'
        instance_lst.append((matrix, layer_id, semantic, typ))

    return instance_lst


def T_decompose(T):
    pos = T[:3, 3]
    euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    scale = np.linalg.norm(T[:3, :3], axis=0)
    return pos, euler, scale

def parse_usd_lights(path, scene):
    """Parse lights from a USD stage and add them to LuisaRender via light_manager."""
    stage = Usd.Stage.Open(path)
    xform_cache  = UsdGeom.XformCache()
    for prim in stage.Traverse():
        typename = prim.GetTypeName()

        if typename in ["RectLight", "SphereLight", "DistantLight"]:
            xformable = UsdGeom.Xformable(prim)
            transform = np.array(xform_cache.GetLocalToWorldTransform(prim)).T
            init_euler = [0, 0, 0]
            init_pos = np.array([0, 0, 0])
            init_T = gs.utils.geom.trans_R_to_T(init_pos, R.from_euler("xyz", init_euler, degrees=True).as_matrix())
            init_T[:3, :3] *= 0.01
            
            final_transform = init_T @ transform
            pos, euler, scale = T_decompose(final_transform)

            attrs = prim.GetAttributes()
            attr_dict = {a.GetName(): a.Get() for a in attrs}

            color = attr_dict.get("inputs:color", (1.0, 1.0, 1.0))
            intensity = attr_dict.get("inputs:intensity", 1.0)
            exposure = attr_dict.get("inputs:exposure", 0.0)
            final_intensity = intensity * (2 ** exposure) 
            if len(color) == 3:
                color = tuple(list(color) + [0.0])
            if typename == "SphereLight":
                radius = attr_dict.get("inputs:radius", 1.0)
                scene.add_sphere_light(
                    radius=radius*0.01,
                    color=color,
                    intensity=final_intensity * 0.0001,
                    pos=pos
                )

            elif typename == "RectLight":
                width = attr_dict.get("inputs:width", 1.0) * 0.01
                height = attr_dict.get("inputs:height", 1.0) * 0.01

                size = (width, height, 0.01) 
                box = gs.morphs.Box(
                    pos=tuple(pos),
                    euler=tuple(euler),
                    size=tuple(size),
                    fixed=True
                )
                scene.add_light(
                    morph=box,
                    color=color,  # expand color from (r,g,b) -> (r,g,b,a)
                    intensity=final_intensity * 0.0001,
                    revert_dir=True,      # RectLight generally emits outward
                    double_sided=True,    # Usually double-sided emission
                    beam_angle=180.0,
                )

            elif typename == "DistantLight":
                radius = 1.0  
                distance = 500.0  
                direction = np.array([1.0, 1.0, 1.0])
                direction /= np.linalg.norm(direction)  # Normalize direction
                sphere_pos =  direction * distance

                scene.add_sphere_light(
                    radius=radius,  
                    color=color,
                    intensity=final_intensity * distance**2,
                    pos=sphere_pos,
                )

            else:
                gs.logger.warning(f"Skipping unknown light type: {typename}")

def quat_to_direction(quat):
    """Convert a quaternion to a forward direction vector."""
    r = R.from_quat(quat)
    return r.apply([0, 0, -1])  # forward direction

if __name__ == "__main__":
    file_path = 'table_scene.usd'
    grouped_meshes = parse_mesh_usd(file_path)
    for mesh in grouped_meshes:
        print(mesh)