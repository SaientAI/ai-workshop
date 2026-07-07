"""Blender-side PNG-to-GLB exporter.

Run only through Blender:

  blender --background --factory-startup --python blender_png_to_glb.py -- --input in.png --output out.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.geometry import tessellate_polygon


def parse_args() -> argparse.Namespace:
    import sys

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="asset")
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--thickness", type=float, default=0.16)
    parser.add_argument("--mesh-json")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_image_material(image_path: Path, name: str, alpha_mode: str = "blend") -> bpy.types.Material:
    image = bpy.data.images.load(str(image_path), check_existing=True)
    mat = bpy.data.materials.new(f"{name}_transparent")
    mat.use_nodes = True
    mat.blend_method = "CLIP" if alpha_mode == "clip" else "BLEND"
    mat.use_screen_refraction = False
    mat.use_backface_culling = alpha_mode == "clip"
    mat.show_transparent_back = alpha_mode != "clip"
    mat.alpha_threshold = 0.08

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = image
    tex.extension = "CLIP"

    links = mat.node_tree.links
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.75
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def make_flat_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(f"{name}_backer")
    mat.diffuse_color = (0.08, 0.06, 0.04, 1.0)
    return mat


def make_side_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(f"{name}_silhouette_edge")
    mat.diffuse_color = (0.035, 0.032, 0.03, 1.0)
    mat.use_nodes = True
    mat.blend_method = "OPAQUE"
    mat.show_transparent_back = False
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.035, 0.032, 0.03, 1.0)
    bsdf.inputs["Alpha"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.82
    return mat


def parent_scene_to_root(name: str) -> None:
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.object
    root.name = f"{name}_root"
    for scene_obj in bpy.context.scene.objects:
        if scene_obj is not root:
            scene_obj.parent = root


def create_silhouette_mesh(image_path: Path, mesh_json: Path, name: str, thickness: float) -> None:
    payload = json.loads(mesh_json.read_text())
    if payload.get("mesh_kind") == "relief":
        create_relief_mesh(image_path, payload, name)
        return

    points = [(float(x), float(z)) for x, z in payload.get("vertices", [])]
    uvs = [(float(u), float(v)) for u, v in payload.get("uvs", [])]
    if len(points) < 3 or len(points) != len(uvs):
        raise ValueError("silhouette mesh payload does not contain a valid outline")

    depth = float(payload.get("depth", thickness))
    if depth <= 0.0:
        depth = max(thickness, 0.01)

    outline = [Vector((x, z, 0.0)) for x, z in points]
    triangles = tessellate_polygon([outline])
    if not triangles:
        raise ValueError("silhouette triangulation failed")

    front_y = -depth * 0.5
    back_y = depth * 0.5
    verts = [(x, front_y, z) for x, z in points] + [(x, back_y, z) for x, z in points]
    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []
    count = len(points)

    for tri in triangles:
        faces.append(tuple(int(i) for i in tri))
        material_indices.append(0)
    for tri in triangles:
        faces.append(tuple(count + int(i) for i in reversed(tri)))
        material_indices.append(0)
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, count + j, count + i))
        material_indices.append(1)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_image_material(image_path, name))
    obj.data.materials.append(make_side_material(name))
    for polygon, material_index in zip(obj.data.polygons, material_indices):
        polygon.material_index = material_index

    uv_layer = obj.data.uv_layers.new(name="UVMap")
    for polygon in obj.data.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = obj.data.loops[loop_index].vertex_index % count
            uv_layer.data[loop_index].uv = uvs[vertex_index]

    parent_scene_to_root(name)


def create_relief_mesh(image_path: Path, payload: dict, name: str) -> None:
    vertices = [tuple(float(v) for v in vertex) for vertex in payload.get("vertices", [])]
    uvs = [tuple(float(v) for v in uv) for uv in payload.get("uvs", [])]
    faces = [tuple(int(index) for index in face) for face in payload.get("faces", [])]
    material_indices = [int(index) for index in payload.get("material_indices", [])]
    if not vertices or not faces or len(vertices) != len(uvs):
        raise ValueError("relief mesh payload does not contain valid geometry")

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_image_material(image_path, name, alpha_mode="clip"))
    obj.data.materials.append(make_side_material(name))

    for polygon_index, polygon in enumerate(obj.data.polygons):
        material_index = material_indices[polygon_index] if polygon_index < len(material_indices) else 0
        polygon.material_index = min(max(material_index, 0), 1)
        polygon.use_smooth = material_index == 0

    uv_layer = obj.data.uv_layers.new(name="UVMap")
    for polygon in obj.data.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = obj.data.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uvs[vertex_index]

    parent_scene_to_root(name)


def create_billboard(image_path: Path, name: str, width: float, height: float, thickness: float) -> None:
    image = bpy.data.images.load(str(image_path), check_existing=True)
    aspect = image.size[1] / max(image.size[0], 1)
    real_height = height * aspect
    half_w = width * 0.5

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    verts = [
        (-half_w, 0.0, 0.0),
        (half_w, 0.0, 0.0),
        (half_w, 0.0, real_height),
        (-half_w, 0.0, real_height),
    ]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_image_material(image_path, name))
    obj.data.uv_layers.new(name="UVMap")
    uv_data = obj.data.uv_layers.active.data
    for uv, value in zip(uv_data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.uv = value

    if thickness > 0.0:
        back_mesh = bpy.data.meshes.new(f"{name}_backer_mesh")
        inset_w = width * 0.92
        inset_h = real_height * 0.92
        hw = inset_w * 0.5
        z0 = real_height * 0.04
        z1 = z0 + inset_h
        y = thickness
        back_mesh.from_pydata(
            [(-hw, y, z0), (hw, y, z0), (hw, y, z1), (-hw, y, z1)],
            [],
            [(0, 1, 2, 3)],
        )
        back_mesh.update()
        back = bpy.data.objects.new(f"{name}_selection_backer", back_mesh)
        bpy.context.collection.objects.link(back)
        back.data.materials.append(make_flat_material(name))
        back.hide_render = False

    parent_scene_to_root(name)


def export_glb(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        export_apply=True,
        export_yup=True,
        use_selection=False,
    )


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Input PNG does not exist: {source}")

    clear_scene()
    mesh_json = Path(args.mesh_json).expanduser().resolve() if args.mesh_json else None
    if mesh_json and mesh_json.exists():
        try:
            create_silhouette_mesh(source, mesh_json, args.name, args.thickness)
        except Exception as exc:
            print(f"warning: silhouette mesh failed ({exc}); using billboard geometry")
            create_billboard(source, args.name, args.width, args.height, args.thickness)
    else:
        create_billboard(source, args.name, args.width, args.height, args.thickness)
    export_glb(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
