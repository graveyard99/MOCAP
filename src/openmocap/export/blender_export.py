"""Standalone headless Blender backend. Run only through export_animation.

Requires only Blender's bundled NumPy and bpy; scipy/project imports are absent.
The exported character has one mesh, one armature, persistent skinning, and
rotational animation. Pose correctives are relative shape keys, not mesh frames.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector
import numpy as np


def rotation_matrix(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return Matrix.Identity(3)
    return Quaternion(Vector((vector / angle).tolist()), angle).to_matrix()


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    source, destination, file_format, fps_string, report_path, validate_string, axis_json = args
    fps, validate = float(fps_string), bool(int(validate_string))
    # Supplied by the shared geometry.coordinates registry, keeping this
    # standalone Blender process free of duplicated axis-conversion logic.
    world_to_blender = np.asarray(json.loads(axis_json), dtype=float)
    with np.load(source, allow_pickle=False) as source_data:
        data = {name: source_data[name] for name in source_data.files}
    names, parents = data["names"].tolist(), data["parents"].astype(int)
    root = int(np.flatnonzero(parents == -1)[0])
    order = [root]
    for index in order:
        order.extend(int(child) for child in np.flatnonzero(parents == index))
    axis = Matrix(world_to_blender.tolist())
    rest = data["rest_joints"] @ world_to_blender.T
    vertices = data["vertices"] @ world_to_blender.T
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps) / fps
    scene.frame_start, scene.frame_end = 1, len(data["times"])
    armature_data = bpy.data.armatures.new("ActorSkeleton")
    armature = bpy.data.objects.new("ActorRig", armature_data)
    scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for joint in order:
        bone = armature_data.edit_bones.new(names[joint])
        bone.head = rest[joint].tolist()
        children = np.flatnonzero(parents == joint)
        direction = rest[children[0]] - rest[joint] if len(children) else np.array([0.0, 0.0, 0.05])
        if np.linalg.norm(direction) < 1e-5:
            direction = np.array([0.0, 0.0, 0.05])
        bone.tail = (rest[joint] + direction).tolist()
        if parents[joint] >= 0:
            bone.parent = armature_data.edit_bones[names[parents[joint]]]
        bone.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh_data = bpy.data.meshes.new("ActorBodyTopology")
    mesh_data.from_pydata(vertices.tolist(), [], data["faces"].tolist())
    mesh_data.update()
    mesh = bpy.data.objects.new("ActorBody", mesh_data)
    scene.collection.objects.link(mesh)
    mesh.parent = armature
    mesh["body_model"] = str(data["model_name"])
    mesh["persistent_shape"] = True
    for joint, name in enumerate(names):
        group = mesh.vertex_groups.new(name=name)
        for vertex in np.flatnonzero(data["weights"][:, joint] > 1e-8):
            group.add([int(vertex)], float(data["weights"][vertex, joint]), "REPLACE")
    modifier = mesh.modifiers.new("PersistentSkinning", "ARMATURE")
    modifier.object = armature
    # Every pose-corrective basis remains tied to the same topology and skin.
    # SMPL coefficients = vectorized (R_nonroot - identity).
    corrective_keys = []
    if "posedirs" in data:
        mesh.shape_key_add(name="Basis")
        dirs = data["posedirs"].reshape(len(vertices), 3, -1)
        for feature in range(dirs.shape[2]):
            key = mesh.shape_key_add(name=f"pose_corrective_{feature:03d}")
            key.slider_min, key.slider_max = -2.0, 2.0
            coords = vertices + dirs[:, :, feature] @ world_to_blender.T
            key.data.foreach_set("co", coords.reshape(-1))
            corrective_keys.append(key)
    material = bpy.data.materials.new("ActorNeutral")
    material.diffuse_color = (0.18, 0.43, 0.62, 1)
    mesh.data.materials.append(material)
    for frame in range(len(data["times"])):
        scene.frame_set(frame + 1)
        global_rotations = {}
        world_positions = {}
        local_matrices = [rotation_matrix(value) for value in data["rotations"][frame]]
        for joint in order:
            parent = int(parents[joint])
            local = axis @ local_matrices[joint] @ axis.transposed()
            if parent == -1:
                global_rotations[joint] = local
                world_positions[joint] = axis @ Vector(data["translations"][frame].tolist())
            else:
                global_rotations[joint] = global_rotations[parent] @ local
                offset = Vector((rest[joint] - rest[parent]).tolist())
                world_positions[joint] = world_positions[parent] + global_rotations[parent] @ offset
            bone = armature.pose.bones[names[joint]]
            bone.rotation_mode = "QUATERNION"
            orientation = (
                global_rotations[joint] @ armature.data.bones[names[joint]].matrix_local.to_3x3()
            )
            desired = orientation.to_4x4()
            desired.translation = world_positions[joint]
            bone.matrix = desired
            bpy.context.view_layer.update()
            bone.keyframe_insert(
                data_path="rotation_quaternion", frame=frame + 1, group=names[joint]
            )
            bone.keyframe_insert(data_path="location", frame=frame + 1, group=names[joint])
        if corrective_keys:
            feature_vector = np.array(
                [
                    np.array(matrix) - np.eye(3)
                    for index, matrix in enumerate(local_matrices)
                    if index != root
                ]
            ).reshape(-1)
            for key, value in zip(corrective_keys, feature_vector, strict=True):
                key.value = float(value)
                key.keyframe_insert(data_path="value", frame=frame + 1)
    if armature.animation_data and armature.animation_data.action:
        for curve in armature.animation_data.action.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    def expected_mesh(frame):
        local = [np.array(rotation_matrix(value)) for value in data["rotations"][frame]]
        global_rot = np.zeros((len(names), 3, 3))
        for joint in order:
            transformed = world_to_blender @ local[joint] @ world_to_blender.T
            parent = int(parents[joint])
            global_rot[joint] = transformed if parent == -1 else global_rot[parent] @ transformed
        local_vertices = vertices.copy()
        if "posedirs" in data:
            feature = np.array(
                [matrix - np.eye(3) for index, matrix in enumerate(local) if index != root]
            ).reshape(-1)
            local_vertices += (data["posedirs"] @ feature).reshape(-1, 3) @ world_to_blender.T
        transforms = np.zeros((len(names), 4, 4))
        transforms[:, :3, :3] = global_rot
        world_joints = data["joints"][frame] @ world_to_blender.T
        transforms[:, :3, 3] = world_joints - np.einsum("jab,jb->ja", global_rot, rest)
        transforms[:, 3, 3] = 1
        homogeneous = np.column_stack([local_vertices, np.ones(len(vertices))])
        return np.einsum("vj,jab,vb->va", data["weights"], transforms, homogeneous, optimize=True)[
            :, :3
        ]

    def mesh_error(item, expected):
        evaluated = item.evaluated_get(bpy.context.evaluated_depsgraph_get())
        actual = np.array(
            [list(evaluated.matrix_world @ vertex.co) for vertex in evaluated.data.vertices]
        )
        if actual.shape != expected.shape:
            raise RuntimeError("Export changed the persistent mesh vertex count")
        return float(np.max(np.linalg.norm(actual - expected, axis=1)))

    # Sample native rig before interchange and verify the FK contract.
    native_errors, native_mesh_errors = [], []
    for frame in [0, len(data["times"]) - 1]:
        scene.frame_set(frame + 1)
        bpy.context.view_layer.update()
        for joint, name in enumerate(names):
            expected = axis @ Vector(data["joints"][frame, joint].tolist())
            native_errors.append((armature.pose.bones[name].head - expected).length)
        native_mesh_errors.append(mesh_error(mesh, expected_mesh(frame)))
    if max(native_errors) > 1e-4:
        raise RuntimeError(f"Native rig FK mismatches metric animation: {max(native_errors)} m")
    if max(native_mesh_errors) > 2e-4:
        raise RuntimeError(f"Native skinning mismatches numerical LBS: {max(native_mesh_errors)} m")
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    scene.frame_set(1)
    if file_format == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=destination,
            use_selection=True,
            object_types={"ARMATURE", "MESH"},
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            bake_anim_simplify_factor=0,
            axis_forward="-Z",
            axis_up="Y",
            apply_unit_scale=True,
            apply_scale_options="FBX_SCALE_UNITS",
            global_scale=1.0,
            use_mesh_modifiers=False,
        )
    else:
        bpy.ops.wm.usd_export(
            filepath=destination,
            selected_objects_only=True,
            export_animation=True,
            export_armatures=True,
        )
    report = {
        "validated": False,
        "skeleton": True,
        "mesh": True,
        "skinning": True,
        "animation": True,
        "native_fk_max_error_m": max(native_errors),
        "native_skinning_max_error_m": max(native_mesh_errors),
        "units": "metres",
        "export_axes": "Y up, -Z forward",
        "pose_correctives": bool(corrective_keys),
    }
    if validate and file_format == "fbx":
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.import_scene.fbx(filepath=destination, use_anim=True, anim_offset=0.0)
        rigs = [item for item in scene.objects if item.type == "ARMATURE"]
        meshes = [item for item in scene.objects if item.type == "MESH"]
        if not rigs or not meshes:
            raise RuntimeError("FBX lacks mesh or skeleton after clean re-import")
        rig = rigs[0]
        body = meshes[0]
        skinned = any(mod.type == "ARMATURE" and mod.object == rig for mod in body.modifiers)
        weighted = all(len(vertex.groups) > 0 for vertex in body.data.vertices)
        action = rig.animation_data.action if rig.animation_data else None
        if action is None:
            raise RuntimeError("FBX lacks skeletal animation")
        if corrective_keys:
            imported_keys = body.data.shape_keys
            if imported_keys is None or len(imported_keys.key_blocks) != len(corrective_keys) + 1:
                raise RuntimeError("FBX lost persistent body pose-corrective shape keys")
            if imported_keys.animation_data is None or imported_keys.animation_data.action is None:
                raise RuntimeError("FBX lost animated pose-corrective coefficients")
        frame_range = list(action.frame_range)
        root_positions, reimport_errors, reimport_mesh_errors = [], [], []
        for frame in [0, len(data["times"]) - 1]:
            scene.frame_set(frame + 1)
            bpy.context.view_layer.update()
            root_positions.append(list(rig.matrix_world @ rig.pose.bones[names[root]].head))
            for joint, name in enumerate(names):
                expected = axis @ Vector(data["joints"][frame, joint].tolist())
                actual = rig.matrix_world @ rig.pose.bones[name].head
                reimport_errors.append((actual - expected).length)
            reimport_mesh_errors.append(mesh_error(body, expected_mesh(frame)))
        correct_range = (
            abs(frame_range[0] - 1) < 1e-3 and abs(frame_range[1] - len(data["times"])) < 1e-3
        )
        if (
            not skinned
            or not weighted
            or not correct_range
            or max(reimport_errors) > 2e-4
            or max(reimport_mesh_errors) > 5e-4
        ):
            raise RuntimeError(
                f"FBX validation failed: skin={skinned}, weights={weighted}, range={frame_range}, metric_error={max(reimport_errors)}"
            )
        report.update(
            validated=True,
            validation="clean Blender FBX re-import",
            bone_count=len(rig.data.bones),
            vertex_count=len(body.data.vertices),
            weighted_vertices=len(body.data.vertices),
            frame_range=frame_range,
            root_positions=root_positions,
            reimport_joint_max_error_m=max(reimport_errors),
            skinning=skinned,
        )
        report["reimport_skinning_max_error_m"] = max(reimport_mesh_errors)
    elif validate:
        # USD backend support varies by Blender build. Authoring succeeded but
        # no skeletal re-import contract is claimed until explicitly tested.
        report.update(
            validation="USD authoring only; skeletal roundtrip unavailable", validated=False
        )
    Path(report_path).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
