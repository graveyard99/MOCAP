from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from openmocap.body_models.model import (
    SMPL_NAMES,
    forward_kinematics,
    load_body_model,
    skin_vertices,
    tree_order,
)
from openmocap.types import Camera


SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
H36M_NAMES = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]
H36M_PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])


@dataclass
class Animation:
    """A single persistent shaped rig, with local joint axis-angle animation.

    vertices are the fitted rest mesh, not unrelated per-frame meshes. joints
    are the actual FK result. Root translations are metric root positions.
    """

    times: NDArray[np.float64]
    joints: NDArray[np.float64]
    rest_joints: NDArray[np.float64]
    parents: NDArray[np.int64]
    names: list[str]
    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    weights: NDArray[np.float64]
    rotations: NDArray[np.float64]
    translations: NDArray[np.float64]
    shape: NDArray[np.float64]
    model_name: str = "fixture"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    posedirs: NDArray[np.float64] | None = None
    confidence: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        frames, joints = len(self.times), len(self.parents)
        tree_order(self.parents)
        if self.joints.shape != (frames, joints, 3):
            raise ValueError("Animation joints must be [frames,joints,3]")
        if self.rotations.shape != (frames, joints, 3) or self.translations.shape != (frames, 3):
            raise ValueError("Animation rotation/translation dimensions are invalid")
        if self.weights.shape != (len(self.vertices), joints):
            raise ValueError("Animation skin weights differ from rig dimensions")
        if len(self.names) != joints or len(set(self.names)) != joints:
            raise ValueError("Rig requires a distinct name for each joint")
        if self.confidence is not None and self.confidence.shape != (frames, joints):
            raise ValueError("Animation confidence must be [frames,joints]")
        if frames < 1 or (frames > 1 and np.any(np.diff(self.times) <= 0)):
            raise ValueError("Animation times must be strictly increasing")
        for value in [
            self.joints,
            self.rest_joints,
            self.vertices,
            self.rotations,
            self.translations,
        ]:
            if not np.isfinite(value).all():
                raise ValueError("Animation contains non-finite rig data")

    def mesh_at(self, frame: int) -> NDArray[np.float64]:
        vertices = self.vertices
        if self.posedirs is not None:
            matrices = Rotation.from_rotvec(self.rotations[frame]).as_matrix()
            root = int(np.flatnonzero(self.parents == -1)[0])
            feature = (np.delete(matrices, root, axis=0) - np.eye(3)).reshape(-1)
            vertices = vertices + (self.posedirs @ feature).reshape(-1, 3)
        return skin_vertices(
            vertices,
            self.rest_joints,
            self.parents,
            self.weights,
            self.rotations[frame],
            self.translations[frame],
        )[0]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            name: getattr(self, name)
            for name in [
                "times",
                "joints",
                "rest_joints",
                "parents",
                "vertices",
                "faces",
                "weights",
                "rotations",
                "translations",
                "shape",
            ]
        }
        arrays.update(
            names=np.array(self.names),
            model_name=np.array(self.model_name),
            diagnostics=np.array(json.dumps(self.diagnostics)),
        )
        if self.posedirs is not None:
            arrays["posedirs"] = self.posedirs
        if self.confidence is not None:
            arrays["confidence"] = self.confidence
        with output.open("wb") as handle:
            np.savez_compressed(handle, **arrays)


def load_animation(path: str | Path) -> Animation:
    with np.load(path, allow_pickle=False) as data:
        fields = {
            name: data[name]
            for name in [
                "times",
                "joints",
                "rest_joints",
                "parents",
                "vertices",
                "faces",
                "weights",
                "rotations",
                "translations",
                "shape",
            ]
        }
        return Animation(
            **fields,
            names=data["names"].tolist(),
            model_name=str(data["model_name"]),
            diagnostics=json.loads(str(data["diagnostics"])),
            posedirs=data["posedirs"] if "posedirs" in data else None,
            confidence=data["confidence"] if "confidence" in data else None,
        )


def _fill_missing(points: NDArray[np.float64], times: NDArray[np.float64]) -> NDArray[np.float64]:
    result = points.copy()
    for joint in range(points.shape[1]):
        for axis in range(3):
            valid = np.isfinite(points[:, joint, axis])
            if not np.any(valid):
                raise ValueError(f"Joint {joint} has no measurements or prior at any time")
            result[:, joint, axis] = np.interp(times, times[valid], points[valid, joint, axis])
    return result


def _persistent_rest(
    points: NDArray[np.float64], confidence: NDArray[np.float64], parents: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One robust session bone-length estimate, never frame-dependent shape."""
    rest = np.zeros((len(parents), 3))
    lengths = np.zeros(len(parents))
    for joint in tree_order(parents):
        parent = int(parents[joint])
        if parent == -1:
            continue
        delta = points[:, joint] - points[:, parent]
        values = np.linalg.norm(delta, axis=1)
        valid = (confidence[:, joint] > 0.3) & (confidence[:, parent] > 0.3) & (values > 1e-5)
        if not valid.any():
            valid = values > 1e-5
        if not valid.any():
            raise ValueError(f"Bone {parent}->{joint} has zero length throughout take")
        lengths[joint] = float(np.median(values[valid]))
        direction = delta[np.flatnonzero(valid)[0]]
        direction = direction / np.linalg.norm(direction)
        rest[joint] = rest[parent] + lengths[joint] * direction
    return rest, lengths


def _fixture_mesh(
    rest: NDArray[np.float64], parents: NDArray[np.int64], radius: float = 0.035
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Procedural technical mannequin, with connected animation via LBS.

    Cylinder surfaces are intentionally recognisable as a fixture. They make
    skinning/export fully testable without distributing licensed human assets.
    """
    vertices, faces, weights = [], [], []
    sides, rings = 8, 4
    for joint in tree_order(parents):
        parent = int(parents[joint])
        if parent == -1:
            continue
        delta = rest[joint] - rest[parent]
        direction = delta / np.linalg.norm(delta)
        helper = np.array([0.0, 1.0, 0.0])
        if abs(direction @ helper) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        first = np.cross(direction, helper)
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        start = len(vertices)
        for ring in range(rings):
            fraction = ring / (rings - 1)
            for side in range(sides):
                angle = 2 * np.pi * side / sides
                vertex = (
                    rest[parent]
                    + fraction * delta
                    + radius * (np.cos(angle) * first + np.sin(angle) * second)
                )
                vertices.append(vertex)
                weight = np.zeros(len(parents))
                # Most of a segment follows its driving parent. Blend near the
                # distal endpoint to expose proper multi-weight DCC skinning.
                weight[parent] = 1 - 0.2 * fraction
                weight[joint] = 0.2 * fraction
                weights.append(weight)
        for ring in range(rings - 1):
            for side in range(sides):
                a = start + ring * sides + side
                b = start + ring * sides + (side + 1) % sides
                c, d = a + sides, b + sides
                faces.extend([[a, b, c], [b, d, c]])
        for side in range(1, sides - 1):
            faces.append([start, start + side + 1, start + side])
            last = start + (rings - 1) * sides
            faces.append([last, last + side, last + side + 1])
    return np.array(vertices), np.array(faces, dtype=np.int64), np.array(weights)


def _alignment(
    rest_vectors: NDArray[np.float64], actual_vectors: NDArray[np.float64]
) -> NDArray[np.float64]:
    if len(rest_vectors) == 0:
        return np.eye(3)
    a = rest_vectors / np.linalg.norm(rest_vectors, axis=1)[:, None]
    b = actual_vectors / np.maximum(np.linalg.norm(actual_vectors, axis=1)[:, None], 1e-9)
    if len(a) == 1 or np.linalg.matrix_rank(a) < 2:
        cross = np.cross(a[0], b[0])
        sine, cosine = np.linalg.norm(cross), np.clip(a[0] @ b[0], -1, 1)
        if sine < 1e-9:
            if cosine > 0:
                return np.eye(3)
            axis = np.cross(a[0], np.eye(3)[np.argmin(abs(a[0]))])
            return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
        return Rotation.from_rotvec(np.arctan2(sine, cosine) * cross / sine).as_matrix()
    left, _, right = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(right.T @ left.T)
    return right.T @ correction @ left.T


def _initialize_pose(
    rest: NDArray[np.float64], parents: NDArray[np.int64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    global_rot = np.repeat(np.eye(3)[None], len(parents), axis=0)
    local = np.zeros((len(parents), 3))
    for joint in tree_order(parents):
        children = np.flatnonzero(parents == joint)
        parent = int(parents[joint])
        if len(children):
            global_rot[joint] = _alignment(
                rest[children] - rest[joint], points[children] - points[joint]
            )
        elif parent >= 0:
            global_rot[joint] = global_rot[parent]
        matrix = global_rot[joint] if parent == -1 else global_rot[parent].T @ global_rot[joint]
        local[joint] = Rotation.from_matrix(matrix).as_rotvec()
    return local


def fit_actor(
    times: NDArray[np.float64],
    joints: NDArray[np.float64],
    confidence: NDArray[np.float64] | None = None,
    *,
    model: str = "fixture",
    model_path: str | Path | None = None,
    joint_names: list[str] | None = None,
    parents: NDArray[np.int64] | list[int] | None = None,
    prior_joints: NDArray[np.float64] | None = None,
    prior_confidence: float = 0.05,
    trust_pickle: bool = False,
    max_nfev: int = 35,
    cameras: list[Camera] | None = None,
    observations2d: NDArray[np.float64] | None = None,
    confidence2d: NDArray[np.float64] | None = None,
    reprojection_weight: float = 0.001,
    strong_evidence_max_displacement_m: float | None = 0.03,
    strong_evidence_confidence: float = 0.85,
    reprojection_outlier_px: float = 12.0,
    num_betas: int = 10,
    reference_animation: Animation | None = None,
) -> Animation:
    """Calibrate one shape and fit independent pose samples to measured geometry.

    A weak prior is admitted only for low-confidence/missing observations. No
    camera parameters are optimized by this API: optional cameras are consumed
    only for projection. Strong 3D evidence dominates numerical regularization.
    Input/output units are metres with world +Y up.
    """
    times, points = np.asarray(times, float), np.asarray(joints, float).copy()
    if points.ndim != 3 or points.shape[2] != 3 or len(times) != len(points):
        raise ValueError("Expected timestamped joints [frames,joints,3]")
    if joint_names is not None and (
        len(joint_names) != points.shape[1] or len(set(joint_names)) != len(joint_names)
    ):
        raise ValueError("Distinct observation joint_names must match the input joint slots")
    if max_nfev < 1:
        raise ValueError("Pose maximum evaluations must be positive")
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Output times must be finite and strictly increasing")
    if strong_evidence_max_displacement_m is not None and (
        not np.isfinite(strong_evidence_max_displacement_m)
        or strong_evidence_max_displacement_m <= 0
    ):
        raise ValueError("Strong-evidence displacement bound must be positive metres or None")
    if not 0 <= strong_evidence_confidence <= 1 or reprojection_outlier_px <= 0:
        raise ValueError("Invalid strong-evidence confidence or reprojection outlier threshold")
    quality = (
        np.ones(points.shape[:2]) if confidence is None else np.asarray(confidence, float).copy()
    )
    if quality.shape != points.shape[:2] or not np.isfinite(quality).all():
        raise ValueError("Joint confidence must be finite [frames,joints]")
    quality = np.clip(quality, 0, 1)
    quality[~np.isfinite(points).all(axis=2)] = 0
    body = None
    template_initialized: list[str] = []
    rig_source = reference_animation
    if reference_animation is not None and reference_animation.model_name != model:
        raise ValueError("Reference animation body model differs from requested model")
    if model != "fixture" and reference_animation is None:
        if model_path is None:
            raise ValueError(
                f"{model} requires your licensed model_path; no automatic fixture substitution"
            )
        body = load_body_model(
            model_path, model_name=model, trust_pickle=trust_pickle, num_betas=num_betas
        )
        rig_source = body
    if rig_source is not None:
        if points.shape[1] != len(rig_source.parents) or (
            joint_names is not None and joint_names != rig_source.names
        ):
            if joint_names is None or len(joint_names) != points.shape[1]:
                raise ValueError("Explicit observation joint_names required to map this body asset")
            lookup = {name: index for index, name in enumerate(joint_names)}
            mapped = np.full((len(times), len(rig_source.parents), 3), np.nan)
            mapped_quality = np.zeros(mapped.shape[:2])
            for target, name in enumerate(rig_source.names):
                if name in lookup:
                    mapped[:, target] = points[:, lookup[name]]
                    mapped_quality[:, target] = quality[:, lookup[name]]
            if observations2d is not None:
                original_pixels = np.asarray(observations2d, float)
                mapped_pixels = np.full(
                    original_pixels.shape[:2] + (len(rig_source.parents), 2), np.nan
                )
                original_quality2d = (
                    np.ones(original_pixels.shape[:3])
                    if confidence2d is None
                    else np.asarray(confidence2d, float)
                )
                mapped_quality2d = np.zeros(mapped_pixels.shape[:3])
                for target, name in enumerate(rig_source.names):
                    if name in lookup:
                        mapped_pixels[:, :, target] = original_pixels[:, :, lookup[name]]
                        mapped_quality2d[:, :, target] = original_quality2d[:, :, lookup[name]]
                observations2d, confidence2d = mapped_pixels, mapped_quality2d
            if prior_joints is not None:
                original_prior = np.asarray(prior_joints, float)
                mapped_prior = np.full_like(mapped, np.nan)
                for target, name in enumerate(rig_source.names):
                    if name in lookup:
                        mapped_prior[:, target] = original_prior[:, lookup[name]]
                prior_joints = mapped_prior
            points, quality = mapped, mapped_quality
    pixels, pixel_quality = None, None
    if observations2d is not None:
        if cameras is None or not cameras:
            raise ValueError("Direct 2D fitting requires explicit calibrated cameras")
        pixels = np.asarray(observations2d, float)
        if pixels.shape != (len(times), len(cameras), points.shape[1], 2):
            raise ValueError("2D body observations must be [frames,cameras,joints,2]")
        pixel_quality = (
            np.ones(pixels.shape[:3]) if confidence2d is None else np.asarray(confidence2d, float)
        )
        if pixel_quality.shape != pixels.shape[:3] or not np.isfinite(pixel_quality).all():
            raise ValueError("2D confidence must be finite [frames,cameras,joints]")
        if not np.isfinite(reprojection_weight) or reprojection_weight < 0:
            raise ValueError("Reprojection metres-per-pixel weight must be finite and nonnegative")
    prior_count = 0
    if prior_joints is not None:
        prior = np.asarray(prior_joints, float)
        if prior.shape != points.shape or not 0 <= prior_confidence <= 0.2:
            raise ValueError("Prior shape mismatch or prior confidence above subordinate limit 0.2")
        use = (quality < 0.2) & np.isfinite(prior).all(axis=2)
        points[use] = prior[use]
        quality[use] = prior_confidence
        prior_count = int(use.sum())
    if rig_source is not None:
        default_rest = (
            reference_animation.rest_joints
            if reference_animation is not None
            else body.rest(np.zeros(body.beta_count))[1]
        )
        model_root = int(np.flatnonzero(rig_source.parents == -1)[0])
        valid_root = np.isfinite(points[:, model_root]).all(axis=1)
        if not valid_root.any():
            raise ValueError(
                "A measured or explicitly initialized model root is required for metric fitting"
            )
        root_path = np.column_stack(
            [
                np.interp(times, times[valid_root], points[valid_root, model_root, axis])
                for axis in range(3)
            ]
        )
        for joint in range(len(rig_source.parents)):
            if not np.isfinite(points[:, joint]).all(axis=1).any():
                points[:, joint] = root_path + default_rest[joint] - default_rest[model_root]
                quality[:, joint] = 0
                template_initialized.append(rig_source.names[joint])
    points = _fill_missing(points, times)
    rejected_2d = 0
    if pixels is not None:
        # Static rejection decisions use the measured 3D trajectory, so the
        # optimizer cannot admit an outlier by moving the body toward it.
        pixel_quality = np.clip(pixel_quality.copy(), 0, 1)
        pixel_quality[~np.isfinite(pixels).all(axis=-1)] = 0
        for camera_index, camera in enumerate(cameras):
            if not camera.enabled:
                pixel_quality[:, camera_index] = 0
                continue
            for frame, time in enumerate(times):
                predicted = camera.project(points[frame], time=float(time))
                error = np.linalg.norm(predicted - pixels[frame, camera_index], axis=1)
                reject = (quality[frame] >= strong_evidence_confidence) & (
                    error > reprojection_outlier_px
                )
                rejected_2d += int(
                    np.count_nonzero(reject & (pixel_quality[frame, camera_index] > 0))
                )
                pixel_quality[frame, camera_index, reject] = 0
        # Number of agreeing cameras improves observability, but should not
        # accidentally multiply the trust assigned to the same measured joint.
        contributing = np.maximum(np.count_nonzero(pixel_quality > 0, axis=1), 1)
        pixel_quality = pixel_quality / contributing[:, None, :]
    posedirs = None
    if reference_animation is not None:
        rig_parents = reference_animation.parents.copy()
        names = reference_animation.names.copy()
        rest, shape = reference_animation.rest_joints.copy(), reference_animation.shape.copy()
        vertices, faces, weights = (
            reference_animation.vertices.copy(),
            reference_animation.faces.copy(),
            reference_animation.weights.copy(),
        )
        posedirs = reference_animation.posedirs
        shape_diagnostics = {
            **reference_animation.diagnostics.get("shape", {}),
            "locked_reference_shape": True,
        }
    elif model == "fixture":
        if parents is None:
            if points.shape[1] == 24:
                parents, default_names = SMPL_PARENTS, SMPL_NAMES
            elif points.shape[1] == 17:
                parents, default_names = H36M_PARENTS, H36M_NAMES
            else:
                raise ValueError("Explicit joint parents are required for this fixture skeleton")
            if joint_names is not None and list(joint_names) != list(default_names):
                raise ValueError(
                    "Joint names differ from default fixture schema; supply explicit parents"
                )
        else:
            default_names = [f"joint_{index:02d}" for index in range(points.shape[1])]
        rig_parents = np.asarray(parents, dtype=np.int64)
        if len(rig_parents) != points.shape[1]:
            raise ValueError("Skeleton parents differ from observations")
        names = list(joint_names or default_names)
        rest, shape = _persistent_rest(points, quality, rig_parents)
        vertices, faces, weights = _fixture_mesh(rest, rig_parents)
        shape_diagnostics = {"type": "persistent_median_bone_lengths", "unit": "metres"}
    else:
        rig_parents, names = body.parents, body.names
        _, target_lengths = _persistent_rest(points, quality, rig_parents)
        nonroot = np.flatnonzero(rig_parents >= 0)
        shape_evidence = np.sqrt(
            np.median(np.minimum(quality[:, nonroot], quality[:, rig_parents[nonroot]]), axis=0)
        )

        def shape_residual(beta: NDArray[np.float64]) -> NDArray[np.float64]:
            _, target_rest = body.rest(beta)
            actual = np.linalg.norm(
                target_rest[nonroot] - target_rest[rig_parents[nonroot]], axis=1
            )
            return np.r_[shape_evidence * (actual - target_lengths[nonroot]), 0.002 * beta]

        shape_result = least_squares(
            shape_residual, np.zeros(body.beta_count), bounds=(-5, 5), max_nfev=100
        )
        shape = shape_result.x
        vertices, rest = body.rest(shape)
        weights, faces, posedirs = body.weights, body.faces, body.posedirs
        shape_diagnostics = {
            "type": "persistent_beta",
            "cost": float(shape_result.cost),
            "asset": str(model_path),
            "asset_sha256": body.asset_sha256,
            "converged": bool(shape_result.success),
        }
    tree_order(rig_parents)
    root = int(np.flatnonzero(rig_parents == -1)[0])
    rotations = np.zeros_like(points)
    translations = points[:, root].copy()
    solved = np.zeros_like(points)
    residuals, evaluations, objective_terms, convergence = [], [], [], []
    active = np.flatnonzero([np.any(rig_parents == index) for index in range(len(rig_parents))])
    for frame in range(len(times)):
        initial = _initialize_pose(rest, rig_parents, points[frame])
        evidence_weight = np.sqrt(np.maximum(quality[frame], 1e-4))

        def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            pose = initial.copy()
            pose[active] = parameters[:-3].reshape(-1, 3)
            actual, _ = forward_kinematics(rest, rig_parents, pose, parameters[-3:])
            geometric = (actual - points[frame]) * evidence_weight[:, None]
            reprojection = []
            if pixels is not None:
                for camera_index, camera in enumerate(cameras):
                    if not camera.enabled:
                        continue
                    valid = np.isfinite(pixels[frame, camera_index]).all(axis=1) & (
                        pixel_quality[frame, camera_index] > 0
                    )
                    projected = camera.project(actual, time=float(times[frame]))
                    errors = projected[valid] - pixels[frame, camera_index, valid]
                    # Behind-camera predictions receive a finite large penalty
                    # rather than contaminating the numerical residual with NaN.
                    errors = np.nan_to_num(errors, nan=1e4, posinf=1e4, neginf=-1e4)
                    strength = np.sqrt(np.clip(pixel_quality[frame, camera_index, valid], 0, 1))
                    strength *= np.sqrt(float(camera.quality.confidence))
                    reprojection.extend(
                        (errors * strength[:, None] * reprojection_weight).reshape(-1)
                    )
            # The tiny initialization term resolves unobservable twist. It is
            # not a learned prior and never has a comparable geometric weight.
            regularization = 1e-5 * (parameters[:-3] - initial[active].reshape(-1))
            return np.r_[geometric.reshape(-1), reprojection, regularization]

        initial_parameters = np.r_[initial[active].reshape(-1), translations[frame]]
        initial_error = float(np.linalg.norm(residual(initial_parameters)))
        if initial_error > 1e-7:
            solution = least_squares(
                residual, initial_parameters, max_nfev=max_nfev, ftol=1e-7, xtol=1e-7
            )
            parameters, evaluations_frame = solution.x, solution.nfev
            convergence.append(bool(solution.success))
        else:
            parameters, evaluations_frame = initial_parameters, 0
            convergence.append(True)
        rotations[frame] = initial
        rotations[frame, active] = parameters[:-3].reshape(-1, 3)
        translations[frame] = parameters[-3:]
        solved[frame], _ = forward_kinematics(
            rest, rig_parents, rotations[frame], translations[frame]
        )
        residuals.append(float(np.mean(np.linalg.norm(solved[frame] - points[frame], axis=1))))
        evaluations.append(int(evaluations_frame))
        terms = residual(parameters)
        split = points.shape[1] * 3
        regularization_count = len(active) * 3
        pixel_terms = terms[split:-regularization_count]
        objective_terms.append(
            {
                "confidence_weighted_3d": float(terms[:split] @ terms[:split]),
                "multiview_2d_reprojection": float(pixel_terms @ pixel_terms),
                "unobservable_twist_initialization": float(
                    terms[-regularization_count:] @ terms[-regularization_count:]
                ),
            }
        )
    rig_confidence = quality * np.exp(-np.linalg.norm(solved - points, axis=2) / 0.02)
    displacement = np.linalg.norm(solved - points, axis=2)
    strong = quality >= strong_evidence_confidence
    strong_maximum = float(displacement[strong].max()) if strong.any() else 0.0
    if (
        strong_evidence_max_displacement_m is not None
        and strong_maximum > strong_evidence_max_displacement_m
    ):
        frame, joint = np.unravel_index(
            np.argmax(np.where(strong, displacement, -1)), displacement.shape
        )
        raise ValueError(
            f"Body fit conflicts with authoritative joint evidence: {names[joint]} at "
            f"t={times[frame]:.6f}s moved {strong_maximum:.4f}m, exceeding "
            f"{strong_evidence_max_displacement_m:.4f}m. Inspect actor proportions, "
            "joint mapping, triangulation and confidence before relaxing the explicit bound."
        )
    return Animation(
        times,
        solved,
        rest,
        rig_parents,
        names,
        vertices,
        faces,
        weights,
        rotations,
        translations,
        shape,
        model,
        {
            "shape": shape_diagnostics,
            "mean_joint_residual_m": float(np.mean(residuals)),
            "per_frame_residual_m": residuals,
            "pose_evaluations": evaluations,
            "objective_terms": objective_terms,
            "per_frame_converged": convergence,
            "solver_warnings": []
            if all(convergence)
            else ["Some IK frames reached the evaluation limit; inspect residuals"],
            "prior_used_observations": prior_count,
            "shape_constant": True,
            "template_initialized_joints": template_initialized,
            "camera_parameters_modified": False,
            "silhouette_fit": "not_implemented",
            "direct_2d_fit": "enabled" if pixels is not None else "not_requested",
            "reprojection_weight_metres_per_pixel": reprojection_weight,
            "reprojection_outliers_rejected": rejected_2d,
            "strong_evidence_max_displacement_m": strong_maximum,
            "strong_evidence_displacement_bound_m": strong_evidence_max_displacement_m,
        },
        posedirs,
        rig_confidence,
    )
