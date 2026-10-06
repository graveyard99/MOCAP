"""Independent NumPy implementation of shape/pose blend shapes and rigid skinning.

No licensed topology, coefficients, regressors, or pretrained data are bundled.
The same numerical contract supports full-rotation SMPL, SMPL-H and SMPL-X
assets; hand PCA and expression parameter conveniences are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class ModelAssetError(ValueError):
    """A licensed model asset is missing, untrusted, or numerically invalid."""


def tree_order(parents: NDArray[np.int64]) -> list[int]:
    """Validate and topologically order a single-root conventional skeleton."""
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1:
        raise ValueError("Skeleton must have exactly one root with parent -1")
    count = len(parents)
    if np.any(parents < -1) or np.any(parents >= count):
        raise ValueError("Skeleton parent index is outside the joint array")
    order = [int(roots[0])]
    for joint in order:
        order.extend(int(x) for x in np.flatnonzero(parents == joint))
        if len(order) > count:
            raise ValueError("Skeleton parent graph contains a cycle")
    if len(order) != count:
        raise ValueError("Skeleton has disconnected joints or a cycle")
    return order


def forward_kinematics(
    rest_joints: NDArray[np.float64],
    parents: NDArray[np.int64],
    rotations: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Local axis-angle rotations to metric joints and global orientations.

    Translation is the world position of the skeleton root, not an additional
    template-space displacement. This convention is shared with the exporters.
    """
    local = Rotation.from_rotvec(np.asarray(rotations)).as_matrix()
    points = np.empty_like(rest_joints)
    global_rot = np.empty_like(local)
    for joint in tree_order(parents):
        parent = int(parents[joint])
        if parent == -1:
            global_rot[joint] = local[joint]
            points[joint] = translation
        else:
            global_rot[joint] = global_rot[parent] @ local[joint]
            points[joint] = points[parent] + global_rot[parent] @ (
                rest_joints[joint] - rest_joints[parent]
            )
    return points, global_rot


def skin_vertices(
    vertices: NDArray[np.float64],
    rest_joints: NDArray[np.float64],
    parents: NDArray[np.int64],
    weights: NDArray[np.float64],
    rotations: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Skin a rest mesh with normalized per-vertex joint weights."""
    points, orientations = forward_kinematics(rest_joints, parents, rotations, translation)
    transforms = np.zeros((len(parents), 4, 4), dtype=float)
    transforms[:, :3, :3] = orientations
    transforms[:, :3, 3] = points - np.einsum("nij,nj->ni", orientations, rest_joints)
    transforms[:, 3, 3] = 1
    blended = np.einsum("vj,jab->vab", weights, transforms)
    homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
    return np.einsum("vab,vb->va", blended, homogeneous)[:, :3], points


@dataclass
class BodyModel:
    """Validated body asset in the installation's metre / +Y-up convention."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    shapedirs: NDArray[np.float64]
    posedirs: NDArray[np.float64]
    joint_regressor: NDArray[np.float64]
    parents: NDArray[np.int64]
    weights: NDArray[np.float64]
    names: list[str]
    model_name: str = "smpl"
    asset_path: str | None = None
    asset_sha256: str | None = None

    def __post_init__(self) -> None:
        vertices, joints = len(self.vertices), len(self.parents)
        tree_order(self.parents)
        arrays = [
            self.vertices,
            self.shapedirs,
            self.posedirs,
            self.joint_regressor,
            self.weights,
        ]
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ModelAssetError("Model contains non-finite coefficients")
        if self.vertices.shape != (vertices, 3) or self.weights.shape != (vertices, joints):
            raise ModelAssetError("Model vertex/weight dimensions are incompatible")
        if self.shapedirs.shape[:2] != (vertices, 3):
            raise ModelAssetError("shapedirs must have shape [vertices,3,beta]")
        if self.shapedirs.shape[2] < 1:
            raise ModelAssetError("Model requires at least one persistent shape direction")
        if self.joint_regressor.shape != (joints, vertices):
            raise ModelAssetError("Joint regressor must have shape [joints,vertices]")
        if self.posedirs.shape != (vertices * 3, (joints - 1) * 9):
            raise ModelAssetError("posedirs must match all non-root rotation matrices")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ModelAssetError("Mesh faces must be triangles")
        if np.any(self.faces < 0) or np.any(self.faces >= vertices):
            raise ModelAssetError("Face index outside mesh")
        if len(self.names) != joints:
            raise ModelAssetError("Joint name count differs from skeleton")
        if len(set(self.names)) != joints or any(not name.strip() for name in self.names):
            raise ModelAssetError("Joint names must be distinct and nonempty")
        if np.any(self.weights < -1e-8) or not np.allclose(self.weights.sum(axis=1), 1, atol=1e-4):
            raise ModelAssetError("Skin weights must be nonnegative and sum to one")
        if self.model_name not in {"smpl", "smplh", "smplx"}:
            raise ModelAssetError("Select smpl, smplh or smplx explicitly")

    @property
    def beta_count(self) -> int:
        return self.shapedirs.shape[2]

    def rest(self, shape: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        shape = np.asarray(shape, dtype=float)
        if shape.shape != (self.beta_count,):
            raise ValueError(f"Expected {self.beta_count} shape coefficients")
        vertices = self.vertices + np.einsum("vck,k->vc", self.shapedirs, shape)
        return vertices, self.joint_regressor @ vertices

    def evaluate(
        self,
        shape: NDArray[np.float64],
        rotations: NDArray[np.float64],
        translation: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        vertices, joints = self.rest(shape)
        matrices = Rotation.from_rotvec(rotations).as_matrix()
        root = int(np.flatnonzero(self.parents == -1)[0])
        feature = (np.delete(matrices, root, axis=0) - np.eye(3)).reshape(-1)
        vertices = vertices + (self.posedirs @ feature).reshape(-1, 3)
        return skin_vertices(vertices, joints, self.parents, self.weights, rotations, translation)


SMPL_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


def family_joint_names(model_name: str, count: int) -> list[str]:
    """Conventional full-rotation SMPL/H/X names, including articulated hands."""
    fingers = [
        f"{side}_{finger}{segment}"
        for side in ("left", "right")
        for finger in ("index", "middle", "pinky", "ring", "thumb")
        for segment in (1, 2, 3)
    ]
    if model_name == "smplh":
        names = SMPL_NAMES[:22] + fingers
    elif model_name == "smplx":
        names = SMPL_NAMES[:22] + ["jaw", "left_eye", "right_eye"] + fingers
    else:
        names = SMPL_NAMES
    return names[:count] + [f"joint_{index:02d}" for index in range(len(names), count)]


def _dense(value: Any) -> NDArray[np.float64]:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value, dtype=float)


def load_body_model(
    path: str | Path,
    *,
    model_name: str = "smpl",
    trust_pickle: bool = False,
    num_betas: int = 10,
) -> BodyModel:
    """Load user-owned numerical arrays; object/pickle loads require consent.

    Official pickle assets can execute arbitrary Python during deserialization.
    Convert only trusted files. Default NPZ loading never enables pickle.
    """
    source = Path(path)
    if num_betas < 1 or (model_name == "smplx" and num_betas > 300):
        raise ModelAssetError(
            "num_betas must be positive; SMPL-X expression directions are not shape betas"
        )
    if not source.is_file():
        raise ModelAssetError(f"Missing licensed {model_name} asset: {source}")
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
    elif source.suffix.lower() in {".pkl", ".pickle"}:
        if not trust_pickle:
            raise ModelAssetError(
                "Pickle model requires explicit trust_pickle=True; prefer safe NPZ"
            )
        with source.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")  # noqa: S301 -- explicit user trust
    else:
        raise ModelAssetError(
            "Use a numerical NPZ model asset or explicitly trusted official pickle"
        )
    required = {"v_template", "f", "shapedirs", "posedirs", "J_regressor", "weights"}
    missing = required - data.keys()
    if missing:
        raise ModelAssetError(f"Missing numerical arrays: {', '.join(sorted(missing))}")
    vertices = _dense(data["v_template"])
    weights = _dense(data["weights"])
    joints = weights.shape[1]
    if "parents" in data:
        parents = np.asarray(data["parents"], dtype=np.int64).copy()
    elif "kintree_table" in data:
        table = np.asarray(data["kintree_table"])
        lookup = {int(value): index for index, value in enumerate(table[1])}
        parents = np.array([lookup.get(int(value), -1) for value in table[0]], dtype=np.int64)
        parents[0] = -1
    else:
        raise ModelAssetError("Model requires parents or kintree_table")
    posedirs = _dense(data["posedirs"])
    if posedirs.ndim == 3:
        posedirs = posedirs.reshape(len(vertices) * 3, -1)
    if posedirs.shape[0] != len(vertices) * 3 and posedirs.shape[1] == len(vertices) * 3:
        posedirs = posedirs.T
    names = (
        [str(name) for name in data["joint_names"]]
        if "joint_names" in data
        else family_joint_names(model_name, joints)
    )
    with source.open("rb") as handle:
        asset_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    shapedirs = _dense(data["shapedirs"])
    if shapedirs.ndim != 3:
        raise ModelAssetError("shapedirs must be a numerical [vertices,3,beta] array")
    # Official SMPL-X commonly appends expression directions after 300 shape
    # directions. They must never become persistent actor proportions.
    shapedirs = shapedirs[:, :, : min(num_betas, shapedirs.shape[2])]
    return BodyModel(
        vertices,
        np.asarray(data["f"], dtype=np.int64),
        shapedirs,
        posedirs,
        _dense(data["J_regressor"]),
        parents,
        weights,
        names,
        model_name,
        str(source),
        asset_hash,
    )
