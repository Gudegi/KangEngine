"""SMPL, SMPL-H, and SMPL-X model assets with body-shape evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch

from .._core import _ke
from ..utils import quat_wxyz_to_matrix
from ..visual.deformable import SkinnedSurface, SkinnedSurfaceAsset

if TYPE_CHECKING:
    from ..animation import SkeletonState, SkeletonTree
    from ..app import App
    from ..material import Material


SMPL_JOINT_NAMES = (
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
)

_BODY_JOINT_NAMES = SMPL_JOINT_NAMES[:22]

_HAND_JOINT_NAMES = (
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)

_FACE_JOINT_NAMES = (
    "jaw",
    "left_eye",
    "right_eye",
)

SMPLH_JOINT_NAMES = _BODY_JOINT_NAMES + _HAND_JOINT_NAMES
SMPLX_JOINT_NAMES = (
    _BODY_JOINT_NAMES + _FACE_JOINT_NAMES + _HAND_JOINT_NAMES
)  # Face's indices are placed first than hand's.

_REPOSITORY_MODEL_FILES = {
    "female": Path("smpl") / "SMPL_FEMALE.pkl",
    "male": Path("smpl") / "SMPL_MALE.pkl",
    "neutral": Path("smpl") / "SMPL_NEUTRAL.pkl",
}

_REPOSITORY_SMPLX_MODEL_FILES = {
    "female": Path("smplx") / "SMPLX_FEMALE.npz",
    "male": Path("smplx") / "SMPLX_MALE.npz",
    "neutral": Path("smplx") / "SMPLX_NEUTRAL.npz",
}

_REPOSITORY_SMPLH_MODEL_FILES = {
    "female": Path("smplh") / "female" / "model.npz",
    "male": Path("smplh") / "male" / "model.npz",
    "neutral": Path("smplh") / "neutral" / "model.npz",
}


class _PickledSparseMatrix:
    """Minimal SciPy CSC/CSR state reader used by legacy SMPL pickles."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def toarray(self):
        result = np.zeros(self._shape, dtype=np.asarray(self.data).dtype)
        if self.format == "csc":
            for column in range(self._shape[1]):
                begin, end = self.indptr[column : column + 2]
                result[self.indices[begin:end], column] = self.data[begin:end]
        elif self.format == "csr":
            for row in range(self._shape[0]):
                begin, end = self.indptr[row : row + 2]
                result[row, self.indices[begin:end]] = self.data[begin:end]
        else:
            raise ValueError(f"unsupported pickled sparse format: {self.format}")
        return result


class _PickledChumpyArray:
    """Read the stored ndarray from old chumpy-backed model fields."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def r(self):
        return self.x


class _SMPLUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("scipy.sparse") and name in ("csc_matrix", "csr_matrix"):
            return _PickledSparseMatrix
        if module.startswith("chumpy"):
            return _PickledChumpyArray
        return super().find_class(module, name)


def _repository_model_path(files, gender: str) -> Path:
    key = str(gender).lower()
    if key not in files:
        choices = ", ".join(sorted(files))
        raise ValueError(f"unknown model gender {gender!r}; expected one of: {choices}")
    root = Path(__file__).resolve().parents[3]
    return root / "assets" / "external" / "smpl_models" / files[key]


def repository_smpl_model_path(gender: str = "neutral") -> Path:
    """Return a repository SMPL model path for KangEngine examples."""
    return _repository_model_path(_REPOSITORY_MODEL_FILES, gender)


def repository_smplx_model_path(gender: str = "neutral") -> Path:
    """Return a repository SMPL-X model path for KangEngine examples."""
    return _repository_model_path(_REPOSITORY_SMPLX_MODEL_FILES, gender)


def repository_smplh_model_path(gender: str = "neutral") -> Path:
    """Return a repository SMPL-H model path for KangEngine examples."""
    return _repository_model_path(_REPOSITORY_SMPLH_MODEL_FILES, gender)


def _dense_array(value, dtype) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "r"):
        value = value.r
    return np.asarray(value, dtype=dtype)


def _load_model_data(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            fields = (
                "v_template",
                "shapedirs",
                "J_regressor",
                "weights",
                "f",
                "posedirs",
                "kintree_table",
            )
            return {name: source[name] for name in fields}
    if suffix == ".pkl":
        with path.open("rb") as source:
            data = _SMPLUnpickler(source, encoding="latin1").load()
        if not isinstance(data, dict):
            raise ValueError(f"SMPL pickle must contain a dictionary: {path}")
        return data
    raise ValueError(f"unsupported SMPL model file: {path} (expected .npz or .pkl)")


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-8)


def _top_four_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    selected = np.argpartition(weights, -4, axis=1)[:, -4:]
    values = np.take_along_axis(weights, selected, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    selected = np.take_along_axis(selected, order, axis=1).astype(np.int32)
    values = np.take_along_axis(values, order, axis=1).astype(np.float32)
    retained = values.sum(axis=1)
    discarded = float(np.max(np.maximum(0.0, 1.0 - retained)))
    values /= np.maximum(retained[:, None], 1e-8)
    return selected, values, discarded


@dataclass(frozen=True)
class SMPLBody:
    skeleton_tree: SkeletonTree
    surface_asset: SkinnedSurfaceAsset
    joints: np.ndarray
    discarded_weight_max: float
    pose_directions: np.ndarray

    def create_visual(
        self,
        app: App,
        path: str,
        *,
        material: Material,
        color: npt.ArrayLike | None = None,
    ) -> SkinnedSurface:
        return SkinnedSurface.create(
            app, path, self.surface_asset, material=material, color=color
        )

    def corrected_bind_vertices(
        self, rotations_wxyz: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        """Evaluate SMPL pose blend shapes before LBS."""
        rotations = np.asarray(rotations_wxyz, np.float32)
        joint_count = len(self.surface_asset.bone_node_indices)
        if rotations.shape != (joint_count, 4):
            raise ValueError(f"rotations_wxyz must have shape [{joint_count}, 4]")
        feature = (
            quat_wxyz_to_matrix(torch.from_numpy(rotations[1:])).numpy()
            - np.eye(3, dtype=np.float32)[None, :, :]
        ).reshape(-1)
        offsets = (self.pose_directions.T @ feature).reshape(-1, 3)
        return self.surface_asset.positions + offsets

    def update_pose_correctives(
        self,
        surface: SkinnedSurface,
        state: SkeletonState,
        *,
        enabled: bool = True,
        update_normals: bool = True,
    ) -> SkinnedSurface:
        """Update or clear pose-dependent geometry from a SkeletonState."""
        if surface.asset is not self.surface_asset:
            raise ValueError("surface was not created from this SMPL body")
        if not enabled:
            return surface.reset_bind_geometry()
        if not hasattr(state, "rotation"):
            raise TypeError("state must be a SkeletonState")
        if not state.is_local():
            raise ValueError("SMPL pose correctives require local rotations")
        rotations_wxyz = np.empty((state.num_joints(), 4), np.float32)
        for joint in range(state.num_joints()):
            rotation = state.rotation(joint)
            rotations_wxyz[joint] = (
                rotation.w,
                rotation.x,
                rotation.y,
                rotation.z,
            )
        return self.update_pose_correctives_from_rotations(
            surface,
            rotations_wxyz,
            enabled=True,
            update_normals=update_normals,
        )

    def update_pose_correctives_from_rotations(
        self,
        surface: SkinnedSurface,
        local_rotations_wxyz: npt.ArrayLike,
        *,
        enabled: bool = True,
        update_normals: bool = True,
    ) -> SkinnedSurface:
        """Update or clear pose-dependent geometry from raw local rotations."""
        if surface.asset is not self.surface_asset:
            raise ValueError("surface was not created from this SMPL body")
        if not enabled:
            return surface.reset_bind_geometry()
        positions = self.corrected_bind_vertices(local_rotations_wxyz)
        normals = (
            _vertex_normals(positions, self.surface_asset.indices.reshape(-1, 3))
            if update_normals
            else None
        )
        return surface.update_bind_geometry(positions, normals)


class SMPLHBody(SMPLBody):
    """A 52-joint SMPL-H body and renderable surface asset."""


class SMPLXBody(SMPLBody):
    """A 55-joint SMPL-X body and renderable surface asset."""


class SMPLModel:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        source = _load_model_data(self.path)
        self.template = _dense_array(source["v_template"], np.float32)
        self.shape_directions = _dense_array(source["shapedirs"], np.float32)
        self.joint_regressor = _dense_array(source["J_regressor"], np.float32)
        self.weights = _dense_array(source["weights"], np.float32)
        self.faces = _dense_array(source["f"], np.int32)
        pose_directions = _dense_array(source["posedirs"], np.float32)
        table = _dense_array(source["kintree_table"], np.int64)
        if pose_directions.shape[:2] == self.template.shape:
            pose_directions = pose_directions.reshape(-1, pose_directions.shape[-1]).T
        else:
            pose_directions = pose_directions.reshape(pose_directions.shape[0], -1)
        self.pose_directions = np.ascontiguousarray(pose_directions)
        if len(table[1]) != 24:
            raise ValueError("SMPLModel expects the standard 24-joint topology")
        ids = {int(value): index for index, value in enumerate(table[1])}
        self.parents = np.full(24, -1, np.int32)
        for joint in range(1, 24):
            self.parents[joint] = ids[int(table[0, joint])]

    @classmethod
    def load(cls, path: str | Path) -> "SMPLModel":
        return cls(path)

    def create_body(self, betas: npt.ArrayLike | None = None) -> SMPLBody:
        coefficients = np.zeros(self.shape_directions.shape[-1], np.float32)
        if betas is not None:
            values = np.asarray(betas, np.float32).reshape(-1)
            count = min(len(values), len(coefficients))
            coefficients[:count] = values[:count]
        vertices = self.template + np.tensordot(
            self.shape_directions, coefficients, axes=([-1], [0])
        )
        joints = self.joint_regressor @ vertices
        local = joints.copy()
        local[1:] -= joints[self.parents[1:]]
        local[0] = 0.0
        rotations = np.zeros((24, 4), np.float32)
        rotations[:, 0] = 1.0
        tree = _ke.animation.SkeletonTree(
            list(SMPL_JOINT_NAMES), self.parents.tolist(), local, rotations
        )
        inverse_binds = np.tile(np.eye(4, dtype=np.float32), (24, 1, 1))
        inverse_binds[:, :3, 3] = -joints
        bone_indices, bone_weights, discarded = _top_four_weights(self.weights)
        asset = SkinnedSurfaceAsset(
            tree,
            vertices.astype(np.float32),
            _vertex_normals(vertices, self.faces).astype(np.float32),
            self.faces.reshape(-1),
            bone_indices,
            bone_weights,
            np.arange(24, dtype=np.int32),
            inverse_binds,
        )
        return SMPLBody(
            tree,
            asset,
            joints.astype(np.float32),
            discarded,
            self.pose_directions,
        )


class SMPLHModel(SMPLModel):
    """Standard 52-joint SMPL-H NPZ model adapter."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        source = _load_model_data(self.path)
        self.template = _dense_array(source["v_template"], np.float32)
        self.shape_directions = _dense_array(source["shapedirs"], np.float32)
        self.joint_regressor = _dense_array(source["J_regressor"], np.float32)
        self.weights = _dense_array(source["weights"], np.float32)
        self.faces = _dense_array(source["f"], np.int32)
        pose_directions = _dense_array(source["posedirs"], np.float32)
        table = _dense_array(source["kintree_table"], np.int64)
        if pose_directions.shape[:2] == self.template.shape:
            pose_directions = pose_directions.reshape(-1, pose_directions.shape[-1]).T
        else:
            pose_directions = pose_directions.reshape(pose_directions.shape[0], -1)
        self.pose_directions = np.ascontiguousarray(pose_directions)
        if len(table[1]) != 52:
            raise ValueError("SMPLHModel expects the standard 52-joint topology")
        ids = {int(value): index for index, value in enumerate(table[1])}
        self.parents = np.full(52, -1, np.int32)
        for joint in range(1, 52):
            self.parents[joint] = ids[int(table[0, joint])]

    def create_body(self, betas: npt.ArrayLike | None = None) -> SMPLHBody:
        coefficients = np.zeros(self.shape_directions.shape[-1], np.float32)
        if betas is not None:
            values = np.asarray(betas, np.float32).reshape(-1)
            count = min(len(values), len(coefficients))
            coefficients[:count] = values[:count]
        vertices = self.template + np.tensordot(
            self.shape_directions, coefficients, axes=([-1], [0])
        )
        joints = self.joint_regressor @ vertices
        local = joints.copy()
        local[1:] -= joints[self.parents[1:]]
        local[0] = 0.0
        rotations = np.zeros((52, 4), np.float32)
        rotations[:, 0] = 1.0
        tree = _ke.animation.SkeletonTree(
            list(SMPLH_JOINT_NAMES), self.parents.tolist(), local, rotations
        )
        inverse_binds = np.tile(np.eye(4, dtype=np.float32), (52, 1, 1))
        inverse_binds[:, :3, 3] = -joints
        bone_indices, bone_weights, discarded = _top_four_weights(self.weights)
        asset = SkinnedSurfaceAsset(
            tree,
            vertices.astype(np.float32),
            _vertex_normals(vertices, self.faces).astype(np.float32),
            self.faces.reshape(-1),
            bone_indices,
            bone_weights,
            np.arange(52, dtype=np.int32),
            inverse_binds,
        )
        return SMPLHBody(
            tree,
            asset,
            joints.astype(np.float32),
            discarded,
            self.pose_directions,
        )


class SMPLXModel(SMPLModel):
    """Standard 55-joint SMPL-X NPZ model adapter."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        source = _load_model_data(self.path)
        self.template = _dense_array(source["v_template"], np.float32)
        self.shape_directions = _dense_array(source["shapedirs"], np.float32)
        self.joint_regressor = _dense_array(source["J_regressor"], np.float32)
        self.weights = _dense_array(source["weights"], np.float32)
        self.faces = _dense_array(source["f"], np.int32)
        pose_directions = _dense_array(source["posedirs"], np.float32)
        table = _dense_array(source["kintree_table"], np.int64)
        if pose_directions.shape[:2] == self.template.shape:
            pose_directions = pose_directions.reshape(-1, pose_directions.shape[-1]).T
        else:
            pose_directions = pose_directions.reshape(pose_directions.shape[0], -1)
        self.pose_directions = np.ascontiguousarray(pose_directions)
        if len(table[1]) != 55:
            raise ValueError("SMPLXModel expects the standard 55-joint topology")
        ids = {int(value): index for index, value in enumerate(table[1])}
        self.parents = np.full(55, -1, np.int32)
        for joint in range(1, 55):
            self.parents[joint] = ids[int(table[0, joint])]

    def create_body(self, betas: npt.ArrayLike | None = None) -> SMPLXBody:
        coefficients = np.zeros(self.shape_directions.shape[-1], np.float32)
        if betas is not None:
            values = np.asarray(betas, np.float32).reshape(-1)
            count = min(len(values), len(coefficients))
            coefficients[:count] = values[:count]
        vertices = self.template + np.tensordot(
            self.shape_directions, coefficients, axes=([-1], [0])
        )
        joints = self.joint_regressor @ vertices
        local = joints.copy()
        local[1:] -= joints[self.parents[1:]]
        local[0] = 0.0
        rotations = np.zeros((55, 4), np.float32)
        rotations[:, 0] = 1.0
        tree = _ke.animation.SkeletonTree(
            list(SMPLX_JOINT_NAMES), self.parents.tolist(), local, rotations
        )
        inverse_binds = np.tile(np.eye(4, dtype=np.float32), (55, 1, 1))
        inverse_binds[:, :3, 3] = -joints
        bone_indices, bone_weights, discarded = _top_four_weights(self.weights)
        asset = SkinnedSurfaceAsset(
            tree,
            vertices.astype(np.float32),
            _vertex_normals(vertices, self.faces).astype(np.float32),
            self.faces.reshape(-1),
            bone_indices,
            bone_weights,
            np.arange(55, dtype=np.int32),
            inverse_binds,
        )
        return SMPLXBody(
            tree,
            asset,
            joints.astype(np.float32),
            discarded,
            self.pose_directions,
        )


__all__ = [
    "SMPLBody",
    "SMPLModel",
    "SMPL_JOINT_NAMES",
    "SMPLHBody",
    "SMPLHModel",
    "SMPLH_JOINT_NAMES",
    "SMPLXBody",
    "SMPLXModel",
    "SMPLX_JOINT_NAMES",
    "repository_smpl_model_path",
    "repository_smplh_model_path",
    "repository_smplx_model_path",
]
