"""Asset loaders and model assets."""

from __future__ import annotations

import numpy.typing as npt
from types import ModuleType

from .. import UpAxis
from ..animation import SkeletonMotion, SkeletonTree
from ..scene import MeshData, SkinnedMeshData
from . import amass as amass, smpl as smpl
from .amass import AMASSInfo as AMASSInfo, AMASSLoader as AMASSLoader
from .smpl import (
    SMPLBody as SMPLBody,
    SMPLHBody as SMPLHBody,
    SMPLHModel as SMPLHModel,
    SMPLModel as SMPLModel,
    SMPLXBody as SMPLXBody,
    SMPLXModel as SMPLXModel,
)

class ImportDiagnostics:
    """Warnings collected while importing an asset."""

    warnings: list[str]

class ArticulationDesc: ...
class CollisionGeomDesc: ...
class CollisionGeomDescType: ...
class InertialDesc: ...
class JointDesc: ...
class SiteDesc: ...
class SiteDescType: ...
class VisualGeomDesc: ...

class MJCFImportResult:
    articulation: ArticulationDesc
    diagnostics: ImportDiagnostics

class MJCFLoader:
    """Load MJCF/XML articulation descriptions."""

    @staticmethod
    def load(
        mjcf_path: str, scale: float = 1.0, order: str = "DFS"
    ) -> ArticulationDesc: ...

    @staticmethod
    def parse(
        mjcf_path: str, scale: float = 1.0, order: str = "DFS"
    ) -> MJCFImportResult: ...

class BVHImportResult:
    """BVH motion and source-file metadata returned by :meth:`BVHLoader.parse`."""

    motion: SkeletonMotion
    diagnostics: ImportDiagnostics
    frame_count: int
    frame_time: float
    frame_rate: float

class BVHLoader:
    """Load a BVH hierarchy or sampled skeleton motion."""

    @staticmethod
    def load_skeleton(bvh_path: str, scale: float = 1.0) -> SkeletonTree:
        """Load only the skeleton hierarchy from ``bvh_path``."""
        ...

    @staticmethod
    def load_motion(bvh_path: str, scale: float = 1.0) -> SkeletonMotion:
        """Load the hierarchy and animation frames as a ``SkeletonMotion``."""
        ...

    @staticmethod
    def parse(bvh_path: str, scale: float = 1.0) -> BVHImportResult:
        """Load BVH motion together with diagnostics and source timing metadata."""
        ...

class FBXAnimationClipInfo:
    """Animation clip metadata discovered in an FBX file."""

    name: str
    start_time: float
    end_time: float
    frame_rate: float

class FBXMaterialInfo:
    """Material metadata imported from an FBX file."""

    name: str
    diffuse_color: tuple[float, float, float, float]
    diffuse_texture_path: str
    normal_texture_path: str
    has_diffuse_texture: bool
    has_embedded_diffuse_texture: bool
    has_normal_texture: bool
    has_embedded_normal_texture: bool

class FBXMeshMetadata:
    name: str
    vertex_count: int
    index_count: int
    material_count: int
    primary_material_index: int
    has_normals: bool
    has_uvs: bool
    has_skin: bool
    skin_cluster_names: list[str]
    materials: list[FBXMaterialInfo]

class FBXSkinClusterInfo:
    mesh_name: str
    cluster_name: str
    link_name: str
    weight_count: int
    index_count: int
    min_index: int
    max_index: int
    min_weight: float
    max_weight: float
    weight_sum: float

class _FBXDebugModule(ModuleType):
    def load_skin_cluster_infos(
        self, fbx_path: str, scale: float = 0.01
    ) -> list[FBXSkinClusterInfo]: ...

FBXDebug: _FBXDebugModule

class FBXSkinnedMeshInfo:
    """Imported skinned-mesh data and skin bindings."""

    name: str
    vertex_count: int
    index_count: int
    material_count: int
    primary_material_index: int
    materials: list[FBXMaterialInfo]
    has_normals: bool
    has_uvs: bool
    has_skin: bool
    skin_cluster_names: list[str]
    bone_names: list[str]
    overweight_vertex_count: int
    unweighted_vertex_count: int
    mismatched_cluster_count: int

    @property
    def mesh_data(self) -> SkinnedMeshData: ...

class FBXMeshInfo:
    """Imported static FBX mesh and material metadata."""

    name: str
    vertex_count: int
    index_count: int
    material_count: int
    primary_material_index: int
    materials: list[FBXMaterialInfo]
    has_normals: bool
    has_uvs: bool
    skin_cluster_names: list[str]

    @property
    def mesh_data(self) -> MeshData: ...

class FBXCharacterData:
    """An FBX motion and its skinned meshes."""

    motion: SkeletonMotion
    skinned_meshes: list[FBXSkinnedMeshInfo]

class FBXImportResult:
    """FBX character data, clips, and import diagnostics."""

    character: FBXCharacterData
    motion: SkeletonMotion
    skinned_meshes: list[FBXSkinnedMeshInfo]
    clips: list[FBXAnimationClipInfo]
    diagnostics: ImportDiagnostics

class FBXLoader:
    """Load FBX skeletons, animation clips, and character data."""

    @staticmethod
    def load_skeleton(fbx_path: str, scale: float = 0.01) -> SkeletonTree:
        """Load only the FBX skeleton hierarchy."""
        ...

    @staticmethod
    def load_animation_clip_infos(fbx_path: str) -> list[FBXAnimationClipInfo]:
        """Return metadata for the animation clips in an FBX file."""
        ...

    @staticmethod
    def load_motion(
        fbx_path: str,
        clip_index: int = -1,
        fps: float = -1.0,
        scale: float = 0.01,
    ) -> SkeletonMotion:
        """Load one FBX animation clip as a ``SkeletonMotion``."""
        ...

    @staticmethod
    def load_meshes(fbx_path: str, scale: float = 0.01) -> list[FBXMeshInfo]:
        """Load static meshes without animation data."""
        ...

    @staticmethod
    def parse(
        fbx_path: str,
        clip_index: int = -1,
        fps: float = -1.0,
        scale: float = 0.01,
    ) -> FBXImportResult:
        """Load FBX character data together with clip metadata and diagnostics."""
        ...

    @staticmethod
    def parse_with_bind(
        motion_fbx_path: str,
        bind_fbx_path: str,
        clip_index: int = -1,
        fps: float = -1.0,
        scale: float = 0.01,
    ) -> FBXImportResult:
        """Parse animation using a separate FBX file for the bind pose."""
        ...

    @staticmethod
    def load_character(
        fbx_path: str,
        clip_index: int = -1,
        fps: float = -1.0,
        scale: float = 0.01,
    ) -> FBXCharacterData:
        """Load one animation clip and all associated skinned meshes."""
        ...

    @staticmethod
    def load_character_with_bind(
        motion_fbx_path: str,
        bind_fbx_path: str,
        clip_index: int = -1,
        fps: float = -1.0,
        scale: float = 0.01,
    ) -> FBXCharacterData:
        """Load character data using a separate FBX file for the bind pose."""
        ...

    @staticmethod
    def load_skinned_meshes(
        fbx_path: str, scale: float = 0.01
    ) -> list[FBXSkinnedMeshInfo]:
        """Load skinned meshes without animation data."""
        ...

class USDMeshInfo:
    prim_path: str
    name: str
    material_path: str
    diffuse_texture_path: str
    normal_texture_path: str
    mesh_data: MeshData
    vertex_count: int
    index_count: int

class USDImportResult:
    meshes: list[USDMeshInfo]
    diagnostics: ImportDiagnostics

class USDLoader:
    """Load meshes from USD scenes."""

    @staticmethod
    def parse(usd_path: str, scale: float = 1.0) -> USDImportResult: ...

    @staticmethod
    def load_meshes(usd_path: str, scale: float = 1.0) -> list[USDMeshInfo]: ...

class ObjMaterialInfo:
    name: str
    ambient_color: tuple[float, float, float]
    diffuse_color: tuple[float, float, float]
    specular_color: tuple[float, float, float]
    shininess: float
    diffuse_texture_path: str
    specular_texture_path: str
    alpha_texture_path: str
    normal_texture_path: str
    has_diffuse_texture: bool
    has_specular_texture: bool
    has_alpha_texture: bool
    has_normal_texture: bool

class ObjMeshSubsetInfo:
    name: str
    material_index: int
    mesh_data: MeshData

class ObjMeshInfo:
    mesh_data: MeshData
    materials: list[ObjMaterialInfo]
    subsets: list[ObjMeshSubsetInfo]
    primary_material_index: int
    material_count: int
    subset_count: int

def load_obj(path: str) -> MeshData: ...
def load_obj_with_materials(path: str) -> ObjMeshInfo: ...
def load_stl(path: str) -> MeshData: ...
def load_heightmap_terrain(
    path: str,
    up_axis: UpAxis = UpAxis.Y,
    horizontal_scale: float = 1.0,
    height_scale: float = 64.0,
    height_offset: float = -16.0,
    sample_stride: int = 1,
) -> MeshData: ...
def height_field_to_mesh(
    heights: npt.ArrayLike,
    up_axis: UpAxis = UpAxis.Y,
    horizontal_scale: float = 1.0,
    center: bool = True,
) -> MeshData: ...
