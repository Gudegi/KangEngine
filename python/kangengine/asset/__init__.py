"""Asset importers, model assets, and import result types."""

from __future__ import annotations

from .._core import _ke
from .._public import export_public_module

__all__ = export_public_module(_ke.asset, globals())

from . import amass, smpl
from .amass import AMASSInfo, AMASSLoader
from .rigid_asset import load_obj_rigid, rigid_from_terrain_mesh
from .smpl import SMPLBody, SMPLHBody, SMPLHModel, SMPLModel, SMPLXBody, SMPLXModel

__all__ += [
    "load_obj_rigid",
    "rigid_from_terrain_mesh",
    "AMASSInfo",
    "AMASSLoader",
    "amass",
    "SMPLBody",
    "SMPLModel",
    "SMPLHBody",
    "SMPLHModel",
    "SMPLXBody",
    "SMPLXModel",
    "smpl",
]
