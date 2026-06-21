"""Animation, skeleton, skinning, and character bridge APIs."""
from __future__ import annotations

from ._core import _ke
from ._public import export_public_module

export_public_module(_ke.animation, globals())
