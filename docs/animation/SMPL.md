# SMPL Models

`ke.asset` contains SMPL, SMPL-H, and SMPL-X model assets and their body-shape
evaluation. Renderable surfaces live under `ke.visual`; see
[Pose Visualization](VISUALS.md)
for the boundary between deformation producers and their visual objects.

`ke.asset.SMPLModel` adapts an SMPL NPZ or PKL file to
`ke.visual.SkinnedSurface`.
Shape coefficients are baked once and each pose is supplied as WXYZ local joint
rotations plus a root translation.

## SMPL model files

Repository examples select a neutral, male, or female model from
`assets/external/smpl_models`. Only the selected model file is needed.

```text
assets/
└── external/
    └── smpl_models/
        ├── smpl/
        │   ├── SMPL_NEUTRAL.pkl
        │   ├── SMPL_MALE.pkl
        │   └── SMPL_FEMALE.pkl
        ├── smplh/
        │   ├── neutral/model.npz
        │   ├── male/model.npz
        │   └── female/model.npz
        └── smplx/
            ├── SMPLX_NEUTRAL.npz
            ├── SMPLX_MALE.npz
            └── SMPLX_FEMALE.npz
```

### SMPL

```python
from pathlib import Path

import kangengine as ke

model_path: Path = ke.asset.smpl.repository_smpl_model_path(gender="neutral")
model: ke.asset.SMPLModel = ke.asset.SMPLModel.load(path=model_path)
```

### SMPL-H

```python
from pathlib import Path

import kangengine as ke

model_path: Path = ke.asset.smpl.repository_smplh_model_path(gender="neutral")
model: ke.asset.SMPLHModel = ke.asset.SMPLHModel.load(path=model_path)
```

### SMPL-X

```python
from pathlib import Path

import kangengine as ke

model_path: Path = ke.asset.smpl.repository_smplx_model_path(gender="neutral")
model: ke.asset.SMPLXModel = ke.asset.SMPLXModel.load(path=model_path)
```

Applications outside the repository can pass their own model path to the same
`load()` methods. Replace the bracketed path below with the location of the
downloaded model file:

```python
model: ke.asset.SMPLModel = ke.asset.SMPLModel.load(
    path="[smpl_path/SMPL_NEUTRAL.pkl]"
)
```

## Create and pose a body

```python
body: ke.asset.SMPLBody | ke.asset.SMPLHBody | ke.asset.SMPLXBody = (
    model.create_body(betas=betas)
)
surface: ke.visual.SkinnedSurface = body.create_visual(
    app=self,
    path="/character",
    material=material,
)

surface.apply_pose(
    root_translation=root_translation,
    local_rotations_wxyz=local_rotations_wxyz,
)
```

For AMASS playback, `AMASSLoader.load_motion()` returns a native
`SkeletonMotion` for the selected SMPL, SMPL-H, or SMPL-X skeleton:

```bash
python python/examples/view_smpl_motion.py --motion /path/to/amass_motion.npz
```

AMASS archives are Z-up. The loader keeps Z-up when requested, while the SMPL
viewer asks it to convert root translation and orientation to Y-up.

`SMPLBody.update_pose_correctives(surface, state)` enables the hybrid path: the
SMPL adapter evaluates pose-dependent blend shapes (pose correctives) on the
CPU before `surface.apply_state(state)`. It computes pose offsets from
`posedirs × (R - I)`, adds them to the shaped template vertices before
skinning, and updates the vertex and normal streams.
Linear blend skinning (LBS) still runs on the GPU. This path is opt-in because
recalculating vertices and normals is substantially more expensive than
uploading only the joint matrix palette. The default path remains matrix-only
GPU LBS. The same process applies to `SMPLHBody` and `SMPLXBody` with their
larger joint palettes. See the
{doc}`Asset API Reference <../../api/asset>` for the exact model and body class
signatures.
