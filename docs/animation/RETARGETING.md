# Motion Retargeting

KangEngine retargets BVH or FBX motion between compatible skeletons. A
`*_retarget.json` config stores the reference skeletons, bind poses, joint
mapping, scale, and coordinate systems.

## Create a config

Launch the editor without arguments, load source and target references, adjust
their bind poses, map joints, preview the result, and export the config.

```bash
python \
  python/tools/animation/retarget_editor.py
```

The reference source only needs a skeleton and bind pose. Motions processed
later must use the same joint names and hierarchy.

Enable **Source has armature joint** when a BVH has a top-level container above
the actual skeleton root. KangEngine removes that container and promotes its
only child (for example `Hips`) to the motion root. The choice is stored as
`source_has_armature_joint` in the config and is also used by headless jobs.

## Headless conversion

Convert one motion without opening a window:

```bash
python \
  python/tools/animation/retarget_motion.py \
  --config character_retarget.json \
  --input walk.bvh \
  --output walk_kw.bvh
```

Convert a directory:

```bash
python \
  python/tools/animation/retarget_motion.py \
  --config character_retarget.json \
  --input-dir motions \
  --output-dir retargeted \
  --suffix _kw
```

## Python API

```python
from kangengine.animation import RetargetBatchProcessor

processor = RetargetBatchProcessor("character_retarget.json")
processor.process_file("walk.bvh", "walk_retargeted.bvh")
processor.process_files(["run.bvh", "jump.bvh"], "retargeted")
```

Input motions are converted to KangEngine's internal coordinate system before
retargeting and converted to the config's output coordinate system when saved.
The headless processor does not create an application, renderer, or window.
