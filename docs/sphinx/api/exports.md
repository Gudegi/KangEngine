# ke.exports

Export utilities for converting KangEngine's internal `SkeletonMotion`
representation into common file formats. These lightweight serialization
helpers can be used independently of interactive applications.

## BVH

The generated BVH uses XYZ root position channels and ZYX local rotation
channels. Joint names and hierarchy offsets come from the motion's
`SkeletonTree`. Leaf joints have no `End Site` block.

### `save_motion_bvh`

```{eval-rst}
.. autofunction:: kangengine.exports.bvh.save_motion_bvh
```

```python
motion: ke.animation.SkeletonMotion = ke.asset.BVHLoader.load_motion(
    bvh_path="input.bvh",
)
ke.exports.save_motion_bvh(path="output.bvh", motion=motion)
```

`save_motion_bvh` is the only public BVH export function. Text/byte conversion
helpers remain private implementation details. SMPL and SMPL-X motion export is
not currently provided.
