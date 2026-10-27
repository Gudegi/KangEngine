# Foot Contact

`ke.animation.FootContactDetector` estimates planted contact points from their
distance to a surface and their speed. It accepts one sample at a time and works
with Motion Matching, ordinary playback, or robot poses. It does not require a
physics world or a complete motion clip.

This is a motion-based estimate, not a measured collision or force. A fast
sliding foot can be classified as non-contact. Foot locking, terrain adaptation,
and IK correction are separate steps and are not applied by this detector.

## Detect contacts

Choose points on the feet, such as toes and heels. Use their world positions in
meters; ankle joint origins may need local offsets to reach the sole. Compute
the world points after pose corrections such as joint-limit clamping.

```python
import numpy as np
import kangengine as ke

detector: ke.animation.FootContactDetector = ke.animation.FootContactDetector()

# These two samples contain the same points, in the same order.
previous_positions = np.array([[0.0, 0.1, 0.02], [0.0, -0.1, 0.15]])
positions = np.array([[0.0, 0.1, 0.02], [0.02, -0.1, 0.16]])
dt = 1 / 60
contacts: np.ndarray = detector.update(
    positions=positions,
    velocities=(positions - previous_positions) / dt,
)
print(contacts)  # [ True False ]
```

The default surface is a Z-up plane at zero height. For Y-up, pass
`ground_normals=(0, 1, 0)`. Pass `ground_points` to place the plane elsewhere.

Contact starts below `height_threshold=0.045` meters and
`velocity_threshold=0.35` m/s. Once active, the looser `release_height=0.065`
and `release_velocity=0.5` limits reduce flicker. Speed includes all three axes.
The limits describe probe points, so tune them for the selected sole offsets.
Negative signed distances count as below the surface; penetration is not corrected.

Update once per new output pose using its actual `dt`. Keep the first sample as
a baseline until a second sample is available; do not infer rest from a made-up
zero velocity. Pause updates during paused playback. On reset or teleport, call
`detector.reset()` and replace the previous-position baseline. Ordinary clip
transitions retain contact state.

## Track skeleton poses

`ke.animation.FootContactModule` selects joint-local probes, stores their previous
world positions, and computes velocities from successive output poses. It works
without a Motion Editor, MM controller, or renderer.

This snippet assumes two successive `ke.animation.SkeletonState` poses:
`initial_state` is the baseline and `current_state` is the next pose, separated
by `dt` seconds. Obtain them from your playback or the
[Motion Matching controller](MOTION_MATCHING.md). Set `foot_joint_names` to
your selected joint names and `sole_offsets` to their corresponding `[K, 3]`
joint-local offsets in meters, in the same order.

```python
# Use joint names from your skeleton; repeated names allow heel/toe probes.
module: ke.animation.FootContactModule = ke.animation.FootContactModule(
    skeleton_tree=initial_state.skeleton_tree,
    joints=foot_joint_names,
    offsets=sole_offsets,  # [K,3], joint-local meters
)
module.reset(state=initial_state)
contacts = module.update(state=current_state, dt=dt)
positions = module.positions  # [K,3], world meters
```

Pass the final pose after joint-limit or other pose corrections. Without an
initial baseline, the first update stores positions and returns false flags.
Use a positive `dt` between output poses and skip updates while paused. Call
`reset(state=...)` after a teleport. Treat stored results as read-only.
To tune thresholds, pass a `FootContactDetector` through `detector=`.

## Use terrain query results

Supply a point on the surface and a unit normal for each contact point. These
arrays have shape `[K,3]`; a common plane can instead use `[3]` values. The
detector measures `dot(position - ground_point, ground_normal)`, so slopes and
different foot heights use the same logic.

```python
contacts = detector.update(
    positions=foot_positions,
    velocities=foot_velocities,
    ground_points=hit_positions,
    ground_normals=hit_normals,
    supported=hit_mask,
)
```

`supported` is a Boolean `[K]` mask. A missing surface immediately releases
contact; missed query points/normals may be NaN. Terrain queries remain the
caller's responsibility. The detector assumes stationary support; on a moving
platform, supply velocities relative to the platform at each probe point.
`FootContactModule.update()` accepts the same ground arguments for stationary
terrain; use the detector directly when supplying support-relative velocities.

## View Motion Matching contacts

Run the [LAFAN or G1 example](MOTION_MATCHING.md) and enable **Show foot contacts**.
Green points indicate contact; red points indicate no contact. Display is off by
default. LAFAN tracks toe-base joints; G1 uses heel/toe sole probes after joint
limits are applied. Both examples currently query a flat ground plane.
