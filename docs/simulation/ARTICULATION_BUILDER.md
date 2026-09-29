# Building an articulation in Python

Use `ArticulationBuilder` to create a robot or mechanism from links and joints
without writing an MJCF or URDF file. It supports boxes and spheres connected
by hinge, slider, spherical, or fixed joints.

## Create a base and hinged arm

```python
import kangengine as ke

Builder = ke.physics.ArticulationBuilder
builder: ke.physics.ArticulationBuilder = Builder()
builder.add_link(
    "base", mass=2., inertia_diagonal=(.1, .1, .1),
    shapes=[Builder.BoxShape(half_extents=(.2, .2, .2))])
builder.add_link(
    "arm", parent="base", position=(0., 0., 1.),
    joint=Builder.RevoluteJoint(
        axis=(0., 1., 0.), joint_offset=(0., 0., -.5)),
    mass=1., inertia_diagonal=(.1, .1, .01),
    shapes=[Builder.BoxShape(half_extents=(.05, .05, .5))])

config = ke.physics.ArticulationConfig.fixed_base()
template: ke.physics.ArticulationTemplate = builder.build_template(config=config)
world: ke.physics.PhysicsWorld = ke.physics.PhysicsWorld()
robot: ke.physics.Articulation = ke.physics.Articulation.build_from_template(
    world, template, config=config)
```

This creates the physics objects; it does not open a viewer. The inertia values
are supplied explicitly. For an interactive example, see below.

Add the root first, then its children. Each parent must already exist, and link
names must be unique. Use `fixed_base()` to fix the root or `free_base()` to let
it move. Reuse the template to create multiple independent robots. You can also
pass `builder.build()` or its template to `KangSimWorld.add_articulation(...)`.

## Choose a joint

Joint and shape specifications belong to `ArticulationBuilder` (aliased as
`Builder` above); they describe the model before runtime objects are created.

| Configuration | Movement | Limits |
| --- | --- | --- |
| `Builder.RevoluteJoint(axis=...)` | One rotation axis | Radians |
| `Builder.PrismaticJoint(axis=...)` | One translation axis | Meters |
| `Builder.SphericalJoint()` | Three angular degrees of freedom | Three pairs, in radians |
| `Builder.FixedJoint()` or `joint=None` on a child | Fixed to parent | None |

Use `limits=(lower, upper)` for a hinge or slider. `kp` and `kd` set joint drive
gains. A root has no parent joint; its fixed/free setting comes from the
articulation config.

## Understand the coordinates

- Link `position` and `rotation_xyzw` place its origin relative to its parent
  when joint positions are zero.
- Joint `axis` and `joint_offset` are expressed in the child link's coordinates.
- Shape positions and `com` are also local to the link. Moving a shape or its
  center of mass does not move the link origin.
- `Builder.BoxShape.half_extents` are half the box dimensions, in meters.

`joint_offset` locates the joint relative to the child link origin. In the arm
example, the box is 1 m long and centered on its link origin. The hinge is at
its lower end, so `joint_offset=(0, 0, -.5)`. If you instead place the link
origin at the hinge and offset the shape from it, the joint offset can be zero.

Supply positive mass and diagonal inertia about `com`, with inertia axes aligned
to the link axes. The builder does not calculate mass or inertia from shapes.

## Try the scissor mechanism

```bash
python ./python/examples/procedural_articulation.py
```

Move the GUI slider to raise or lower the three-stage mechanism. It combines
one slider, six hinges, and three external D6 pivots.

An articulation's internal links form a tree: each child has one parent.
To close a loop, first build the articulation, then connect existing links with
[external D6 joints](D6_JOINTS.md).

## Current limits

- Define the topology before creating the articulation; the builder does not
  edit a running robot's topology.
- Authored collision shapes are boxes and spheres. Use asset importers for
  other supported geometry types.
- Automatic inertia calculation and rotated inertia frames are not exposed.
- Nonzero `joint_offset` is supported in physics, but `ArticulationMotionMapper`
  rejects it when converting to or from SkeletonState/SkeletonMotion. Those
  formats use fixed child translations and cannot represent these displaced
  pivots with the same body topology. Store joint motion as `ArticulationMotion`
  and read link positions from physics instead. External D6 frames do not
  impose this motion-conversion restriction.
