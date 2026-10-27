# Changelog

## Unreleased

### Rigid body state

Rigid `num_bodies` is now always one. Code that used rigid `body_*` slots as
collider slots must use `get_collision_shape_pose()` for those reads instead;
even a single offset collider has a different pose from its body.
Mass, force, and contact state refer to the physical actor. See
[Collision Shapes](simulation/COLLISION_SHAPES.md) for shape poses and mappings.
