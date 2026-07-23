# Rigid Bodies

Load a rigid description, register it in one environment, and keep the returned
simulation view.

```python
world = ke.sim.KangSimWorld(num_envs=1, sim_dt=1.0 / 120.0, add_ground=True)

ball_data = world.load_mjcf(ball_xml)
ball = world.add_rigid(
    ball_data,
    env_id=0,
    obj_id=0,
    name="ball",
    pos=[0.0, 0.0, 1.8],
    density=600.0,
)
```

Reset position, rotation, and velocity through the same object view:

```python
ball.set_root_state(
    None,
    [0.0, 0.0, 1.8],
    [0.0, 0.0, 0.0, 1.0],  # quaternion xyzw
    linear_velocity=[0.0, 0.0, 0.0],
    angular_velocity=[0.0, 0.0, 0.0],
)
```

`None` selects every environment represented by the view. For one object in
one environment, that still means a batch of size one.

Complete example: `python/examples/sim_world_minimal.py`.
