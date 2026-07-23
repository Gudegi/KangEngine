# First Simulation

This page creates one dynamic rigid body, steps PhysX, and synchronizes its
visual representation.

```python
self.standard_materials = self.create_standard_materials()
self.world = ke.sim.KangSimWorld(
    num_envs=1,
    sim_dt=1.0 / 120.0,
    add_ground=True,
)
self.visual = ke.visual.sim.SimWorldVisualizer(self, self.world)

ball_xml = package_asset_path("objects", "ball.xml")
ball_data = self.world.load_mjcf(ball_xml)
self.ball = self.world.add_rigid(
    ball_data,
    env_id=0,
    obj_id=0,
    name="ball",
    pos=[0.0, 0.0, 1.8],
    density=600.0,
)
self.visual.add(self.ball, ball_xml, material=self.standard_materials.common)
```

Advance and display the simulation once per frame:

```python
def preRender(self):
    self.world.step(substeps=2)
    self.visual.sync()
```

Run the complete example:

```bash
python ./python/examples/sim_world_minimal.py --width 1280 --height 720
```

Expected result: the ball falls onto the ground. Press Space to pause and `R`
to reset it.

![A rigid-body ball](../images/getting_started/first_simulation.png)

Next: [Simulation](../simulation/INDEX.md).
