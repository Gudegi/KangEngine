# Contact Sensors

Contact sensors attach to a rigid or articulation simulation view. The current
GPU sensor reports contact count, contact mask, and accumulated normal impulse.

```python
rigids = world.get_rigid_batch(obj_id=0)
contact = rigids.add_contact_sensor(body_ids=[0], name="contact")
force = rigids.add_force_sensor(body_ids=[0], name="force")

world.step()

counts = contact.contact_count
in_contact = contact.in_contact
impulse = contact.net_impulse
average_normal_force = force.force
```

`ForceSensor.force` is `net_impulse / world.sim_dt`. It is not a full six-axis
wrench: tangential friction impulse and contact torque are not present.

The high-level GPU example keeps all sensor outputs as Torch CUDA tensors:

```bash
python ./python/examples/sim_gpu_contact_sensor.py --num-envs 8
python ./python/examples/sim_gpu_contact_sensor.py --num-envs 8 --viewer
```

The second command requires Linux/NVIDIA CUDA/OpenGL interop.

![GPU contact points and force arrows](../images/simulation/contact.png)
