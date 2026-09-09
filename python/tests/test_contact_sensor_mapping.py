from types import SimpleNamespace

import pytest

from kangengine.sim.sensor import ContactSensor


def sensor_with_maps(maps, body_ids):
    sensor = ContactSensor.__new__(ContactSensor)
    sensor._kind = 1
    sensor.obj_id = 0
    sensor.env_ids = tuple(range(len(maps)))
    sensor.body_ids = body_ids
    sensor.world = SimpleNamespace(
        articulations={
            (i, 0): SimpleNamespace(
                articulation=SimpleNamespace(get_link_indices=lambda m=m: m)
            )
            for i, m in enumerate(maps)
        }
    )
    return sensor


def test_public_body_order_maps_to_internal_physx_links():
    sensor = sensor_with_maps([(0, 3, 1, 2)] * 2, (2, 1))
    assert sensor._gpu_body_ids() == (1, 3)


def test_inconsistent_environment_maps_are_rejected():
    sensor = sensor_with_maps([(0, 2, 1), (0, 1, 2)], (1,))
    with pytest.raises(RuntimeError, match="maps differ"):
        sensor._gpu_body_ids()


def test_rigid_body_ids_are_not_remapped():
    sensor = sensor_with_maps([], (0,))
    sensor._kind = 0
    assert sensor._gpu_body_ids() == (0,)
