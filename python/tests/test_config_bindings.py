from __future__ import annotations

import unittest

import kangengine as ke


class ConfigBindingsTest(unittest.TestCase):
    def test_physics_config_keyword_construction(self):
        gpu_dynamics = ke.physics.PhysicsGpuDynamicsConfig(
            max_num_partitions=16,
        )
        config = ke.physics.PhysicsConfig(
            dt=1.0 / 120.0,
            enable_gpu=True,
            gpu_dynamics=gpu_dynamics,
            enable_contact_reports=False,
            cpu_dispatcher_threads=2,
        )

        self.assertAlmostEqual(config.dt, 1.0 / 120.0)
        self.assertTrue(config.enable_gpu)
        self.assertEqual(config.gpu_dynamics.max_num_partitions, 16)
        self.assertFalse(config.enable_contact_reports)
        self.assertEqual(config.cpu_dispatcher_threads, 2)
        self.assertIn("enable_gpu=True", repr(config))
        self.assertIn("max_num_partitions=16", repr(config))

    def test_articulation_and_gpu_config_keyword_construction(self):
        articulation = ke.physics.ArticulationConfig(
            fix_base=False,
            solver_position_iteration_count=32,
            root_angular_damping=0.1,
            enable_ccd=True,
        )
        gpu = ke.physics.GpuPhysicsConfig(
            cuda_device_id=1,
            max_contact_pairs=123,
            max_contact_points=456,
        )

        self.assertFalse(articulation.fix_base)
        self.assertEqual(articulation.solver_position_iteration_count, 32)
        self.assertAlmostEqual(articulation.root_angular_damping, 0.1)
        self.assertTrue(articulation.enable_ccd)
        self.assertEqual(gpu.cuda_device_id, 1)
        self.assertEqual(gpu.max_contact_pairs, 123)
        self.assertEqual(gpu.max_contact_points, 456)
        self.assertIn("fix_base=False", repr(articulation))
        self.assertIn("cuda_device_id=1", repr(gpu))

    def test_render_config_keyword_construction(self):
        sampler = ke.render.SamplerDesc(
            wrap_u=ke.render.TextureWrap.CLAMP_TO_EDGE,
            min_filter=ke.render.TextureFilter.LINEAR,
        )
        descriptor = ke.render.ExternalBufferDesc(
            count=4,
            stride_bytes=64,
            sync_policy=ke.render.ExternalSyncPolicy.VERSIONED,
        )

        self.assertEqual(sampler.wrap_u, ke.render.TextureWrap.CLAMP_TO_EDGE)
        self.assertEqual(sampler.min_filter, ke.render.TextureFilter.LINEAR)
        self.assertEqual(descriptor.count, 4)
        self.assertEqual(descriptor.stride_bytes, 64)
        self.assertEqual(
            descriptor.sync_policy,
            ke.render.ExternalSyncPolicy.VERSIONED,
        )
        self.assertIn("TextureWrap.CLAMP_TO_EDGE", repr(sampler))
        self.assertIn("sync_policy=ExternalSyncPolicy.VERSIONED", repr(descriptor))

    def test_keyword_only_and_default_construction(self):
        self.assertIsInstance(
            ke.physics.PhysicsConfig(),
            ke.physics.PhysicsConfig,
        )
        self.assertIsInstance(
            ke.render.SamplerDesc(),
            ke.render.SamplerDesc,
        )
        with self.assertRaises(TypeError):
            ke.physics.PhysicsConfig(1.0 / 120.0)
        with self.assertRaises(TypeError):
            ke.render.SamplerDesc(ke.render.TextureWrap.REPEAT)


    def test_skeletal_visual_config_keyword_construction(self):
        config = ke.visual.SkeletalVisualConfig(
            bone_radius=0.01,
            joint_radius=0.03,
            segments=12,
            visible=False,
            show_joints=False,
        )

        self.assertAlmostEqual(config.bone_radius, 0.01)
        self.assertAlmostEqual(config.joint_radius, 0.03)
        self.assertEqual(config.segments, 12)
        self.assertFalse(config.visible)
        self.assertFalse(config.show_joints)
        self.assertIn("bone_radius=0.01", repr(config))

if __name__ == "__main__":
    unittest.main()
