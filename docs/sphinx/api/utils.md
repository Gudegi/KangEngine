# ke.utils

Pure Python helpers for joint mapping, math, tensor conversion, colors, and
debug drawing.

## API overview

```{eval-rst}
.. currentmodule:: kangengine.utils

.. autosummary::
   :nosignatures:

   JointMapper
   normalize_joint_name
   as_tensor
   as_cpu_numpy
   as_sim_buffer
   preset_rgba
```

## Joint Mapping

```{eval-rst}
.. currentmodule:: kangengine.utils

.. autoclass:: JointSemantic

.. autoclass:: JointMapper

.. autofunction:: normalize_joint_name
```

## Math Helpers

```{eval-rst}
.. currentmodule:: kangengine.utils.math

.. autofunction:: normalize_vector

.. autofunction:: quat_xyzw_normalize

.. autofunction:: quat_xyzw_multiply

.. autofunction:: quat_xyzw_conjugate

.. autofunction:: quat_xyzw_rotate

.. autofunction:: quat_xyzw_from_two_vectors

.. autofunction:: quat_wxyz_to_xyzw

.. autofunction:: quat_wxyz_twist_angle
```

## Tensor Helpers

```{eval-rst}
.. currentmodule:: kangengine.utils.tensor

.. autofunction:: resolve_device

.. autofunction:: as_tensor

.. autofunction:: as_cpu_numpy

.. currentmodule:: kangengine.utils

.. autofunction:: as_sim_buffer
```

## Environment Helpers

```{eval-rst}
.. currentmodule:: kangengine.utils.env_utils

.. autofunction:: env_id_list

.. autofunction:: select_env_value

.. autofunction:: select_optional_env_value
```

## Color And Debug Helpers

```{eval-rst}
.. currentmodule:: kangengine.utils.color

.. autofunction:: preset_rgba

.. currentmodule:: kangengine.utils.debug_draw

.. autofunction:: log_debug_axes
```
