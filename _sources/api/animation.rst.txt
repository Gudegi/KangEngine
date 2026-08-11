ke.animation
============

Skeleton hierarchies, motion clips, pose state, and skinning helpers.

Scene/render visual bridge classes are documented under
``kangengine.visual``. The native binding may still contain implementation
types for these bridges, but the public ``kangengine.animation`` module removes
them. Python code should use ``ke.visual.ArticulationVisual`` or
``ke.visual.SkeletalVisual`` for viewer-side objects.

.. currentmodule:: kangengine.animation

API overview
------------

.. autosummary::
   :nosignatures:

   SkeletonTree
   SkeletonMotion
   SkeletonState
   Transform
   cpu_skin

.. autofunction:: cpu_skin

.. autofunction:: compute_skinning_matrices

.. autofunction:: compute_skinning_matrices_into

.. autoclass:: SkeletonTree

.. autoclass:: SkeletonMotion
   :members:

``SkeletonMotion`` is the canonical public motion container. Importers such as
``BVHLoader`` and ``AMASSLoader`` return it directly; preprocessing code should
construct edited clips with ``SkeletonMotion.from_arrays()``. Batched FK,
linear/angular velocity, and acceleration arrays are computed through its
``global_*`` and ``root_*`` methods.

.. autoclass:: Transform

.. autoclass:: SkeletonState
