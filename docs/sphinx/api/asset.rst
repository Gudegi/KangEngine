ke.asset
========

Loaders, model assets, and result types for MJCF, BVH, FBX, USD, AMASS,
SMPL, SMPL-H, and SMPL-X.

.. currentmodule:: kangengine.asset

API overview
------------

.. autosummary::
   :nosignatures:

   MJCFLoader
   BVHLoader
   FBXLoader
   USDLoader
   AMASSLoader
   AMASSInfo
   ArticulationDesc
   JointDesc
   SiteDesc
   InertialDesc
   VisualGeomDesc
   CollisionGeomDesc
   SMPLModel
   SMPLBody
   SMPLHModel
   SMPLHBody
   SMPLXModel
   SMPLXBody
   ImportDiagnostics


Loader return contracts
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - API
     - Return
   * - ``MJCFLoader.parse()`` / ``BVHLoader.parse()``
     - An import-result value containing the parsed payload and diagnostics.
   * - ``MJCFLoader.load()`` / ``BVHLoader.load()``
     - The commonly used character or motion payload without the result wrapper.
   * - ``FBXLoader.load()``
     - ``FBXImportResult`` containing meshes, animation clips, materials, and
       diagnostics. Mesh payloads retain shared native storage.
   * - ``USDLoader.parse()``
     - ``USDImportResult`` containing imported mesh records and diagnostics.
   * - ``AMASSLoader.inspect()``
     - ``AMASSInfo`` containing gender, shape coefficients, frame rate, and
       frame count without constructing a motion.
   * - ``AMASSLoader.load_motion()``
     - A native ``SkeletonMotion`` mapped onto a supplied SMPL, SMPL-H, or SMPL-X
       ``SkeletonTree``. AMASS Z-up data can be returned as Z-up or converted
       to KangEngine's usual Y-up coordinates.

Missing or unreadable files and malformed source data raise ``RuntimeError``;
invalid argument values raise ``ValueError`` where validation is available.
Inspect ``ImportDiagnostics`` for supported non-fatal import warnings.

.. autoclass:: ImportDiagnostics

.. autoclass:: ArticulationDesc

.. autoclass:: JointDesc

.. autoclass:: SiteDesc

.. autoclass:: InertialDesc

.. autoclass:: VisualGeomDesc

.. autoclass:: CollisionGeomDesc

.. autoclass:: MJCFLoader

.. autoclass:: MJCFImportResult

.. autoclass:: BVHLoader

.. autoclass:: BVHImportResult

.. autoclass:: FBXLoader

.. autoclass:: FBXImportResult

.. autoclass:: FBXAnimationClipInfo

.. autoclass:: FBXMaterialInfo

.. autoclass:: FBXMeshMetadata

.. autoclass:: FBXMeshInfo

.. autoclass:: FBXSkinnedMeshInfo

.. autoclass:: FBXCharacterData

.. autoclass:: USDLoader

.. autoclass:: USDImportResult

.. autoclass:: USDMeshInfo

.. autoclass:: AMASSInfo

.. autoclass:: AMASSLoader
   :members:

.. autoclass:: SMPLModel

.. autoclass:: SMPLBody

.. autoclass:: SMPLHModel

.. autoclass:: SMPLHBody

.. autoclass:: SMPLXModel

.. autoclass:: SMPLXBody
