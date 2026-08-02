ke.asset
========

Loaders and result types for MJCF, BVH, FBX, and USD assets.

.. currentmodule:: kangengine.asset

API overview
------------

.. autosummary::
   :nosignatures:

   MJCFLoader
   BVHLoader
   FBXLoader
   USDLoader
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

Missing or unreadable files and malformed source data raise ``RuntimeError``;
invalid argument values raise ``ValueError`` where validation is available.
Inspect ``ImportDiagnostics`` for supported non-fatal import warnings.

.. autoclass:: ImportDiagnostics

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
