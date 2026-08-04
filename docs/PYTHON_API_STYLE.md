# Python API naming policy

This policy applies to pybind declarations, Python facades, examples,
documentation, and stubs. Python names follow Python conventions independently
of the corresponding C++ identifier.

| API element | Required form | Example |
| --- | --- | --- |
| Class, enum, descriptor, config, info, result, state | `PascalCase` | `SceneHookPipelineDesc` |
| Function and method | `snake_case` | `create_scene_hook_pipeline()` |
| Field, property, argument | `snake_case` | `use_scene_frame_bindings` |
| Enum value | `UPPER_SNAKE_CASE` | `AFTER_TRANSPARENT` |
| Python protocol name | Protocol-defined | `__cuda_array_interface__` |

An adapter that must mirror a third-party protocol may retain that protocol's
spelling. Such exceptions must be named explicitly in the style checker;
`MimicControlMode` and `MimicObjType` mirror MimicKit's external contract.

`Desc`, `Config`, `Info`, `Result`, and `State` are class-name suffixes and
remain PascalCase. Their fields are snake_case. Math value types are classes
and use names such as `Vec3`, `Quat`, and `Mat4`.

C++ APIs retain KangEngine's camelCase convention. Bindings explicitly spell
the Python name. Python-overridable C++ virtual methods use
`PYBIND11_OVERRIDE_NAME` when the names differ.

Enum values remain qualified by their enum type. Do not use
`py::enum_::export_values()` or add naming-only compatibility aliases.

Run:

```bash
make check_python_api_style
```

`make validate_python_api` runs this check automatically.
