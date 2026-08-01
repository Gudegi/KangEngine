# kangengine

This page documents application-wide entry points and helpers exported directly
by the `kangengine` package. With the conventional
`import kangengine as ke` alias, these are accessed as `ke.App`,
`ke.DebugGeometry`, and similar top-level names.

Specialized APIs are grouped by responsibility under domain modules such as
`ke.scene`, `ke.physics`, `ke.material`, and `ke.sim`. Use the documented
public paths regardless of whether an object is implemented in C++ or Python.

Use the API Reference navigation to jump to a specific area.

```{include} app.md
:heading-offset: 1
```
