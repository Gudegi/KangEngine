"""Exercise NewtonViewer model replacement, external stepping, and teardown."""

from kangengine.adapters.newton import NewtonViewer


def _make_model(shape: str):
    import newton

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    body = builder.add_body(label=shape)
    if shape == "sphere":
        builder.add_shape_sphere(body, radius=0.5)
    elif shape == "box":
        builder.add_shape_box(body, hx=0.5, hy=0.4, hz=0.3)
    else:
        raise ValueError(f"unsupported smoke shape: {shape}")
    return builder.finalize()


def _exercise_viewer(model_a, model_b):
    viewer = NewtonViewer(width=320, height=240, headless=True)
    try:
        viewer.set_model(model_a)
        viewer.log_state(model_a.state())
        viewer.end_frame()
        if len(viewer._instances) != 1:
            raise RuntimeError("first Newton model was not registered")

        viewer.set_model(model_b)
        viewer.log_state(model_b.state())
        viewer.end_frame()
        if len(viewer._instances) != 1:
            raise RuntimeError("replacement Newton model retained stale batches")

        if not viewer.should_step():
            raise RuntimeError("running Newton viewer unexpectedly rejected a step")
        viewer.app.set_simulation_paused(True)
        if viewer.should_step():
            raise RuntimeError("paused Newton viewer unexpectedly accepted a step")
        viewer.app.request_simulation_step()
        viewer.app.render_frame_once()
        if not viewer.should_step() or viewer.should_step():
            raise RuntimeError("single-step request was not consumed exactly once")

        viewer.set_model(None)
        if viewer._instances or viewer._meshes:
            raise RuntimeError("clearing the Newton model retained render resources")
    finally:
        viewer.close()


def main():
    model_a = _make_model("sphere")
    model_b = _make_model("box")
    _exercise_viewer(model_a, model_b)
    _exercise_viewer(model_b, model_a)
    print("PASS: NewtonViewer model replacement and teardown")


if __name__ == "__main__":
    main()
