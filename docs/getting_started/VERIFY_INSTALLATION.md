# Verify Installation

Run in the environment where you installed KangEngine:

```bash
python -c "import kangengine as ke; print(ke.__file__); print('USD:', ke.scene.has_usd_support())"
python -m pip check
```

The PyPI wheel should report `USD: True`. The printed path should point to
your installed package. Source builds can disable USD.

## Open a window

Save and run the complete [Hello App](HELLO_APP.md) example. Expect an orange
cube on a ground plane.

## Common problems

- **No matching distribution:** check Python version, OS version, and CPU
  architecture against [Installation](INSTALLATION.md).
- **Import fails on Linux:** check that the NVIDIA driver meets the documented
  requirement and the PhysX GPU and CUDA runtime packages installed successfully.
- **Import works but no window opens:** check your graphical session and OpenGL
  support. A headless server needs a separate rendering setup.

Source contributors can use `make validate_python_api` from a checkout.
That command rebuilds the development extension; it is not needed to verify
a pip installation.

Next: [Hello App](HELLO_APP.md).
