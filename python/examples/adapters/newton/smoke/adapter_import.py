"""Check that the optional Newton adapter does not affect normal imports."""

from kangengine.adapters.newton import (
    NewtonUnavailableError,
    ViewerKE,
    is_newton_available,
)


def main():
    if is_newton_available():
        print("PASS: Newton adapter dependencies are available")
        return

    try:
        ViewerKE(headless=True)
    except NewtonUnavailableError:
        print("PASS: Newton adapter reports its missing optional dependency")
        return
    raise RuntimeError("ViewerKE unexpectedly initialized without Newton")


if __name__ == "__main__":
    main()
