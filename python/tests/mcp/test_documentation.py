"""Keep public documentation lookup bounded and confined to its configured guide."""

import pytest

from kangengine.adapters.mcp import documentation
from kangengine.app.capture import AutomationError


def test_pages_search_pagination_and_symlink_confinement(tmp_path, monkeypatch):
    root = tmp_path / "guide"
    root.mkdir()
    (root / "page.md").write_text("First match\nsecond match\nthird\n")
    secret = tmp_path / "private.md"
    secret.write_text("hidden match")
    (root / "link.md").symlink_to(secret)
    monkeypatch.setattr(documentation, "GUIDE_ROOT", root)
    result = documentation.search_docs(query="MATCH", max_results=1)
    assert result["truncated"] and result["results"][0]["line"] == 1
    assert len(documentation.search_docs(query="match")["results"]) == 2
    page = documentation.get_doc(path="page.md", start_line=2, max_lines=1)
    assert page["text"] == "second match" and page["truncated"]
    for path in ["../private.md", str(secret), "link.md"]:
        with pytest.raises(AutomationError):
            documentation.get_doc(path=path)


def test_wheel_without_guide_still_has_public_api_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(documentation, "GUIDE_ROOT", tmp_path / "missing")
    with pytest.raises(AutomationError, match="source checkout"):
        documentation.search_docs(query="mesh")
    assert (
        "width" in documentation.get_api_docs(symbol="ke.App.initialize")["signature"]
    )
    assert (
        "box"
        in documentation.get_api_docs(symbol="geometry.create_box_data")[
            "documentation"
        ]
    )


@pytest.mark.parametrize(
    "symbol", ["App.__init__", "os.system", "App()", "_core", "App.missing"]
)
def test_lookup_cannot_execute_or_access_private_symbols(symbol):
    with pytest.raises(AutomationError):
        documentation.get_api_docs(symbol=symbol)
