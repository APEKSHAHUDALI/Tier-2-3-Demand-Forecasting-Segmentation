from pathlib import Path


def test_app_uses_streamlit_stretch_width():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert "use_container_width" not in source
    assert 'width="stretch"' in source
    assert "get_script_run_context" not in source
