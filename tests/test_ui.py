"""UI panel checks."""
from ui.left_panel import create_left_panel
from ui.right_panel import create_right_panel
def test_panels_import():
    assert callable(create_left_panel)
    assert callable(create_right_panel)
