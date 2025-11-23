"""UI panel checks."""
from ui.left_panel import create_left_panel
from ui.right_panel import create_right_panel
def test_panels_import():
    assert callable(create_left_panel)
    assert callable(create_right_panel)

def test_left_vars():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    left = create_left_panel(root)
    assert "clients_var" in left and "alpha_var" in left
    root.destroy()
