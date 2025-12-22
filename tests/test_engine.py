"""Headless mock tk for engine tests."""
import tkinter as tk
class MockTk:
    def after(self, ms, func): func()
    def update_idletasks(self): pass

def test_single_round():
    from core.engine import Engine
    assert Engine is not None

# mock tk helper for headless
