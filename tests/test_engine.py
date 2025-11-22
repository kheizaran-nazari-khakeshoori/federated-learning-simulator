"""Headless mock tk for engine tests."""
import tkinter as tk
class MockTk:
    def after(self, ms, func): func()
    def update_idletasks(self): pass
