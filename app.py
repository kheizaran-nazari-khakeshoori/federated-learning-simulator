"""Federated Learning Simulator - entry point (thin wrapper, logic in separate modules)."""
import tkinter as tk
from ui.left_panel import create_left_panel
from ui.right_panel import create_right_panel
from core.engine import Engine

def main():
    root = tk.Tk()
    root.title("Federated Learning Simulator")
    root.geometry("1000x600")
    root.minsize(800, 500)
    root.configure(bg="#f0f0f0")

    left = create_left_panel(root)    # -> ui/left_panel.py
    right = create_right_panel(root)  # -> ui/right_panel.py (chart+clients+logs)

    engine = Engine(root, left, right)  # -> core/engine.py (real CNN + FedAvg/Prox/Adam)

    left["start_btn"].config(command=engine.start)
    left["stop_btn"].config(command=engine.stop)

    root.mainloop()

if __name__ == "__main__":
    main()
