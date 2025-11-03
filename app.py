import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Federated Learning Simulator")
    root.geometry("1000x600")
    root.minsize(800, 500)
    root.configure(bg="#f0f0f0")

    # --- Left Panel ---
    left_panel = tk.Frame(root, bg="#2c3e50", width=260)
    left_panel.pack(side=tk.LEFT, fill=tk.Y)
    left_panel.pack_propagate(False)  # keep fixed width

    # Title in left panel
    title_label = tk.Label(left_panel, text="Controls", bg="#2c3e50", fg="white", font=("Arial", 16, "bold"))
    title_label.pack(pady=(20, 15), padx=10)

    separator = tk.Frame(left_panel, bg="#34495e", height=2)
    separator.pack(fill=tk.X, padx=15, pady=(0, 15))

    # Helper to create label + widget
    def add_section_label(text):
        lbl = tk.Label(left_panel, text=text, bg="#2c3e50", fg="#bdc3c7", font=("Arial", 9, "bold"), anchor="w")
        lbl.pack(fill=tk.X, padx=15, pady=(10, 2))
        return lbl

    # Number of Clients
    add_section_label("NUM CLIENTS")
    clients_var = tk.StringVar(value="5")
    clients_spin = tk.Spinbox(left_panel, from_=2, to=100, textvariable=clients_var, wrap=True, font=("Arial", 10), width=10)
    clients_spin.pack(padx=15, fill=tk.X, pady=(0, 5))

    # Number of Rounds
    add_section_label("ROUNDS")
    rounds_var = tk.StringVar(value="10")
    rounds_spin = tk.Spinbox(left_panel, from_=1, to=1000, textvariable=rounds_var, wrap=True, font=("Arial", 10), width=10)
    rounds_spin.pack(padx=15, fill=tk.X, pady=(0, 5))

    # Local Epochs
    add_section_label("LOCAL EPOCHS")
    epochs_var = tk.StringVar(value="3")
    epochs_spin = tk.Spinbox(left_panel, from_=1, to=50, textvariable=epochs_var, wrap=True, font=("Arial", 10), width=10)
    epochs_spin.pack(padx=15, fill=tk.X, pady=(0, 5))

    # Dataset
    add_section_label("DATASET")
    dataset_var = tk.StringVar(value="MNIST")
    dataset_menu = tk.OptionMenu(left_panel, dataset_var, "MNIST", "CIFAR-10", "Fashion-MNIST", "Synthetic")
    dataset_menu.config(bg="white", font=("Arial", 10), width=12)
    dataset_menu.pack(padx=15, fill=tk.X, pady=(0, 5))

    # Aggregation
    add_section_label("AGGREGATION")
    agg_var = tk.StringVar(value="FedAvg")
    agg_menu = tk.OptionMenu(left_panel, agg_var, "FedAvg", "FedProx", "FedAdam")
    agg_menu.config(bg="white", font=("Arial", 10), width=12)
    agg_menu.pack(padx=15, fill=tk.X, pady=(0, 5))

    # Spacer
    tk.Frame(left_panel, bg="#2c3e50", height=20).pack()

    # Start Button
    start_btn = tk.Button(left_panel, text="▶ Start Training", bg="#27ae60", fg="white", activebackground="#2ecc71",
                          font=("Arial", 11, "bold"), relief=tk.FLAT, padx=10, pady=8)
    start_btn.pack(padx=15, fill=tk.X, pady=5)

    stop_btn = tk.Button(left_panel, text="■ Stop", bg="#c0392b", fg="white", activebackground="#e74c3c",
                         font=("Arial", 10), relief=tk.FLAT, padx=10, pady=6)
    stop_btn.pack(padx=15, fill=tk.X, pady=(5, 10))

    # --- Main Content Area (Right) ---
    main_area = tk.Frame(root, bg="white")
    main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    main_label = tk.Label(main_area, text="Main Visualization Area", bg="white", fg="#7f8c8d", font=("Arial", 14))
    main_label.pack(expand=True)

    # Example placeholder box
    info = tk.Label(main_area, text="(Clients accuracy / Server model will be shown here)", bg="white", fg="#95a5a6", font=("Arial", 10, "italic"))
    info.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
