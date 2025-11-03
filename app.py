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

    # --- Right Panel (Main Content) ---
    right_panel = tk.Frame(root, bg="#ecf0f1")
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Top: Header / Status bar
    header = tk.Frame(right_panel, bg="white", height=50)
    header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))
    header.pack_propagate(False)
    tk.Label(header, text="Federated Learning - Global Model", bg="white", fg="#2c3e50", font=("Arial", 13, "bold")).pack(side=tk.LEFT, padx=15)
    status_var = tk.StringVar(value="Idle")
    status_lbl = tk.Label(header, textvariable=status_var, bg="#f1c40f", fg="#2c3e50", font=("Arial", 9, "bold"), padx=10, pady=4)
    status_lbl.pack(side=tk.RIGHT, padx=15)

    # Middle: Visualization area (split into Server and Clients)
    viz_frame = tk.Frame(right_panel, bg="#ecf0f1")
    viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    # Server card (top of viz)
    server_card = tk.Frame(viz_frame, bg="white", relief=tk.FLAT, bd=1)
    server_card.pack(fill=tk.X, pady=(0, 10))
    tk.Label(server_card, text="SERVER", bg="white", fg="#7f8c8d", font=("Arial", 9, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
    server_inner = tk.Frame(server_card, bg="white")
    server_inner.pack(fill=tk.X, padx=12, pady=(0, 10))
    # Accuracy bar
    tk.Label(server_inner, text="Global Accuracy:", bg="white", fg="#2c3e50", font=("Arial", 10)).pack(side=tk.LEFT)
    accuracy_var = tk.StringVar(value="0.00%")
    tk.Label(server_inner, textvariable=accuracy_var, bg="white", fg="#27ae60", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
    tk.Label(server_inner, text="Round: 0 / 10", bg="white", fg="#95a5a6", font=("Arial", 10)).pack(side=tk.RIGHT)
    # Canvas placeholder for chart
    chart_canvas = tk.Canvas(viz_frame, bg="white", height=180, highlightthickness=0)
    chart_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    chart_canvas.create_text(370, 90, text="Accuracy Chart (Rounds vs Accuracy) - will plot here", fill="#bdc3c7", font=("Arial", 11, "italic"))
    # Simple axes
    chart_canvas.create_line(50, 150, 700, 150, fill="#bdc3c7")  # x-axis
    chart_canvas.create_line(50, 20, 50, 150, fill="#bdc3c7")   # y-axis

    # Clients grid
    clients_frame = tk.Frame(viz_frame, bg="white")
    clients_frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(clients_frame, text="CLIENTS", bg="white", fg="#7f8c8d", font=("Arial", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 5))
    grid = tk.Frame(clients_frame, bg="white")
    grid.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)
    # Create 5 placeholder client boxes
    for i in range(5):
        card = tk.Frame(grid, bg="#f8f9fa", relief=tk.SOLID, bd=1)
        card.grid(row=i//3, column=i%3, padx=5, pady=5, sticky="nsew")
        tk.Label(card, text=f"Client {i+1}", bg="#f8f9fa", fg="#2c3e50", font=("Arial", 10, "bold")).pack(pady=(8, 2))
        tk.Label(card, text="acc: --", bg="#f8f9fa", fg="#7f8c8d", font=("Arial", 9)).pack()
        tk.Label(card, text="● idle", bg="#f8f9fa", fg="#95a5a6", font=("Arial", 8)).pack(pady=(2, 8))
    for c in range(3):
        grid.columnconfigure(c, weight=1)

    # Bottom: Log console
    log_frame = tk.Frame(right_panel, bg="white", height=140)
    log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
    log_frame.pack_propagate(False)
    tk.Label(log_frame, text="Logs", bg="white", fg="#7f8c8d", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
    text_scroll = tk.Scrollbar(log_frame)
    text_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
    log_text = tk.Text(log_frame, height=6, bg="#2c3e50", fg="#ecf0f1", font=("Consolas", 9), relief=tk.FLAT, yscrollcommand=text_scroll.set, padx=8, pady=5)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
    text_scroll.config(command=log_text.yview)
    log_text.insert(tk.END, "[INFO] Simulator ready. Configure left panel and press Start.\n")
    log_text.config(state=tk.DISABLED)

    root.mainloop()

if __name__ == "__main__":
    main()
