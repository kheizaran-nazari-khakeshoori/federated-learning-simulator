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

    # Bottom: Log console (enhanced)
    log_frame = tk.Frame(right_panel, bg="white", height=160, relief=tk.SOLID, bd=1)
    log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
    log_frame.pack_propagate(False)

    # Log header toolbar
    log_header = tk.Frame(log_frame, bg="white")
    log_header.pack(fill=tk.X, padx=10, pady=(6, 2))
    tk.Label(log_header, text="Logs", bg="white", fg="#2c3e50", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
    tk.Label(log_header, text="● live", bg="white", fg="#27ae60", font=("Arial", 8)).pack(side=tk.LEFT, padx=(8, 0))

    # Auto-scroll var
    autoscroll_var = tk.BooleanVar(value=True)
    tk.Checkbutton(log_header, text="Auto-scroll", variable=autoscroll_var, bg="white", fg="#7f8c8d", font=("Arial", 8), selectcolor="white", activebackground="white").pack(side=tk.RIGHT, padx=5)

    def clear_logs():
        log_text.config(state=tk.NORMAL)
        log_text.delete("1.0", tk.END)
        log_text.config(state=tk.DISABLED)

    def save_logs():
        content = log_text.get("1.0", tk.END)
        try:
            with open("federated_logs.txt", "w") as f:
                f.write(content)
            log("Logs saved to federated_logs.txt", "SUCCESS")
        except Exception as e:
            log(f"Save failed: {e}", "ERROR")

    tk.Button(log_header, text="Save", command=save_logs, bg="#ecf0f1", fg="#2c3e50", font=("Arial", 8), relief=tk.FLAT, padx=8, pady=2).pack(side=tk.RIGHT, padx=2)
    tk.Button(log_header, text="Clear", command=clear_logs, bg="#ecf0f1", fg="#2c3e50", font=("Arial", 8), relief=tk.FLAT, padx=8, pady=2).pack(side=tk.RIGHT, padx=2)

    # Separator
    tk.Frame(log_frame, bg="#ecf0f1", height=1).pack(fill=tk.X, padx=10, pady=2)

    # Text area + scrollbar
    log_body = tk.Frame(log_frame, bg="white")
    log_body.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

    text_scroll = tk.Scrollbar(log_body)
    text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    log_text = tk.Text(log_body, height=6, bg="#1e2a33", fg="#ecf0f1", font=("Consolas", 9), relief=tk.FLAT,
                       yscrollcommand=text_scroll.set, padx=8, pady=5, insertbackground="white", wrap=tk.WORD)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    text_scroll.config(command=log_text.yview)

    # Tag colors for log levels
    log_text.tag_config("INFO", foreground="#3498db")
    log_text.tag_config("SUCCESS", foreground="#2ecc71")
    log_text.tag_config("ROUND", foreground="#f1c40f")
    log_text.tag_config("CLIENT", foreground="#e67e22")
    log_text.tag_config("ERROR", foreground="#e74c3c")
    log_text.tag_config("TIME", foreground="#95a5a6")

    import datetime

    def log(msg, level="INFO"):
        """Append colored log line with timestamp."""
        log_text.config(state=tk.NORMAL)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        log_text.insert(tk.END, f"[{ts}] ", "TIME")
        log_text.insert(tk.END, f"[{level}] ", level)
        log_text.insert(tk.END, f"{msg}\n")
        log_text.config(state=tk.DISABLED)
        if autoscroll_var.get():
            log_text.see(tk.END)

    # Initial logs
    log("Simulator ready. Configure left panel and press Start.", "INFO")
    log("Tip: Logs will show each round & client update here.", "INFO")

    # Wire left panel buttons to logs (demo)
    def on_start():
        status_var.set("Training...")
        status_lbl.config(bg="#3498db", fg="white")
        log(f"Starting training: {clients_var.get()} clients, {rounds_var.get()} rounds, {dataset_var.get()}, {agg_var.get()}", "INFO")
        # Demo: simulate 2 rounds of fake logs
        for r in range(1, 3):
            log(f"--- Round {r}/{rounds_var.get()} ---", "ROUND")
            for c in range(int(clients_var.get())):
                log(f"Client {c+1} training (epochs={epochs_var.get()})... acc=0.{80+r+c}%", "CLIENT")
            log(f"Aggregated with {agg_var.get()} - global accuracy updated", "SUCCESS")
        log("Demo finished. Wire your real FedAvg loop here.", "SUCCESS")
        status_var.set("Idle")
        status_lbl.config(bg="#f1c40f", fg="#2c3e50")
        accuracy_var.set("84.30%")

    def on_stop():
        log("Training stopped by user.", "ERROR")
        status_var.set("Stopped")
        status_lbl.config(bg="#e74c3c", fg="white")

    start_btn.config(command=on_start)
    stop_btn.config(command=on_stop)

    root.mainloop()

if __name__ == "__main__":
    main()
