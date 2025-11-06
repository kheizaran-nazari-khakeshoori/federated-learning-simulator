import tkinter as tk
from algorithms import get_algorithm

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
    round_var = tk.StringVar(value="Round: 0 / 10")
    tk.Label(server_inner, textvariable=round_var, bg="white", fg="#95a5a6", font=("Arial", 10)).pack(side=tk.RIGHT)
    # Canvas for chart (real FedAvg plot)
    chart_canvas = tk.Canvas(viz_frame, bg="white", height=180, highlightthickness=0)
    chart_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    def draw_chart(history, total_rounds):
        chart_canvas.delete("all")
        w = chart_canvas.winfo_width() or 700
        h = 180
        pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 30
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        # axes
        chart_canvas.create_line(pad_l, h-pad_b, w-pad_r, h-pad_b, fill="#bdc3c7")
        chart_canvas.create_line(pad_l, pad_t, pad_l, h-pad_b, fill="#bdc3c7")
        # y labels 0-100%
        for yv in [0, 50, 100]:
            y = h - pad_b - (yv/100)*plot_h
            chart_canvas.create_line(pad_l-4, y, pad_l, y, fill="#bdc3c7")
            chart_canvas.create_text(pad_l-8, y, text=f"{yv}%", anchor="e", fill="#95a5a6", font=("Arial", 7))
            chart_canvas.create_line(pad_l, y, w-pad_r, y, fill="#ecf0f1", dash=(2,2))
        # x labels
        if total_rounds > 0:
            for i in [0, total_rounds//2, total_rounds]:
                if total_rounds == 0: continue
                x = pad_l + (i/total_rounds)*plot_w if total_rounds else pad_l
                chart_canvas.create_text(x, h-pad_b+8, text=f"{i}", fill="#95a5a6", font=("Arial", 7))
            chart_canvas.create_text(w/2, h-8, text="Rounds", fill="#95a5a6", font=("Arial", 7))
        if not history:
            chart_canvas.create_text(w/2, h/2, text="Accuracy Chart (Rounds vs Accuracy) - waiting for training", fill="#bdc3c7", font=("Arial", 11, "italic"))
            return
        # plot line
        points = []
        for idx, acc in enumerate(history):
            x = pad_l + ((idx+1)/ max(1,total_rounds))*plot_w
            y = h - pad_b - (acc/100)*plot_h
            points.extend([x, y])
        if len(points) >= 4:
            chart_canvas.create_line(points, fill="#27ae60", width=2, smooth=True)
        for idx, acc in enumerate(history):
            x = pad_l + ((idx+1)/ max(1,total_rounds))*plot_w
            y = h - pad_b - (acc/100)*plot_h
            chart_canvas.create_oval(x-3, y-3, x+3, y+3, fill="#27ae60", outline="white")
            if idx == len(history)-1:
                chart_canvas.create_text(x, y-10, text=f"{acc:.1f}%", fill="#27ae60", font=("Arial", 7, "bold"))

    # Clients grid (dynamic)
    clients_frame = tk.Frame(viz_frame, bg="white")
    clients_frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(clients_frame, text="CLIENTS", bg="white", fg="#7f8c8d", font=("Arial", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 5))
    grid = tk.Frame(clients_frame, bg="white")
    grid.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)
    for c in range(3):
        grid.columnconfigure(c, weight=1)

    client_widgets = []  # list of {acc_var, status_var, card}

    def rebuild_clients(num):
        for w in grid.winfo_children():
            w.destroy()
        client_widgets.clear()
        cols = 3
        for i in range(num):
            card = tk.Frame(grid, bg="#f8f9fa", relief=tk.SOLID, bd=1)
            card.grid(row=i//cols, column=i%cols, padx=5, pady=5, sticky="nsew")
            tk.Label(card, text=f"Client {i+1}", bg="#f8f9fa", fg="#2c3e50", font=("Arial", 10, "bold")).pack(pady=(8, 2))
            acc_v = tk.StringVar(value="acc: --")
            st_v = tk.StringVar(value="● idle")
            acc_lbl = tk.Label(card, textvariable=acc_v, bg="#f8f9fa", fg="#7f8c8d", font=("Arial", 9))
            acc_lbl.pack()
            st_lbl = tk.Label(card, textvariable=st_v, bg="#f8f9fa", fg="#95a5a6", font=("Arial", 8))
            st_lbl.pack(pady=(1, 3))
            # progress bar for dummy loop visualization
            bar_bg = tk.Frame(card, bg="#e0e0e0", height=6)
            bar_bg.pack(fill=tk.X, padx=10, pady=(2, 8))
            bar_bg.pack_propagate(False)
            bar_fill = tk.Frame(bar_bg, bg="#27ae60")
            bar_fill.place(relwidth=0, relheight=1)
            client_widgets.append({"acc_var": acc_v, "status_var": st_v, "card": card, "acc_lbl": acc_lbl, "st_lbl": st_lbl, "bar": bar_fill, "bar_bg": bar_bg})

    rebuild_clients(5)

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

    # --- FedAvg Simulation Engine (now using 3 model files) ---
    import random

    sim_state = {"running": False, "history": [], "global_acc": 10.0}

    def fedavg_simulate():
        """Real FedAvg loop wired to left panel."""
        if sim_state["running"]:
            log("Training already running.", "ERROR")
            return
        try:
            n_clients = int(clients_var.get())
            n_rounds = int(rounds_var.get())
            n_epochs = int(epochs_var.get())
        except ValueError:
            log("Invalid numeric config.", "ERROR")
            return
        if not (2 <= n_clients <= 100 and 1 <= n_rounds <= 1000):
            log("Check clients (2-100) and rounds (1-1000).", "ERROR")
            return

        dataset = dataset_var.get()
        agg = agg_var.get()
        # dataset difficulty factor
        diff = {"MNIST": 1.0, "CIFAR-10": 0.75, "Fashion-MNIST": 0.85, "Synthetic": 1.1}.get(dataset, 1.0)
        algo = get_algorithm(agg)  # -> algorithms/fedavg.py | fedprox.py | fedadam.py
        if hasattr(algo, "reset"):
            algo.reset()

        sim_state["running"] = True
        sim_state["history"] = []
        sim_state["global_acc"] = random.uniform(12, 20)  # start low
        rebuild_clients(n_clients)
        accuracy_var.set(f"{sim_state['global_acc']:.2f}%")
        round_var.set(f"Round: 0 / {n_rounds}")
        draw_chart([], n_rounds)
        for cw in client_widgets:
            cw["acc_var"].set("acc: --")
            cw["status_var"].set("● idle")
            cw["st_lbl"].config(fg="#95a5a6")

        status_var.set(f"Training ({agg})...")
        status_lbl.config(bg="#3498db", fg="white")
        start_btn.config(state=tk.DISABLED)
        log(f"Starting {algo.name}: {n_clients} clients, {n_rounds} rounds, {n_epochs} epochs, {dataset}, {algo.name}", "INFO")
        log(f"Initial global accuracy: {sim_state['global_acc']:.2f}%", "INFO")
        log(f"Loaded model: algorithms/{agg.lower()}.py", "INFO")

        # Dummy-loop style: animate each client sequentially, then aggregate + update chart/logs
        def run_round(r_idx):
            if not sim_state["running"]:
                return
            if r_idx > n_rounds:
                finish()
                return

            round_var.set(f"Round: {r_idx} / {n_rounds}")
            log(f"--- Round {r_idx}/{n_rounds} ---", "ROUND")
            # reset bars
            for cw in client_widgets:
                cw["bar"].place(relwidth=0, relheight=1)
                cw["status_var"].set("● training")
                cw["st_lbl"].config(fg="#e67e22")
                cw["card"].config(bg="#fef9e7")

            client_accs = []
            # animate clients one-by-one (dummy loop visualization)
            def animate_client(idx):
                if not sim_state["running"]:
                    return
                if idx >= len(client_widgets):
                    # all clients done -> aggregate via selected algorithm file
                    new_global = algo.aggregate(client_accs, sim_state["global_acc"])
                    sim_state["global_acc"] = new_global
                    sim_state["history"].append(new_global)
                    accuracy_var.set(f"{new_global:.2f}%")
                    draw_chart(sim_state["history"], n_rounds)
                    log(f"Aggregated ({algo.name}) -> global accuracy: {new_global:.2f}%", "SUCCESS")
                    for cw in client_widgets:
                        cw["card"].config(bg="#f8f9fa")
                    root.after(600, lambda: run_round(r_idx+1))
                    return

                cw = client_widgets[idx]
                # delegate to selected algorithm model file
                local_acc = algo.local_update(sim_state["global_acc"], n_epochs, diff)
                client_accs.append(local_acc)

                # update card
                cw["acc_var"].set(f"acc: {local_acc:.1f}%")
                cw["bar"].place(relwidth=min(1, local_acc/100), relheight=1)
                cw["bar"].config(bg="#27ae60" if local_acc > 70 else "#e67e22" if local_acc > 50 else "#e74c3c")
                log(f"Client {idx+1} local update: {local_acc:.1f}% (epochs={n_epochs}) [{algo.name}]", "CLIENT")
                cw["status_var"].set("● done")
                cw["st_lbl"].config(fg="#27ae60")

                root.after(180, lambda: animate_client(idx+1))

            animate_client(0)

        def finish():
            sim_state["running"] = False
            start_btn.config(state=tk.NORMAL)
            status_var.set("Completed")
            status_lbl.config(bg="#27ae60", fg="white")
            log(f"Training completed. Final accuracy: {sim_state['global_acc']:.2f}% over {len(sim_state['history'])} rounds.", "SUCCESS")
            for cw in client_widgets:
                cw["status_var"].set("● idle")
                cw["st_lbl"].config(fg="#95a5a6")

        # start loop
        root.after(300, lambda: run_round(1))

        # handle resize -> redraw chart
        def on_resize(event):
            if sim_state["history"]:
                draw_chart(sim_state["history"], n_rounds)
        chart_canvas.bind("<Configure>", on_resize)
        # initial draw
        root.after(100, lambda: draw_chart([], n_rounds))

    def on_stop():
        if sim_state["running"]:
            sim_state["running"] = False
            log("Training stopped by user.", "ERROR")
            status_var.set("Stopped")
            status_lbl.config(bg="#e74c3c", fg="white")
            start_btn.config(state=tk.NORMAL)
            for cw in client_widgets:
                if cw["status_var"].get() == "● training":
                    cw["status_var"].set("● stopped")
                    cw["st_lbl"].config(fg="#e74c3c")
        else:
            log("No training to stop.", "INFO")

    start_btn.config(command=fedavg_simulate)
    stop_btn.config(command=on_stop)

    root.mainloop()

if __name__ == "__main__":
    main()
