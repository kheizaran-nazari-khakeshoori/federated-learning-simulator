"""Right panel - chart + clients + logs. Separate module."""
import tkinter as tk
import datetime

def create_right_panel(root):
    right_panel = tk.Frame(root, bg="#ecf0f1")
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    header = tk.Frame(right_panel, bg="white", height=50)
    header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))
    header.pack_propagate(False)
    tk.Label(header, text="Federated Learning - Global Model", bg="white", fg="#2c3e50", font=("Arial", 13, "bold")).pack(side=tk.LEFT, padx=15)
    status_var = tk.StringVar(value="Idle")
    status_lbl = tk.Label(header, textvariable=status_var, bg="#f1c40f", fg="#2c3e50", font=("Arial", 9, "bold"), padx=10, pady=4)
    status_lbl.pack(side=tk.RIGHT, padx=15)
    comm_var = tk.StringVar(value="comm 0.0MB")
    comm_lbl = tk.Label(header, textvariable=comm_var, bg="white", fg="#95a5a6", font=("Arial", 8))
    comm_lbl.pack(side=tk.RIGHT, padx=10)

    viz_frame = tk.Frame(right_panel, bg="#ecf0f1")
    viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

    server_card = tk.Frame(viz_frame, bg="white", relief=tk.FLAT, bd=1)
    server_card.pack(fill=tk.X, pady=(0, 10))
    tk.Label(server_card, text="SERVER", bg="white", fg="#7f8c8d", font=("Arial", 9, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
    server_inner = tk.Frame(server_card, bg="white")
    server_inner.pack(fill=tk.X, padx=12, pady=(0, 10))
    tk.Label(server_inner, text="Global Accuracy:", bg="white", fg="#2c3e50", font=("Arial", 10)).pack(side=tk.LEFT)
    accuracy_var = tk.StringVar(value="0.00%")
    tk.Label(server_inner, textvariable=accuracy_var, bg="white", fg="#27ae60", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
    round_var = tk.StringVar(value="Round: 0 / 10")
    tk.Label(server_inner, textvariable=round_var, bg="white", fg="#95a5a6", font=("Arial", 10)).pack(side=tk.RIGHT)

    chart_canvas = tk.Canvas(viz_frame, bg="white", height=180, highlightthickness=0)
    chart_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    def draw_chart(history, total_rounds):
        chart_canvas.delete("all")
        w = chart_canvas.winfo_width() or 700
        h = 180
        pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 30
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        chart_canvas.create_line(pad_l, h-pad_b, w-pad_r, h-pad_b, fill="#bdc3c7")
        chart_canvas.create_line(pad_l, pad_t, pad_l, h-pad_b, fill="#bdc3c7")
        for yv in [0, 50, 100]:
            y = h - pad_b - (yv/100)*plot_h
            chart_canvas.create_line(pad_l-4, y, pad_l, y, fill="#bdc3c7")
            chart_canvas.create_text(pad_l-8, y, text=f"{yv}%", anchor="e", fill="#95a5a6", font=("Arial", 7))
            chart_canvas.create_line(pad_l, y, w-pad_r, y, fill="#ecf0f1", dash=(2,2))
        if total_rounds > 0:
            for i in [0, total_rounds//2, total_rounds]:
                x = pad_l + (i/total_rounds)*plot_w if total_rounds else pad_l
                chart_canvas.create_text(x, h-pad_b+8, text=f"{i}", fill="#95a5a6", font=("Arial", 7))
            chart_canvas.create_text(w/2, h-8, text="Rounds", fill="#95a5a6", font=("Arial", 7))
        if not history:
            chart_canvas.create_text(w/2, h/2, text="Accuracy Chart (Rounds vs Accuracy) - waiting for training", fill="#bdc3c7", font=("Arial", 11, "italic"))
            return
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

    clients_frame = tk.Frame(viz_frame, bg="white")
    clients_frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(clients_frame, text="CLIENTS", bg="white", fg="#7f8c8d", font=("Arial", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 5))
    grid = tk.Frame(clients_frame, bg="white")
    grid.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)
    for c in range(3):
        grid.columnconfigure(c, weight=1)

    client_widgets = []
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
            bar_bg = tk.Frame(card, bg="#e0e0e0", height=6)
            bar_bg.pack(fill=tk.X, padx=10, pady=(2, 8))
            bar_bg.pack_propagate(False)
            bar_fill = tk.Frame(bar_bg, bg="#27ae60")
            bar_fill.place(relwidth=0, relheight=1)
            client_widgets.append({"acc_var": acc_v, "status_var": st_v, "card": card, "acc_lbl": acc_lbl, "st_lbl": st_lbl, "bar": bar_fill, "bar_bg": bar_bg})
    # label histogram canvas (per-client label dist)
    hist_frame = tk.Frame(viz_frame, bg="white", height=60)
    hist_frame.pack(fill=tk.X, pady=(0,5))
    def draw_hist(partitions):
        hist_canvas.delete("all")
        # hist_bars per client placeholder
        hist_canvas.create_text(150,25,text="label histogram per client",fill="#bdc3c7",font=("Arial",8))
    hist_canvas = tk.Canvas(hist_frame, bg="white", height=50, highlightthickness=0)
    hist_canvas.pack(fill=tk.BOTH, expand=True)
    rebuild_clients(5)

    log_frame = tk.Frame(right_panel, bg="white", height=160, relief=tk.SOLID, bd=1)
    log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
    log_frame.pack_propagate(False)
    log_header = tk.Frame(log_frame, bg="white")
    log_header.pack(fill=tk.X, padx=10, pady=(6, 2))
    tk.Label(log_header, text="Logs", bg="white", fg="#2c3e50", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
    tk.Label(log_header, text="● live", bg="white", fg="#27ae60", font=("Arial", 8)).pack(side=tk.LEFT, padx=(8, 0))
    autoscroll_var = tk.BooleanVar(value=True)
    tk.Checkbutton(log_header, text="Auto-scroll", variable=autoscroll_var, bg="white", fg="#7f8c8d", font=("Arial", 8), selectcolor="white", activebackground="white").pack(side=tk.RIGHT, padx=5)
    log_body = tk.Frame(log_frame, bg="white")
    log_body.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))
    text_scroll = tk.Scrollbar(log_body)
    text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    log_text = tk.Text(log_body, height=6, bg="#1e2a33", fg="#ecf0f1", font=("Consolas", 9), relief=tk.FLAT,
                       yscrollcommand=text_scroll.set, padx=8, pady=5, insertbackground="white", wrap=tk.WORD)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    text_scroll.config(command=log_text.yview)
    log_text.tag_config("INFO", foreground="#3498db")
    log_text.tag_config("SUCCESS", foreground="#2ecc71")
    log_text.tag_config("ROUND", foreground="#f1c40f")
    log_text.tag_config("CLIENT", foreground="#e67e22")
    log_text.tag_config("ERROR", foreground="#e74c3c")
    log_text.tag_config("TIME", foreground="#95a5a6")
    tk.Frame(log_frame, bg="#ecf0f1", height=1).pack(fill=tk.X, padx=10, pady=2)

    def log(msg, level="INFO"):
        log_text.config(state=tk.NORMAL)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        log_text.insert(tk.END, f"[{ts}] ", "TIME")
        log_text.insert(tk.END, f"[{level}] ", level)
        log_text.insert(tk.END, f"{msg}\n")
        log_text.config(state=tk.DISABLED)
        if autoscroll_var.get():
            log_text.see(tk.END)

    def export_canvas(canvas, path="chart.eps"):
        try: canvas.postscript(file=path); return True
        except: return False
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

    # confusion 10x10 grid placeholder
    confusion_frame = tk.Frame(viz_frame, bg="white", height=80)
    confusion_frame.pack(fill=tk.X, pady=5)
    confusion_grid = tk.Canvas(confusion_frame, bg="white", height=70, highlightthickness=0)
    confusion_grid.pack(fill=tk.BOTH, expand=True)
    confusion_grid.create_text(200,35,text="confusion matrix 10x10",fill="#bdc3c7",font=("Arial",8))
    log("Simulator ready. Configure left panel and press Start.", "INFO")
    log("Tip: Logs will show each round & client update here.", "INFO")

    return {
        "frame": right_panel,
        "status_var": status_var,
        "status_lbl": status_lbl,
        "accuracy_var": accuracy_var,
        "round_var": round_var,
        "chart_canvas": chart_canvas,
        "draw_chart": draw_chart,
        "client_widgets": client_widgets,
        "rebuild_clients": rebuild_clients,
        "log": log,
        "log_text": log_text,
        "autoscroll_var": autoscroll_var,
        "comm_var": comm_var,
        "comm_lbl": comm_lbl,
    }
