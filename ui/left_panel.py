"""Left control panel - separate module."""
import tkinter as tk
from data.datasets import get_dataset_info
try:
    from core.config import load_config
    _cfg=load_config()
except: _cfg={}

def create_left_panel(root):
    left_panel = tk.Frame(root, bg="#2c3e50", width=260)
    left_panel.pack(side=tk.LEFT, fill=tk.Y)
    left_panel.pack_propagate(False)

    tk.Label(left_panel, text="Controls", bg="#2c3e50", fg="white", font=("Arial", 16, "bold")).pack(pady=(20, 15), padx=10)
    tk.Frame(left_panel, bg="#34495e", height=2).pack(fill=tk.X, padx=15, pady=(0, 15))

    def add_section_label(text):
        lbl = tk.Label(left_panel, text=text, bg="#2c3e50", fg="#bdc3c7", font=("Arial", 9, "bold"), anchor="w")
        lbl.pack(fill=tk.X, padx=15, pady=(10, 2))
        return lbl

    add_section_label("NUM CLIENTS")
    clients_var = tk.StringVar(value=str(_cfg.get("clients",5)))
    tk.Spinbox(left_panel, from_=2, to=100, textvariable=clients_var, wrap=True, font=("Arial", 10), width=10).pack(padx=15, fill=tk.X, pady=(0, 5))

    add_section_label("ROUNDS")
    rounds_var = tk.StringVar(value="10")
    tk.Spinbox(left_panel, from_=1, to=1000, textvariable=rounds_var, wrap=True, font=("Arial", 10), width=10).pack(padx=15, fill=tk.X, pady=(0, 5))

    add_section_label("LOCAL EPOCHS")
    epochs_var = tk.StringVar(value="3")
    tk.Spinbox(left_panel, from_=1, to=50, textvariable=epochs_var, wrap=True, font=("Arial", 10), width=10).pack(padx=15, fill=tk.X, pady=(0, 5))

    add_section_label("DATASET")
    dataset_var = tk.StringVar(value="MNIST")
    m = tk.OptionMenu(left_panel, dataset_var, "MNIST", "CIFAR-10", "Fashion-MNIST", "Synthetic")
    m.config(bg="white", font=("Arial", 10), width=12)
    m.pack(padx=15, fill=tk.X, pady=(0, 5))
    info_lbl = tk.Label(left_panel, text=get_dataset_info("MNIST"), bg="#2c3e50", fg="#7f8c8d", font=("Arial", 7), wraplength=220, justify="left")
    info_lbl.pack(padx=15, anchor="w", pady=(0, 5))
    def _update_info(*a): info_lbl.config(text=get_dataset_info(dataset_var.get()))
    dataset_var.trace_add("write", _update_info)

    add_section_label("AGGREGATION")
    # mu slider wired in next engine commit
    mu_var = tk.DoubleVar(value=0.1)
    tk.Scale(left_panel, from_=0.01, to=1.0, resolution=0.01, variable=mu_var, orient=tk.HORIZONTAL).pack()
    agg_var = tk.StringVar(value="FedAvg")
    m2 = tk.OptionMenu(left_panel, agg_var, "FedAvg", "FedProx", "FedAdam")
    m2.config(bg="white", font=("Arial", 10), width=12)
    m2.pack(padx=15, fill=tk.X, pady=(0, 5))

    add_section_label("NON-IID ALPHA (Dirichlet)")
    alpha_var = tk.DoubleVar(value=0.5)
    tk.Label(left_panel, text="0.1=very non-IID  •  10=IID-like", bg="#2c3e50", fg="#7f8c8d", font=("Arial", 7)).pack(padx=15, anchor="w")
    alpha_row = tk.Frame(left_panel, bg="#2c3e50")
    alpha_row.pack(fill=tk.X, padx=15, pady=(2, 5))
    alpha_label = tk.Label(alpha_row, text="0.50", bg="#34495e", fg="white", font=("Arial", 9, "bold"), width=5)
    alpha_label.pack(side=tk.RIGHT, padx=(5, 0))
    def _fmt_alpha(v): return f"{float(v):.2f}"
    tk.Scale(alpha_row, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, variable=alpha_var,
             bg="#2c3e50", fg="white", troughcolor="#34495e", highlightthickness=0,
             activebackground="#3498db", showvalue=0, length=140,
             command=lambda v: alpha_label.config(text=_fmt_alpha(v))).pack(side=tk.LEFT, fill=tk.X, expand=True)
    preset_frame = tk.Frame(left_panel, bg="#2c3e50")
    preset_frame.pack(fill=tk.X, padx=15, pady=(0, 5))
    for val, txt in [(0.1, "High"), (0.5, "Med"), (10.0, "IID")]:
        def _set(v=val): alpha_var.set(v); alpha_label.config(text=_fmt_alpha(v))
        tk.Button(preset_frame, text=txt, command=_set, bg="#34495e", fg="#bdc3c7", font=("Arial", 7), relief=tk.FLAT, padx=6, pady=2).pack(side=tk.LEFT, padx=2)

    add_section_label("LEARNING RATE")
    lr_var = tk.DoubleVar(value=0.05)
    tk.Label(left_panel, text="small=stable • large=fast/diverge", bg="#2c3e50", fg="#7f8c8d", font=("Arial", 7)).pack(padx=15, anchor="w")
    lr_row = tk.Frame(left_panel, bg="#2c3e50")
    lr_row.pack(fill=tk.X, padx=15, pady=(2, 5))
    lr_label = tk.Label(lr_row, text="0.050", bg="#34495e", fg="white", font=("Arial", 9, "bold"), width=5)
    lr_label.pack(side=tk.RIGHT, padx=(5, 0))
    def _fmt_lr(v): return f"{float(v):.3f}"
    tk.Scale(lr_row, from_=0.005, to=0.2, resolution=0.005, orient=tk.HORIZONTAL, variable=lr_var,
             bg="#2c3e50", fg="white", troughcolor="#34495e", highlightthickness=0,
             activebackground="#3498db", showvalue=0, length=140,
             command=lambda v: lr_label.config(text=_fmt_lr(v))).pack(side=tk.LEFT, fill=tk.X, expand=True)
    lr_preset = tk.Frame(left_panel, bg="#2c3e50")
    lr_preset.pack(fill=tk.X, padx=15, pady=(0, 5))
    for val, txt in [(0.01, "Low"), (0.05, "Med"), (0.1, "High")]:
        def _set_lr(v=val): lr_var.set(v); lr_label.config(text=_fmt_lr(v))
        tk.Button(lr_preset, text=txt, command=_set_lr, bg="#34495e", fg="#bdc3c7", font=("Arial", 7), relief=tk.FLAT, padx=6, pady=2).pack(side=tk.LEFT, padx=2)

    dp_var = tk.BooleanVar(value=False)
    tk.Checkbutton(left_panel, text="DP enabled", variable=dp_var, bg="#2c3e50", fg="white").pack(anchor="w", padx=15)
    noise_var = tk.DoubleVar(value=0.01)
    tk.Scale(left_panel, from_=0.001, to=0.1, resolution=0.001, variable=noise_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15)
    add_section_label("CLIENT FRACTION C")
    client_frac_var = tk.DoubleVar(value=1.0)
    tk.Label(left_panel, text="fraction sampled per round", bg="#2c3e50", fg="#7f8c8d", font=("Arial", 7)).pack(padx=15, anchor="w")
    c_row = tk.Frame(left_panel, bg="#2c3e50")
    c_row.pack(fill=tk.X, padx=15, pady=(2, 5))
    c_label = tk.Label(c_row, text="1.00", bg="#34495e", fg="white", font=("Arial", 9, "bold"), width=5)
    c_label.pack(side=tk.RIGHT, padx=(5, 0))
    def _fmt_c(v): return f"{float(v):.2f}"
    tk.Scale(c_row, from_=0.2, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=client_frac_var,
             bg="#2c3e50", fg="white", troughcolor="#34495e", highlightthickness=0,
             activebackground="#3498db", showvalue=0, length=140,
             command=lambda v: c_label.config(text=_fmt_c(v))).pack(side=tk.LEFT, fill=tk.X, expand=True)
    c_preset = tk.Frame(left_panel, bg="#2c3e50")
    c_preset.pack(fill=tk.X, padx=15, pady=(0, 5))
    for val, txt in [(0.5, "50%"), (0.8, "80%"), (1.0, "100%")]:
        def _set_c(v=val): client_frac_var.set(v); c_label.config(text=_fmt_c(v))
        tk.Button(c_preset, text=txt, command=_set_c, bg="#34495e", fg="#bdc3c7", font=("Arial", 7), relief=tk.FLAT, padx=6, pady=2).pack(side=tk.LEFT, padx=2)

    # presets fast/accurate
    pf = tk.Frame(left_panel, bg="#2c3e50")
    pf.pack(fill=tk.X, padx=15, pady=5)
    tk.Button(pf, text="Fast", command=lambda: [clients_var.set(3), rounds_var.set(5)]).pack(side=tk.LEFT)
    tk.Button(pf, text="Accurate", command=lambda: [clients_var.set(5), rounds_var.set(20)]).pack(side=tk.LEFT)
    tk.Frame(left_panel, bg="#2c3e50", height=10).pack()

    start_btn = tk.Button(left_panel, text="▶ Start Training", bg="#27ae60", fg="white", activebackground="#2ecc71",
                          font=("Arial", 11, "bold"), relief=tk.FLAT, padx=10, pady=8)
    start_btn.pack(padx=15, fill=tk.X, pady=5)

    stop_btn = tk.Button(left_panel, text="■ Stop", bg="#c0392b", fg="white", activebackground="#e74c3c",
                         font=("Arial", 10), relief=tk.FLAT, padx=10, pady=6)
    stop_btn.pack(padx=15, fill=tk.X, pady=(5, 10))

    return {
        "frame": left_panel,
        "clients_var": clients_var,
        "rounds_var": rounds_var,
        "epochs_var": epochs_var,
        "dataset_var": dataset_var,
        "agg_var": agg_var,
        "alpha_var": alpha_var,
        "lr_var": lr_var,
        "client_frac_var": client_frac_var,
        "dp_var": dp_var,
        "noise_var": noise_var,
        "start_btn": start_btn,
        "stop_btn": stop_btn,
    }
