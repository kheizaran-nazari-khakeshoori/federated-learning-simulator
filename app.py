import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Federated Learning Simulator")
    root.geometry("800x600")
    # Center window on screen
    root.minsize(600, 400)
    
    # Empty window - we'll add controls here later
    label = tk.Label(root, text="Federated Learning Simulator - Empty Window", font=("Arial", 14))
    label.pack(expand=True)

    root.mainloop()

if __name__ == "__main__":
    main()
