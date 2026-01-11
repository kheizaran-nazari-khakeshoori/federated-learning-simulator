class FlowerServer:
    def __init__(self, strategy="fedavg"): self.strategy=strategy
    def aggregate(self, results): return results[0]
