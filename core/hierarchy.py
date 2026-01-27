class Hierarchy:
    def __init__(self): self.edge=[]; self.cloud=None

    def edge_aggregate(self, ws):
        import numpy as np
        return sum(ws)/len(ws)
