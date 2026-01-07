import numpy as np
class FedNova:
    def aggregate(self, ws, ns):
        total=sum(ns); return sum(w*n/total for w,n in zip(ws, ns))
