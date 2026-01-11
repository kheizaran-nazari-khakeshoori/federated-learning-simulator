import numpy as np
def clip_grad(g, clip=1.0):
    n=np.linalg.norm(g); return g*min(1, clip/(n+1e-6))
