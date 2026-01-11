import numpy as np
def clip_grad(g, clip=1.0):
    n=np.linalg.norm(g); return g*min(1, clip/(n+1e-6))

def add_gaussian_noise(g, sigma=0.01):
    return g + np.random.randn(*g.shape)*sigma

def epsilon_budget(sigma, steps, delta=1e-5):
    return steps/(2*sigma*sigma)
