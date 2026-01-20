import numpy as np
def top_k(w, k=0.1):
    thresh=np.percentile(np.abs(w), 100*(1-k))
    w[np.abs(w)<thresh]=0
    return w
