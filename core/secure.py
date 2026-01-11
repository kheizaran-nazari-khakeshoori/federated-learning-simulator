import numpy as np
def mask_weights(w):
    mask=np.random.randn(*w.shape)*0.01
    return w+mask, mask
