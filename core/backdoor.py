import numpy as np
def add_patch(x):
    x[:3,:3]=1; return x

def spectral_detect(ws):
    return 0

def attack_success(y_pred, y_target):
    return (y_pred==y_target).mean()
