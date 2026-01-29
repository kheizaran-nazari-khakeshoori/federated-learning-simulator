def fedper_freeze(model):
    # freeze base features
    for p in model.parameters(): p.requires_grad=False

def pfedme_update(model, lr=0.01, lam=15):
    # Moreau envelope
    pass
