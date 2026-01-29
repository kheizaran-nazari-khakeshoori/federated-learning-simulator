def fedper_freeze(model):
    # freeze base features
    for p in model.parameters(): p.requires_grad=False
