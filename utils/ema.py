

def ema_update(model, old_model, alpha=0.9):
    for param, old_param in zip(model.parameters(), old_model.parameters()):
        param.data = alpha * param.data + (1 - alpha) * old_param.data