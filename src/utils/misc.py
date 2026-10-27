def adjust_learning_rate(datasets, model, epoch, lr, args):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in args.schedule:
        lr *= args.gamma
        optimizers = [getattr(model.model, attr) for attr in dir(model.model) if attr.startswith("optimizer") and getattr(model.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:  # access the current learning rate
        optimizers = [getattr(model.model, attr) for attr in dir(model.model) if attr.startswith("optimizer") and getattr(model.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

    return lr
