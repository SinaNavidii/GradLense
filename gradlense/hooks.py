def attach_hooks(model, recorder):
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            module.register_full_backward_hook(recorder.get_hook(name))