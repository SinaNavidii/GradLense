import numpy as np

class GradRecorder:
    def __init__(self):
        self.history = {}

    def get_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                norm = grad_output[0].norm().item()
                self.history.setdefault(name, []).append(norm)
        return hook
    def summarize(self, zero_thresh=1e-6, spike_factor=10):
        alerts = []
        for name, grads in self.history.items():
            grads = np.array(grads)

            if np.allclose(grads, 0):
                alerts.append(f"[ALERT] {name} has zero gradients (dead layer).")
                continue

            if (grads < zero_thresh).sum() > len(grads) // 2:
                alerts.append(f"[WARN] {name} has low gradients for most steps.")

            spike_threshold = spike_factor * (np.median(grads) + 1e-8)
            spike_count = np.sum(grads > spike_threshold)
            if spike_count > len(grads) * 0.1:
                alerts.append(f"[ALERT] {name} has exploding gradients detected.")
        return alerts
