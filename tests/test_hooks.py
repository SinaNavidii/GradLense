import torch
import torch.nn as nn
from gradlense.hooks import attach_hooks
from gradlense.recorder import GradRecorder

def test_hook_records_gradients():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    recorder = GradRecorder()
    attach_hooks(model, recorder)

    x = torch.randn(4, 10, requires_grad=True)
    y = model(x).sum()
    y.backward()

    # Make sure something was recorded
    assert len(recorder.history) > 0, "No gradients recorded"
    for name, grads in recorder.history.items():
        assert isinstance(grads, list) and len(grads) > 0
        assert grads[-1] > 0, f"Gradient recorded for {name} is zero"
