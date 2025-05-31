import numpy as np
from gradlense.recorder import GradRecorder

def test_vanishing_gradient_alert():
    recorder = GradRecorder()
    name = "vanish_layer"
    recorder.history[name] = [1e-7] * 100
    alerts = recorder.summarize()
    assert any("low gradients" in alert for alert in alerts), "Vanishing gradient not detected"

def test_exploding_gradient_alert_multiple_spikes():
    recorder = GradRecorder()
    name = "spiky_layer"
    base = [0.01] * 89
    spikes = [5.0] * 11
    recorder.history[name] = base + spikes
    alerts = recorder.summarize()
    assert any("exploding gradients" in alert for alert in alerts), "Exploding gradients not detected with multiple spikes"

def test_no_alert_for_single_spike():
    recorder = GradRecorder()
    name = "one_spike_layer"
    recorder.history[name] = [0.01] * 99 + [10.0]
    alerts = recorder.summarize()
    assert not any("exploding gradients" in alert for alert in alerts), "False positive on single gradient spike"

def test_zero_gradients_detection():
    recorder = GradRecorder()
    name = "dead_layer"
    recorder.history[name] = [0.0] * 100
    alerts = recorder.summarize()
    assert any("zero gradients" in alert for alert in alerts), "Zero-gradient (dead) layer not detected"

def test_no_alerts_for_normal_gradients():
    recorder = GradRecorder()
    name = "normal_layer"
    recorder.history[name] = np.random.uniform(0.01, 0.1, size=100).tolist()
    alerts = recorder.summarize()
    assert len(alerts) == 0, "False positive alert on normal gradients"
