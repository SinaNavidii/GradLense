from .hooks import attach_hooks
from .recorder import GradRecorder
from .visualizer import GradVisualizer

class GradLense:
    def __init__(self, model):
        self.model = model
        self.recorder = GradRecorder()
        self.visualizer = GradVisualizer(self.recorder)

    def attach(self):
        attach_hooks(self.model, self.recorder)

    def step(self):
        # Placeholder for tracking step count or batch info
        pass

    def plot_line(self):
        self.visualizer.plot_gradient_lines()

    def plot_heatmap(self):
        self.visualizer.plot_gradient_heatmap()

    def summarize_alerts(self):
        alerts = self.recorder.summarize()
        for alert in alerts:
            print(alert)
