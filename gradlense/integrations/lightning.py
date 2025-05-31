from pytorch_lightning import Callback

class GradLenseCallback(Callback):
    def __init__(self, gradlense):
        super().__init__()
        self.gradlense = gradlense

    def on_fit_start(self, trainer, pl_module):
        self.gradlense.attach()

    def on_after_backward(self, trainer, pl_module):
        self.gradlense.step()

    def on_train_end(self, trainer, pl_module):
        self.gradlense.plot_line()
        self.gradlense.plot_heatmap()
        self.gradlense.summarize_alerts()
