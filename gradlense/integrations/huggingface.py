from transformers import TrainerCallback

class GradLenseTrainerCallback(TrainerCallback):
    def __init__(self, gradlense):
        self.gradlense = gradlense

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.gradlense.attach()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.gradlense.step()

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.gradlense.plot_line()
        self.gradlense.plot_heatmap()
        self.gradlense.summarize_alerts()
