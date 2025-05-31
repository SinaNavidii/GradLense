import matplotlib.pyplot as plt
import numpy as np

class GradVisualizer:
    def __init__(self, recorder):
        self.recorder = recorder

    def plot_gradient_lines(self):
        plt.figure(figsize=(10, 6), dpi=150)
        for name, values in self.recorder.history.items():
            plt.plot(values, label=name)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 7287573 (clean up visualizer!)
        plt.xlabel('Batch Step', fontsize=16)
        plt.ylabel('Gradient Norm', fontsize=16)
        plt.title('Gradient Flow Over Time', fontsize=18)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
<<<<<<< HEAD
=======
        plt.xlabel('Batch Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Flow Over Time')
        plt.legend(fontsize=8)
        plt.grid(True)
>>>>>>> ed662b8 (Add filters to top k layers of total layers in large models!)
=======
>>>>>>> 7287573 (clean up visualizer!)
        plt.tight_layout()
        plt.show()

    def plot_gradient_heatmap(self, top_k=None):
        if not self.recorder.history:
            print("No gradient data to plot.")
            return

        total_layers = len(self.recorder.history)
        if top_k is None and total_layers > 50:
            top_k = 30

        # Select top_k layers based on mean gradient norm if specified
        sorted_keys = sorted(
            self.recorder.history,
            key=lambda k: -np.mean(self.recorder.history[k])
        )
        keys = sorted_keys[:top_k] if top_k else sorted_keys

        # Normalize and pad to same length
        max_len = max(len(self.recorder.history[k]) for k in keys)
        data = [self.recorder.history[k] + [0]*(max_len - len(self.recorder.history[k])) for k in keys]

        # Clean up long labels
        short_names = [
            k.replace("bert.encoder.layer.", "L").replace(".attention.self.", ".attn.")
             .replace("intermediate.dense", "inter.dense").replace("output.dense", "out.dense")
             for k in keys
        ]

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 7287573 (clean up visualizer!)
        plt.figure(figsize=(10, max(6, 0.3 * len(keys))), dpi=100)
        im = plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Gradient Norm')
        plt.yticks(range(len(short_names)), short_names, fontsize=14)
        plt.xticks(fontsize=14)
        plt.xlabel('Step', fontsize=16)
        plt.ylabel('Layer', fontsize=16)
        plt.title('Gradient Heatmap', fontsize=18)
        plt.grid(False)
<<<<<<< HEAD
=======
        plt.figure(figsize=(10, max(6, 0.3 * len(keys))))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Gradient Norm')
        plt.yticks(range(len(short_names)), short_names, fontsize=6)
        plt.xticks(fontsize=8)
        plt.xlabel('Step')
        plt.ylabel('Layer')
        plt.title('Gradient Heatmap')
>>>>>>> ed662b8 (Add filters to top k layers of total layers in large models!)
=======
>>>>>>> 7287573 (clean up visualizer!)
        plt.tight_layout()
        plt.show()
