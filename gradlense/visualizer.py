import matplotlib.pyplot as plt
import numpy as np
import os

class GradVisualizer:
    def __init__(self, recorder):
        self.recorder = recorder
        self.output_dir = "../figures"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_gradient_lines(self):
        plt.figure(figsize=(10, 6), dpi=150)
        for name, values in self.recorder.history.items():
            plt.plot(values, label=name)
        plt.xlabel('Batch Step', fontsize=16)
        plt.ylabel('Gradient Norm', fontsize=16)
        plt.title('Gradient Flow Over Time', fontsize=18)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "1.png"), dpi=300)
        plt.show()

    def plot_gradient_heatmap(self, top_k=None):
        if not self.recorder.history:
            print("No gradient data to plot.")
            return

        total_layers = len(self.recorder.history)
        if top_k is None and total_layers > 50:
            top_k = 30

        # Select top_k layers based on mean gradient norm
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
            k.replace("bert.encoder.layer.", "L")
             .replace(".attention.self.", ".attn.")
             .replace("intermediate.dense", "inter.dense")
             .replace("output.dense", "out.dense")
            for k in keys
        ]

        plt.figure(figsize=(12, max(6, 0.4 * len(keys))), dpi=200)
        im = plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Gradient Norm', shrink=0.8)
        plt.yticks(range(len(short_names)), short_names, fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel('Step', fontsize=16)
        plt.ylabel('Layer', fontsize=16)
        plt.title('Gradient Heatmap', fontsize=18)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "2.png"), dpi=300)
        plt.show()
