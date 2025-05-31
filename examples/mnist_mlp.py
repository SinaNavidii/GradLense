import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from gradlense import GradLense

# Simple MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        return self.fc2(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
dataloader = DataLoader(datasets.MNIST('.', train=True, download=True,
                          transform=transforms.ToTensor()), batch_size=64, shuffle=True)

gradlense = GradLense(model)
gradlense.attach()

model.train()
for i, (X, y) in enumerate(dataloader):
    if i > 100:  # Limit for quick testing
        break
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Visualize
gradlense.plot_line()
gradlense.plot_heatmap()
# Print gradient flow diagnostics
gradlense.summarize_alerts()
