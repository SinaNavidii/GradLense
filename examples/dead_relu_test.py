import torch
import torch.nn as nn
import torch.optim as optim
from gradlense import GradLense

# Model designed to break
class BrokenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)  # likely to be zero
        return self.fc2(x)

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(128, 10) - 5.0  # Shifted input → mostly negative → ReLU outputs zero
y = torch.randn(128, 1)

model = BrokenMLP()
optimizer = optim.SGD(model.parameters(), lr=10.0)  # Exploding LR
criterion = nn.MSELoss()

gradlense = GradLense(model)
gradlense.attach()

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

# Plot and summarize
gradlense.plot_line()
gradlense.plot_heatmap()
gradlense.summarize_alerts()
