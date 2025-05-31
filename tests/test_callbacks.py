import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from gradlense import GradLense
from gradlense.integrations.lightning import GradLenseCallback

class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

def test_lightning_callback_runs():
    X = torch.randn(64, 10)
    y = torch.randn(64, 1)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16)

    model = TinyModel()
    gradlense = GradLense(model)
    trainer = pl.Trainer(max_epochs=1, enable_model_summary=False, logger=False, callbacks=[GradLenseCallback(gradlense)])
    trainer.fit(model, dataloader)

    assert len(gradlense.recorder.history) > 0, "GradLense did not record any gradients via callback"