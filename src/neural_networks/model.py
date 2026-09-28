"""The same small network used in the lesson, training, and live demo."""
import torch
from torch import nn

FEATURES = ["Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]


class ChurnNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.sigmoid(self.output(x))

    def inspect(self, x):
        """Return actual intermediate values for the educational frontend."""
        with torch.no_grad():
            z1 = self.layer1(x)
            a1 = self.relu(z1)
            z2 = self.layer2(a1)
            a2 = self.relu(z2)
            z3 = self.output(a2)
            p = self.sigmoid(z3)
        return {k: v.cpu().numpy()[0] for k, v in
                dict(inputs=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, probability=p).items()}
