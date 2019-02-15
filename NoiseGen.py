import torch


class NoiseGen(torch.nn.Module):
    def __init__(self):
        super(NoiseGen, self).__init__()
        self.GRU = torch.nn.GRU(input_size=10, hidden_size=10, batch_first=True)

    def forward(self, x):
        y, hidden = self.GRU(x)
        return y
