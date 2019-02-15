"""Sequence classifier"""
import torch

class FinalClassifier(torch.nn.Module):
    def __init__(self):
        super(FinalClassifier, self).__init__()
        self.FinalClassify1 = torch.nn.Linear(in_features=306, out_features=153)
        self.FinalClassify2 = torch.nn.Linear(in_features=153, out_features=1)
        self.sigmoid = torch.nn.Sigmoid
        self.relu = torch.nn.ReLU
        self.normalisation = torch.nn.BatchNorm2d

    def forward(self, x):
        y = x

        return y

a = torch.random(5, 306)
