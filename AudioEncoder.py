"""Produces [batch_size, 256] encoding from MFCC input"""
import torch
from torch.nn import functional as t_fnc


def calculate_limit(seq):
    """Calculates 'upper bound limit' of the input data based on given sequence length.
    Skipping 10 points on dimension 0"""
    return 10 + 8 + (4 * seq) + 8


class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.AudEnConv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.AudEnBN1 = torch.nn.BatchNorm2d(num_features=64)
        self.AudEnConv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.AudEnBN2 = torch.nn.BatchNorm2d(num_features=128)
        self.AudEncPool2 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2))
        self.AudEnConv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.AudEnBN3 = torch.nn.BatchNorm2d(num_features=256)
        self.AudEnConv4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.AudEnBN4 = torch.nn.BatchNorm2d(num_features=256)
        self.AudEnConv5 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.AudEnBN5 = torch.nn.BatchNorm2d(num_features=512)
        self.AudEncPool5 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.AudEncFC6 = torch.nn.Linear(in_features=8192, out_features=512)
        self.AudEnBN6 = torch.nn.LayerNorm(normalized_shape=512)
        self.AudEncFC7 = torch.nn.Linear(in_features=512, out_features=256)
        self.AudEnBN7 = torch.nn.LayerNorm(normalized_shape=256)
        self.relu = torch.nn.ReLU()

    def forward(self, aud_tensor):
        aud_tensor = self.AudEnBN1(self.relu(self.AudEnConv1(aud_tensor)))
        aud_tensor = self.AudEnBN2(self.relu(self.AudEnConv2(aud_tensor)))
        # noinspection PyTypeChecker
        aud_tensor = t_fnc.pad(input=aud_tensor, pad=(0, 1, 0, 1))
        aud_tensor = self.AudEncPool2(aud_tensor)
        aud_tensor = self.AudEnBN3(self.relu(self.AudEnConv3(aud_tensor)))
        aud_tensor = self.AudEnBN4(self.relu(self.AudEnConv4(aud_tensor)))
        aud_tensor = self.AudEnBN5(self.relu(self.AudEnConv5(aud_tensor)))
        aud_tensor = self.AudEncPool5(aud_tensor)
        aud_tensor = torch.flatten(aud_tensor, start_dim=1)
        aud_tensor = self.AudEnBN6(self.relu(self.AudEncFC6(aud_tensor)))
        aud_tensor = self.AudEnBN7(self.relu(self.AudEncFC7(aud_tensor)))

        return aud_tensor
