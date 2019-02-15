"""GRU For Sequence Discriminator"""
import torch


class SequenceGRU(torch.nn.Module):
    def __init__(self):
        super(SequenceGRU, self).__init__()
        self.SeqGRU = torch.nn.GRU(input_size=306, hidden_size=306, num_layers=2, batch_first=True)
        self.SeqBN = torch.nn.LayerNorm(normalized_shape=306)
        self.tanh = torch.nn.Tanh()

    def forward(self, sequence):
        sequence, h0 = self.SeqGRU(sequence)
        sequence = self.tanh(self.SeqBN(sequence))

        return sequence
