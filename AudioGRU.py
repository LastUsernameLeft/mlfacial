"""GRU For Audio encoder"""
import torch


class AudioGRU(torch.nn.Module):
    def __init__(self):
        super(AudioGRU, self).__init__()
        self.AudEncGRU = torch.nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.AudEnBN8 = torch.nn.LayerNorm(normalized_shape=256)
        self.tanh = torch.nn.Tanh()

    def forward(self, aud_tensor):
        aud_tensor, h0 = self.AudEncGRU(aud_tensor)
        aud_tensor = self.tanh(self.AudEnBN8(aud_tensor))

        return aud_tensor
