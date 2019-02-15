import torch


class IDEncoder(torch.nn.Module):
    def __init__(self):
        super(IDEncoder, self).__init__()
        self.IdEnConv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1)
        self.IdEnBN1 = torch.nn.BatchNorm2d(num_features=64)
        self.IdEnConv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1)
        self.IdEnBN2 = torch.nn.BatchNorm2d(num_features=128)
        self.IdEnConv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1)
        self.IdEnBN3 = torch.nn.BatchNorm2d(num_features=256)
        self.IdEnConv4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1)
        self.IdEnBN4 = torch.nn.BatchNorm2d(num_features=512)
        self.IdEnConv5 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=2, padding=1)
        self.IdEnBN5 = torch.nn.BatchNorm2d(num_features=1024)
        self.IdEnConv6 = torch.nn.Conv2d(in_channels=1024, out_channels=50, kernel_size=(4, 3), stride=2)
        self.IdEnBN6 = torch.nn.LayerNorm(normalized_shape=50)
        self.relu = torch.nn.ReLU()
        self.Tanh = torch.nn.Tanh()

    def forward(self, id_tensor):
        id_tensor = self.IdEnConv1(id_tensor)
        u_net1 = self.IdEnConv2(self.IdEnBN1(self.relu(id_tensor)))
        u_net2 = self.IdEnConv3(self.IdEnBN2(self.relu(u_net1)))
        u_net3 = self.IdEnConv4(self.IdEnBN3(self.relu(u_net2)))
        u_net4 = self.IdEnConv5(self.IdEnBN4(self.relu(u_net3)))
        id_tensor = self.IdEnConv6(self.IdEnBN5(self.relu(u_net4)))
        id_tensor = id_tensor.view(-1, 50)
        id_tensor = self.Tanh(self.IdEnBN6(id_tensor))

        return id_tensor, u_net1, u_net2, u_net3, u_net4
