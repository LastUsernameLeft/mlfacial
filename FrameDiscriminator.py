import torch


class FrameDiscriminator(torch.nn.Module):
    def __init__(self):
        super(FrameDiscriminator, self).__init__()
        self.FDConv1 = torch.nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.FDbn1 = torch.nn.BatchNorm2d(num_features=32)
        self.FDConv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.FDbn2 = torch.nn.BatchNorm2d(num_features=64)
        self.FDConv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
        self.FDbn3 = torch.nn.BatchNorm2d(num_features=128)
        self.FDConv4 = torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(5, 5), stride=1, padding=2)
        self.FDbn4 = torch.nn.BatchNorm2d(num_features=96)
        self.FDFC1 = torch.nn.Linear(in_features=4608, out_features=1024)
        self.FDbnfc = torch.nn.LayerNorm(normalized_shape=1024)
        self.FDFC2 = torch.nn.Linear(in_features=1024, out_features=1)
        self.FDMaxPool = torch.nn.MaxPool2d(kernel_size=(5, 5), stride=2, padding=2)
        self.FDLeakyReLU = torch.nn.LeakyReLU(negative_slope=0.2)
        self.FDSig = torch.nn.Sigmoid()

    def forward(self, image_frame):
        image_frame = self.FDConv1(image_frame)
        image_frame = self.FDMaxPool(image_frame)
        image_frame = self.FDLeakyReLU(image_frame)
        image_frame = self.FDbn1(image_frame)
        image_frame = self.FDConv2(image_frame)
        image_frame = self.FDMaxPool(image_frame)
        image_frame = self.FDLeakyReLU(image_frame)
        image_frame = self.FDbn2(image_frame)
        image_frame = self.FDConv3(image_frame)
        image_frame = self.FDMaxPool(image_frame)
        image_frame = self.FDLeakyReLU(image_frame)
        image_frame = self.FDbn3(image_frame)
        image_frame = self.FDConv4(image_frame)
        image_frame = self.FDMaxPool(image_frame)
        image_frame = self.FDLeakyReLU(image_frame)
        image_frame = self.FDbn4(image_frame)
        image_frame = torch.flatten(image_frame, start_dim=1)
        image_frame = self.FDFC1(image_frame)
        image_frame = self.FDLeakyReLU(image_frame)
        image_frame = self.FDbnfc(image_frame)
        image_frame = self.FDFC2(image_frame)
        image_frame_tv = self.FDSig(image_frame)

        return image_frame_tv
