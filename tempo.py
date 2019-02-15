import numpy as np
import torch

for a in range(100, 900):
    # a = 100
    output_size = 100
    x = torch.randn(1, 306, a)
    # print(x.size())

    N, C, H = x.size()
    sizeX = int(np.ceil(H / output_size))
    stride = int(np.floor(H / output_size))
    temp = torch.nn.MaxPool1d(kernel_size=sizeX, stride=stride)
    print(x.size(), temp(x).size())
