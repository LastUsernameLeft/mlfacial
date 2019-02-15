import torch
from math import floor, ceil

for x in range(100, 800):
    a = torch.randn(1, 5, x)
    # pool1 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 64), stride=floor(a.size(2) / 64))
    # pool2 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 32), stride=floor(a.size(2) / 32))
    # pool16 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 16), stride=floor(a.size(2) / 16))
    # pool10 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 10), stride=floor(a.size(2) / 10))
    # pool8 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 8), stride=floor(a.size(2) / 8))
    # pool15 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 15), stride=floor(a.size(2) / 15))


    w1 = ceil(a.size(2) / 14)
    padding = int((w1 * 14 - a.size(2) + 1) / 2)

    pool14 = torch.nn.MaxPool1d(kernel_size=w1, stride=floor(a.size(2) / 14), padding=padding)

    w2 = ceil(a.size(2) / 13)
    padding = int((w2 * 13 - a.size(2) + 1) / 2)

    pool13 = torch.nn.MaxPool1d(kernel_size=w2, stride=floor(a.size(2) / 13), padding=padding)
    pool12 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 12), stride=floor(a.size(2) / 12))
    pool11 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 11), stride=floor(a.size(2) / 11))
    pool10 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 10), stride=floor(a.size(2) / 10))
    pool9 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 9), stride=floor(a.size(2) / 9))
    pool8 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 8), stride=floor(a.size(2) / 8))
    pool7 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 7), stride=floor(a.size(2) / 7))
    pool6 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 6), stride=floor(a.size(2) / 6))
    pool5 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 5), stride=floor(a.size(2) / 5))
    pool4 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 4), stride=floor(a.size(2) / 4))
    pool3 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 3), stride=floor(a.size(2) / 3))
    pool2 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 2), stride=floor(a.size(2) / 2))
    pool1 = torch.nn.MaxPool1d(kernel_size=ceil(a.size(2) / 1), stride=floor(a.size(2) / 1))



    if torch.cat((pool1(a), pool2(a), pool3(a), pool4(a), pool5(a), pool6(a), pool7(a), pool8(a), pool9(a), pool10(a), pool11(a), pool12(a), pool13(a), pool14(a)), 2).size(2) != 105:
        print(pool1(a).size(2), pool2(a).size(2), pool3(a).size(2), pool4(a).size(2), pool5(a).size(2), pool6(a).size(2), pool7(a).size(2), pool8(a).size(2), pool9(a).size(2), pool10(a).size(2), pool11(a).size(2), pool12(a).size(2), pool13(a).size(2), pool14(a).size(2), "======", torch.cat((pool1(a), pool2(a), pool3(a), pool4(a), pool5(a), pool6(a), pool7(a), pool8(a), pool9(a), pool10(a), pool11(a), pool12(a), pool13(a), pool14(a)), 2).size(2))

    # temp = torch.cat((pool1(a), pool2(a), pool3(a), pool4(a), pool5(a), pool8(a), pool10(a), pool16(a)), 2)
    # print(temp.size())
