from PIL import Image
from torchvision import transforms
from IDEncoder import IDEncoder
from AudioEncoder import AudioEncoder, calculate_limit
from AudioGRU import AudioGRU
from NoiseGen import NoiseGen
from FrameDecoder import FrameDecoder
from FrameDiscriminator import FrameDiscriminator
from SequenceGRU import SequenceGRU
import numpy as np
import torch

seq_len = 5
batch_size = 1

IdNet = IDEncoder()
AudNet = AudioEncoder()
AudGRU = AudioGRU()
SeqGRU = SequenceGRU()
NoiseNet = NoiseGen()
DecoderNet = FrameDecoder()
FrameDiscNet = FrameDiscriminator()

# Image
input_img = Image.open('299.jpg', mode='r')
ImgToTensor = transforms.ToTensor()
input_img = ImgToTensor(input_img)
input_img = input_img.view(1, 3, 128, 96)
IdOutput, u_net1, u_net2, u_net3, u_net4 = IdNet(input_img)
IdOutput = IdOutput.repeat(1, seq_len).view(-1, seq_len)
u_net1 = u_net1.repeat(seq_len, 1, 1, 1)
u_net2 = u_net2.repeat(seq_len, 1, 1, 1)
u_net3 = u_net3.repeat(seq_len, 1, 1, 1)
u_net4 = u_net4.repeat(seq_len, 1, 1, 1)
IdOutput = IdOutput.view(batch_size, seq_len, 50)
print("IdOutput.size()\n", IdOutput.size())

# Noise
np.random.seed(0)
input_noise = np.random.normal(loc=0, scale=np.sqrt(0.6), size=batch_size * seq_len * 10)
input_noise = np.reshape(input_noise, (batch_size, seq_len, 10))
input_noise = torch.from_numpy(input_noise).float()
NoiseOutput = NoiseNet(input_noise)
print("\nNoiseOutput.size()\n", NoiseOutput.size())

# Audio
input_Audio = np.load("00013.npy")
lim = calculate_limit(seq_len)
input_Audio = input_Audio[10:lim, 1:]
input_Audio = torch.from_numpy(input_Audio).float()
input_seq = input_Audio.detach().unfold(0, size=20, step=4)
input_seq = input_seq.view(seq_len * batch_size, 1, 12, 20)
AudOutput = AudNet(input_seq)
AudOutput = AudOutput.view(batch_size, seq_len, 256)
AudOutput = AudGRU(AudOutput)
print("\nAudOutput.size()\n", AudOutput.size())

# Concatenates
FinalOutput = torch.cat((NoiseOutput, IdOutput, AudOutput), 2)        # 2 works, if removing seq length, make it 1 again
# Concatenate must be done on the dimension that is unequal
# Repeating image tensor is the option we used
print("\nFinalOutput.size()\n", FinalOutput.size())
final_encoding = FinalOutput.view((batch_size * seq_len, 316, 1, 1))
print("\nFinalOutput.size()\n", FinalOutput.size())

# Decoder
GeneratedImage = DecoderNet(final_encoding, u_net1, u_net2, u_net3, u_net4)
print("\nGeneratedImage.size()\n", GeneratedImage.size())

# TODO: Save generated image at this point

# Frame Discriminator
ConcatenatedDecoderImage = torch.cat((GeneratedImage, input_img.repeat(seq_len, 1, 1, 1)), 1)
ImageTruthValue = FrameDiscNet(ConcatenatedDecoderImage)
print("\nImageTruthValue\n", ImageTruthValue.size(), '\n', list(ImageTruthValue.data))

# Sequence Discriminator
# Reuse image encoder
GeneratedImageEncoded = IdNet(GeneratedImage)[0]
print("GeneratedImageEncoded.size()", GeneratedImageEncoded.size())
# cat here
GeneratedImageEncoded = GeneratedImageEncoded.view(batch_size, seq_len, 50)
print("AudOutput.size(), GeneratedImageEncoded.size()", AudOutput.size(), GeneratedImageEncoded.size())
sequence = torch.cat((AudOutput, GeneratedImageEncoded), 2)
# gru here
sequence = SeqGRU(sequence)
sequence = sequence.view(batch_size, seq_len * 306)
print(sequence.size())

# classifier here OR gru+classifier (one piece) here
