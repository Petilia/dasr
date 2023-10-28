import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(
            bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim
        )
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


class SimpleModelLSTM(nn.Module):
    """
    Dummy speech enhancement model (like Facebook Denoiser Demics).
    Model is small and have not skip connections and lstm module
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.

        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    """

    def __init__(
        self,
        chin=1,
        chout=1,
        depth=3,
        kernel_size=8,
        stride=4,
        causal=True,
        hidden=48,
        growth=2,
        max_hidden=4096,
        normalize=True,
        resample=1,
        floor=1e-3,
        sample_rate=16_000,
    ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalize = normalize
        self.resample = resample
        self.floor = floor
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(depth):
            encoder_block = nn.Sequential(
                nn.Conv1d(chin, hidden, kernel_size, stride), nn.ReLU()
            )
            self.encoder.append(encoder_block)

            decoder_block = [
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decoder_block.append(nn.ReLU())
            decoder_block = nn.Sequential(*decoder_block)
            self.decoder.insert(0, decoder_block)

            chin = hidden
            chout = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1

        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        # Здесь мог быть ваш upsampling

        for encode in self.encoder:
            x = encode(x)
            # Здесь могли быть ваши skip connections

        x = x.permute(2, 0, 1)
        # print(x.shape)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        for decode in self.decoder:
            x = decode(x)
            # Здесь могли быть ваши skip connections

        x = x[..., :length]

        # Здесь мог быть ваш downsampling

        return std * x
