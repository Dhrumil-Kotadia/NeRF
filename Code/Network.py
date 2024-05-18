import torch
import torch.nn as nn
import torch.nn.functional as F

# Nerf Network
class NeRF(nn.Module):
    """
    NeRF Network Class.
    Args:
        Depth (int): Number of layers in the network.
        Width (int): Width of the network.
        input_size (int): Input size of the network.
        output_ch (int): Output size of the network.
    """
    def __init__(self, Depth=8, Width=256, input_size=3, output_ch=4):
        super(NeRF, self).__init__()
        self.Depth = Depth
        self.Width = Width
        self.fc1 = nn.Linear(3, Width)
        self.fc2 = nn.Linear(Width, Width)
        self.fc3 = nn.Linear(Width + 3, Width)
        self.fc4 = nn.Linear(Width, 4)
        self.sigma = nn.Linear(Width, 1)
        self.DirectionLayer = nn.Linear(Width+3, Width//2)
        self.OutputLayer = nn.Linear(Width//2, 3)
        self.PositionEncoder = PositionEncoder(input_channels=input_size)
        self.DirectionEncoder = PositionEncoder(input_channels=input_size, num_freqs=4)

    def forward(self, Sample_Points, Sample_Directions):
        # Sample_Points = self.PositionEncoder(Sample_Points)
        # Sample_Directions = self.DirectionEncoder(Sample_Directions)
        Input = Sample_Points
        
        for i in range(self.Depth):
            if i == 0:
                x = F.relu(self.fc1(Input)).to(Sample_Points.device)
            elif i == self.Depth//2:
                x = torch.cat([x, Input], -1)
                x = F.relu(self.fc3(x))
            else:
                x = F.relu(self.fc2(x))
        sigma = self.sigma(x).to(Sample_Points.device)
        x = self.fc2(x)
        X_Dir = torch.cat([x, Sample_Directions], -1).to(Sample_Points.device)
        x = F.relu(self.DirectionLayer(X_Dir))
        x = F.relu(self.OutputLayer(x))

        x = torch.cat([x, sigma], -1)

        return x


class PositionEncoder(torch.nn.Module):
    """
    Simplified version of PositionEncoder without the encoder selection functionality.
    
    Args:
        input_channels (int): number of input channels (3 for both xyz and direction)
        num_freqs (int): Number of frequencies to use for encoding.
        log_scale (bool): Whether to use logarithmic scaling for frequency bands.

    Note: This class is inspired by this NeRF implementation:  https://github.com/murumura/NeRF-Simple
    """
    def __init__(self, input_channels: int = 3, num_freqs: int = 10, log_scale: bool = True):
        super(PositionEncoder, self).__init__()
        self.num_freqs = num_freqs
        self.input_channels = input_channels
        self.encode_fn = [torch.sin, torch.cos]
        self.output_channels = input_channels * (len(self.encode_fn) * num_freqs + 1)
        if log_scale:
            self.freq_bands = 2**torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(num_freqs - 1), num_freqs)

    def forward(self, x):
        """
        Encodes the input tensor with sinusoidal encoding.

        Args:
            x (torch.Tensor): Input tensor to encode.

        Returns:
            torch.Tensor: Encoded output tensor.
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.encode_fn:
                out.append(func(freq * x))
        return torch.cat(out, -1)

    def get_output_channel_nums(self):
        """Returns the number of output channels after encoding."""
        return self.output_channels

def use_position_encoder(input_tensor, input_channels=3, num_freqs=10, log_scale=True):
    """Function to apply position encoding to an input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor to encode.
        input_channels (int, optional): The number of input channels. Defaults to 3.
        num_freqs (int, optional): The number of frequency bands to use. Defaults to 10.
        log_scale (bool, optional): Whether to use logarithmic scaling for frequencies. Defaults to True.

    Returns:
        torch.Tensor: The encoded tensor.
    """
    encoder = PositionEncoder(input_channels=input_channels, num_freqs=num_freqs, log_scale=log_scale)
    return encoder(input_tensor)