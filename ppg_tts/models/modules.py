import torch
import numpy as np
from torch import nn
from collections import OrderedDict

class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor
    Input:
        x: (B, T, E)
        energy/pitch target: (B, T)
    """

    def __init__(self,
                 input_size: int,
                 filter_size: int,
                 kernel_size: int,
                 n_bins: int,
                 energy_min: float,
                 energy_max: float,
                 pitch_min: float,
                 pitch_max: float,
                 dropout: float=0.1):
        super(VarianceAdaptor, self).__init__()
        self.pitch_predictor = VariancePredictor(input_size,
                                                 filter_size,
                                                 kernel_size,
                                                 dropout)
        self.energy_predictor = VariancePredictor(input_size,
                                                 filter_size,
                                                 kernel_size,
                                                 dropout)
        

        self.n_bins = n_bins

        self.pitch_bins = nn.Parameter(
            torch.exp(
                torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
            ),
            requires_grad=False,
        )
        
        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(
            n_bins, input_size
        )
        self.energy_embedding = nn.Embedding(
            n_bins, input_size
        )

    def get_pitch_embedding(self, x, target, mask):
        prediction = self.pitch_predictor(x, mask)
        embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask):
        prediction = self.energy_predictor(x, mask)
        embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        
        return prediction, embedding

    def forward(
        self,
        x,
        mel_mask=None,
        pitch_target=None,
        energy_target=None,
    ):

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, mel_mask
        )
        x = x + pitch_embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, mel_mask
        )
        x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
        )


class SpeakerEmbeddingEncoder(nn.Module):
    """
    Mapping Speaker Embedding to target dimension
    """
    def __init__(self,
                 input_size: int,
                 model_size: int,
                 output_size: int,
                 dropout: float=0.5):
        super(SpeakerEmbeddingEncoder, self).__init__()

        self.input_size = input_size
        self.model_size = model_size
        self.output_size = output_size
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=model_size, bias=True),
            nn.LayerNorm(model_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=model_size, out_features=output_size, bias=True)
        )

    def forward(self, spk_embs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            spk_embs: tensor with shape (B, E)
        Returns:
            tensor with shape (B, E_out)
        """
        return self.encoder(spk_embs)
    
class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class VariancePredictor(nn.Module):
    """
    Duration, Pitch and Energy Predictor
    """

    def __init__(self,
                 input_size: int,
                 filter_size: int,
                 kernel_size: int,
                 dropout: float=0.1):
        super(VariancePredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel_size
        self.conv_output_size = filter_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Arguments:
            encoder_output: (B, T, E)
            mask: (B, T) or None
        Returns:
            (B, T) for predictions
        """
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out