import torch
import unittest
from ppg_tts.models.encoder import ConvReluNorm

class TestConvReluNorm(unittest.TestCase):
    def setUp(self):
        self.model = ConvReluNorm(
            in_channels=10,
            hidden_channels=20,
            out_channels=20,
            kernel_size=3,
            n_layers=3,
        )

        self.x = torch.randn((2,6,10))

        self.x_mask = torch.tensor([
            [False, False, False, False, True, True],
            [False, False, False, False, False, False]
        ])
    
    def testForward(self):
        out = self.model.forward(
            x = self.x,
            x_mask = self.x_mask
        )

        self.assertTupleEqual(out.shape, (2,6,20))

if __name__ == "__main__":
    TestConvReluNorm.run()