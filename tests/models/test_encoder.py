import torch
import unittest
from ppg_tts.models.encoder import ConvReluNorm, TransformerWrapper

class TestTransformerWrapper(unittest.TestCase):
    def setUp(self):
        self.trans = TransformerWrapper(
            input_dim=20,
            ffn_dim=40,
            nhead=2,
            nlayers=2,
            dropout=0.1,
            transformer_type='transformer',
        )

        self.con = TransformerWrapper(
            input_dim=20,
            ffn_dim=40,
            nhead=2,
            nlayers=2,
            dropout=0.1,
            kernel_size=3,
        )

        self.ro = TransformerWrapper(
            input_dim=20,
            ffn_dim=40,
            nhead=2,
            nlayers=2,
            dropout=0.1,
            kernel_size=3,
            transformer_type='roformer'
        )

        self.x = torch.randn((2,8,20))

        self.x_mask = torch.tensor([
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False, False]
        ])

    def testTransformerForward(self):
        out = self.trans.forward(
            x = self.x,
            x_mask = self.x_mask,
        )

        self.assertTupleEqual(out.shape, (2,8,20))

    def testConformerForward(self):
        out = self.con.forward(
            x = self.x,
            x_mask = self.x_mask,
        )

        self.assertTupleEqual(out.shape, (2,8,20))

    def testRoformerForward(self):
        out = self.ro.forward(
            x = self.x,
            x_mask = self.x_mask,
        )

        self.assertTupleEqual(out.shape, (2,8,20))

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