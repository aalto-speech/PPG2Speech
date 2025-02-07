import torch
import unittest
from ppg_tts.models.encoder import ConvReluNorm, RelPosTransformerWrapper, RoFormerWrapper

class TestRoFormerWrapper(unittest.TestCase):
    def setUp(self):
        self.trans = RoFormerWrapper(
            input_dim=20, # (input_dim / 2) % nhead == 0
            ffn_dim=80,
            nhead=2,
            nlayers=2,
            dropout=0.1,
        )

        self.x = torch.randn((2,8,20))

        self.x_mask = torch.tensor([
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False, False]
        ])

    def testForward(self):
        out = self.trans.forward(
            x = self.x,
            x_mask = self.x_mask,
        )

        self.assertTupleEqual(out.shape, (2,8,20))

class TestRelPosTransformerWrapper(unittest.TestCase):
    def setUp(self):
        self.trans = RelPosTransformerWrapper(
            input_dim=10,
            ffn_dim=40,
            nhead=2,
            nlayers=2,
            dropout=0.1,
            transformer_type='transformer',
        )

        self.con = RelPosTransformerWrapper(
            input_dim=10,
            ffn_dim=40,
            nhead=2,
            nlayers=2,
            dropout=0.1,
            kernel_size=3,
        )

        self.x = torch.randn((2,8,10))

        self.x_mask = torch.tensor([
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False, False]
        ])

    def testTransformerForward(self):
        out = self.trans.forward(
            x = self.x,
            x_mask = self.x_mask,
        )

        self.assertTupleEqual(out.shape, (2,8,10))

    def testConformerForward(self):
        out = self.con.forward(
            x = self.x,
            x_mask = self.x_mask,
        )

        self.assertTupleEqual(out.shape, (2,8,10))

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