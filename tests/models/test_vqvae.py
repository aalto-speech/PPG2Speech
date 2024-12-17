import torch
import unittest
from ppg_tts.models.AutoEnc import VQVAE

class TestVQVAE(unittest.TestCase):
    def setUp(self):
        self.model = VQVAE(
            input_channel=8,
            hidden_channel=2,
            cond_channel=6,
            kernel_sizes=[1,3,3],
            dilations=[1,2,4],
            num_emb=4,
        )

        self.x = torch.randn((4,8,8))
        self.cond = torch.randn((4,6))
        self.mask = torch.tensor([
            [False, False, False, False, True, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, True, True],
            [False, False, False, False, False, False, False, False],
        ])

    def testForward(self):
        z_q, z_dec, e_loss, c_loss = self.model.forward(
            x = self.x,
            cond = self.cond,
            mask = self.mask
        )

        self.assertTupleEqual(z_q.shape, (4,8,2))

        self.assertTupleEqual(z_dec.shape, (4,8,8))

if __name__ == "__main__":
    TestVQVAE.run()
    