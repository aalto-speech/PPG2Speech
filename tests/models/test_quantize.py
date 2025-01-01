import unittest
import torch
from ppg_tts.models.AutoEnc import QuantizeLayer

class TestQuantizeLayer(unittest.TestCase):
    def setUp(self):
        self.layer = QuantizeLayer(4, 4)

        self.x = torch.randn((2, 6, 4))

        self.mask = torch.tensor([
            [False, False, False, False, True, True],
            [False, False, False, False, False, True]
        ])

        self.x.masked_fill_(self.mask.unsqueeze(-1), 0.0)

    def testForward(self):
        z_q, e_loss, c_loss = self.layer(self.x, self.mask)

        self.assertTupleEqual(z_q.shape, (2,6,4))

if __name__ == '__main__':
    TestQuantizeLayer.run()
