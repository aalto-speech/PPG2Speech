import unittest
import torch
from ppg_tts.models.AutoEnc import AutoEncoder

class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.model = AutoEncoder(10,2,8)

        self.content = torch.randn((2,10,3))
        self.cond = torch.randn((2,8,3))

        self.mask = torch.tensor([
            [False, True, True],
            [False, False, True]
        ],
        dtype=torch.bool).unsqueeze(1)

    def testForward(self):
        z, x = self.model(self.content, self.cond, self.mask)

        self.assertTupleEqual(z.shape, (2,2,3))
        self.assertTupleEqual(x.shape, (2,10,3))

if __name__ == "__main__":
    TestAutoEncoder.run()