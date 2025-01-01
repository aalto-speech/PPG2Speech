import unittest
import torch
from ppg_tts.models.AutoEnc import AutoEncoder

class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.model = AutoEncoder(10,2,8, [1,3,3], [1,2,3])

        self.content = torch.randn((2,3,10))
        self.cond = torch.randn((2,8))

        self.mask = torch.tensor([
            [False, True, True],
            [False, False, True]
        ],
        dtype=torch.bool)

    def testForward(self):
        z, x = self.model(self.content, self.cond, self.mask)

        self.assertTupleEqual(z.shape, (2,3,2))
        self.assertTupleEqual(x.shape, (2,3,10))

if __name__ == "__main__":
    TestAutoEncoder.run()