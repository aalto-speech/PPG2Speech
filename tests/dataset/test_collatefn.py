import unittest
import torch
import numpy as np
from ppg_tts.dataset import PersoCollateFn

class TestPersoCollateFn(unittest.TestCase):
    def setUp(self):
        dummy_batch_lst = [
            {
                "melspectrogram": torch.randn((4,10)),
                "ppg": torch.randn((3,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(4),
                "energy": torch.randn(4)
            },
            {
                "melspectrogram": torch.randn((6,10)),
                "ppg": torch.randn((4,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(6),
                "energy": torch.randn(6)
            },
            {
                "melspectrogram": torch.randn((9,10)),
                "ppg": torch.randn((7,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(9),
                "energy": torch.randn(9)
            },
            {
                "melspectrogram": torch.randn((13,10)),
                "ppg": torch.randn((11,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(13),
                "energy": torch.randn(13)
            },
        ]

        self.mel_batch, self.mel_mask, self.ppg_batch, self.ppg_mask, \
            self.spk_emb_batch, self.spk_emb_mask, self.log_F0_batch, \
            self.log_F0_mask, self.energy_batch, self.energy_mask = PersoCollateFn(dummy_batch_lst)
        
    def testMel(self):
        self.assertTupleEqual(self.mel_batch.shape, (4, 13, 10),
                              "The batched melspectrogram shape is not correct")
        
        ref_mask = np.array([
            [False,False,False,False,True,True,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False,False,False]
        ])

        np.testing.assert_allclose(ref_mask, self.mel_mask.numpy())

    def testPPG(self):
        self.assertTupleEqual(self.ppg_batch.shape, (4, 11, 20),
                              "The batched ppg shape is not correct")
        
        ref_mask = np.array([
            [False,False,False,True,True,True,True,True,True,True,True],
            [False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False]
        ])

        np.testing.assert_allclose(ref_mask, self.ppg_mask.numpy())

    def testLogF0(self):
        self.assertTupleEqual(self.log_F0_batch.shape, (4, 13))

        ref_mask = np.array([
            [False,False,False,False,True,True,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False,False,False]
        ])

        np.testing.assert_allclose(ref_mask, self.log_F0_mask)

    def testEnergy(self):
        self.assertTupleEqual(self.energy_batch.shape, (4, 13))

        ref_mask = np.array([
            [False,False,False,False,True,True,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False,False,False]
        ])

        np.testing.assert_allclose(ref_mask, self.energy_mask)


if __name__ == "__main__":
    TestPersoCollateFn.run()