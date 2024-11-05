import unittest
import torch
import numpy as np
from ppg_tts.dataset import PersoCollateFn

class TestPersoCollateFn(unittest.TestCase):
    def setUp(self):
        dummy_batch_lst = [
            {
                "melspectrogram": torch.randn((10,4)),
                "ppg": torch.randn((3,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(4),
                "energy": torch.randn(4),
                "v_flag": torch.rand(4) > 0.6,
            },
            {
                "melspectrogram": torch.randn((10,6)),
                "ppg": torch.randn((4,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(6),
                "energy": torch.randn(6),
                "v_flag": torch.rand(6) > 0.6,
            },
            {
                "melspectrogram": torch.randn((10,9)),
                "ppg": torch.randn((7,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(9),
                "energy": torch.randn(9),
                "v_flag": torch.rand(9) > 0.6,
            },
            {
                "melspectrogram": torch.randn((10,13)),
                "ppg": torch.randn((11,20)),
                "spk_emb": torch.randn(16),
                "log_F0": torch.randn(13),
                "energy": torch.randn(13),
                "v_flag": torch.rand(13) > 0.6,
            },
        ]

        self.batch_dict = PersoCollateFn(dummy_batch_lst)
        
    def testMel(self):
        self.assertTupleEqual(self.batch_dict["mel"].shape, (4, 13, 10),
                              "The batched melspectrogram shape is not correct")
        
        ref_mask = np.array([
            [False,False,False,False,True,True,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False,False,False]
        ])

        np.testing.assert_allclose(ref_mask, self.batch_dict["mel_mask"].numpy())

    def testPPG(self):
        self.assertTupleEqual(self.batch_dict["ppg"].shape, (4, 11, 20),
                              "The batched ppg shape is not correct")
        
        ref_mask = np.array([
            [False,False,False,True,True,True,True,True,True,True,True],
            [False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False]
        ])

        ref_length = np.array([3,4,7,11])

        np.testing.assert_allclose(ref_mask, self.batch_dict["ppg_mask"].numpy())
        np.testing.assert_allclose(ref_length, self.batch_dict["ppg_len"].numpy())

    def testLogF0(self):
        self.assertTupleEqual(self.batch_dict["log_F0"].shape, (4, 13))

        ref_mask = np.array([
            [False,False,False,False,True,True,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False,False,False]
        ])

        np.testing.assert_allclose(ref_mask, self.batch_dict["log_F0_mask"].numpy())

        ref_length = np.array([4,6,9,13])

        np.testing.assert_allclose(ref_length, self.batch_dict["log_F0_len"].numpy())

    def testEnergy(self):
        self.assertTupleEqual(self.batch_dict["energy"].shape, (4, 13))

        ref_mask = np.array([
            [False,False,False,False,True,True,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False,False,False]
        ])

        np.testing.assert_allclose(ref_mask, self.batch_dict["energy_mask"].numpy())

        ref_length = np.array([4,6,9,13])

        np.testing.assert_allclose(ref_length, self.batch_dict["energy_len"].numpy())

    def testVFlag(self):
        self.assertTupleEqual(self.batch_dict["v_flag"].shape, (4, 13))

if __name__ == "__main__":
    TestPersoCollateFn.run()