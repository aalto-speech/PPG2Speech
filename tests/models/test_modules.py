import torch
import unittest
from ppg_tts.models.modules import \
    VariancePredictor, SpeakerEmbeddingEncoder,\
    VarianceAdaptor, PitchEncoder, HiddenEncoder

class TestHiddenEncoder(unittest.TestCase):
    def setUp(self):
        self.module = HiddenEncoder(
            input_channel=12,
            output_channel=6,
            n_layers=1,
            transformer_type='transformer'
        )

        self.con = HiddenEncoder(
            input_channel=12,
            output_channel=6,
            n_layers=1,
            kernel_size=3,
        )

        self.x = torch.randn((4, 5, 12))
        self.mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ]).unsqueeze(-1)

    def testTransformerForward(self):
        out = self.module(self.x, self.mask)

        self.assertTupleEqual(out.shape, (4, 5, 6))

    def testConformerForward(self):
        out = self.con(self.x, self.mask)

        self.assertTupleEqual(out.shape, (4, 5, 6))

class TestPitchEncoder(unittest.TestCase):
    def setUp(self):
        self.module = PitchEncoder(8, -10, 10)

        self.pitch = torch.randn((2,3,1))
        self.v_flag = torch.randn((2,3,1))

        self.mask = torch.tensor([
            [False, True, True],
            [False, False, True]
        ],
        dtype=torch.bool).unsqueeze(-1)

    def testForward(self):
        enc_pitch = self.module.forward(
            self.pitch,
            self.v_flag,
            self.mask
        )

        self.assertTupleEqual(enc_pitch.shape, (2,3,9))

class TestVariancePredictor(unittest.TestCase):
    def setUp(self):
        self.module = VariancePredictor(input_size=256,
                                        filter_size=256,
                                        kernel_size=3,
                                        dropout=0.5)
        
        self.x = torch.randn((4, 10, 256))

    def testForward(self):
        y = self.module(self.x)

        self.assertTupleEqual(y.shape, (4, 10))

class TestSpeakerEmbeddingEncoder(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn((4,32))

        self.module = SpeakerEmbeddingEncoder(input_size=32,
                                              output_size=4)

    def testForward(self):
        y = self.module(self.x)
        self.assertTupleEqual(y.shape, (4,1,4))

class TestVarianceAdapter(unittest.TestCase):
    def setUp(self):
        self.module = VarianceAdaptor(input_size=256,
                                      filter_size=256,
                                      kernel_size=3,
                                      dropout=0.5,
                                      n_bins=100,
                                      pitch_min=1,
                                      pitch_max=100,
                                      energy_min=0,
                                      energy_max=1000)
        
        self.x = torch.randn((4, 10, 256))

        self.pitch = torch.randn((4, 10)) * 49
        self.energy = torch.randn((4, 10)) * 500

    def testForward(self):
        y, pitch_pred, energy_pred = self.module(self.x,
                                                 pitch_target=self.pitch,
                                                 energy_target=self.energy)
        
        self.assertTupleEqual(y.shape, (4, 10, 256))

        self.assertTupleEqual(pitch_pred.shape, (4, 10))
        self.assertTupleEqual(energy_pred.shape, (4, 10))

    def testForwardNoTarget(self):
        y, pitch_pred, energy_pred = self.module(self.x)
        
        self.assertTupleEqual(y.shape, (4, 10, 256))

        self.assertTupleEqual(pitch_pred.shape, (4, 10))
        self.assertTupleEqual(energy_pred.shape, (4, 10))

if __name__ == "__main__":
    TestVariancePredictor.run()
    TestSpeakerEmbeddingEncoder.run()
    TestVarianceAdapter.run()
    TestPitchEncoder.run()
    TestHiddenEncoder.run()