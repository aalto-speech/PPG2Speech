import torch
import unittest
from ppg_tts.models.modules import VariancePredictor, SpeakerEmbeddingEncoder, VarianceAdaptor

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
        self.x = torch.randn((4,512))

        self.module = SpeakerEmbeddingEncoder(input_size=512,
                                              model_size=2048,
                                              output_size=256)

    def testForward(self):
        y = self.module(self.x)
        self.assertTupleEqual(y.shape, (4,256))

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

if __name__ == "__main__":
    TestVariancePredictor.run()
    TestSpeakerEmbeddingEncoder.run()
    TestVarianceAdapter.run()