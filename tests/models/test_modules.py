import torch
import unittest
from ppg_tts.models.modules import VariancePredictor, SpeakerEmbeddingEncoder

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

if __name__ == "__main__":
    TestVariancePredictor.run()
    TestSpeakerEmbeddingEncoder.run()