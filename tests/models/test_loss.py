import unittest
import torch
from ppg_tts.models.loss import speaker_contrasive_loss

class TestSpeakerContrasiveLoss(unittest.TestCase):
    def setUp(self):
        self.target = torch.randn((4, 10))
        self.pred = torch.randn((4, 10))

    def test_loss(self):
        loss = speaker_contrasive_loss(self.target, self.pred)

        self.assertTupleEqual(loss.shape, tuple())

if __name__ == "__main__":
    TestSpeakerContrasiveLoss.run()