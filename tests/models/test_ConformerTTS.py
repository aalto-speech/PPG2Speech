import torch
import unittest
from ppg_tts.models import ConformerTTS

class TestConformerTTSTorchAudio(unittest.TestCase):
    def setUp(self):
        self.model = ConformerTTS(ppg_dim=30,
                                  encode_dim=256,
                                  num_heads=2,
                                  num_layers=2,
                                  encode_ffn_dim=1024,
                                  encode_kernel_size=15,
                                  adapter_filter_size=256,
                                  adapter_kernel_size=3,
                                  n_bins=256,
                                  energy_min=1,
                                  energy_max=100,
                                  pitch_min=1,
                                  pitch_max=100,
                                  spk_emb_size=16,
                                  emb_hidden_size=64)
        
        self.x = torch.randn((4, 10, 30))

        self.x_length = torch.tensor([4, 4, 6, 10])

        self.spk_emb = torch.randn((4, 16))

        self.pitch_target = torch.randn((4, 13))
        self.energy_target = torch.randn((4, 13))

        self.energy_length = torch.tensor([5, 6, 7, 13])

        self.mel_mask = torch.BoolTensor([
            [False,False,False,False,False,True,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,True,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,True,True,True,True,True,True],
            [False,False,False,False,False,False,False,False,False,False,False,False,False]
        ])

    
    def testForward(self):
        y, pitch_pred, energy_pred = self.model(self.x,
                                                self.x_length,
                                                self.spk_emb,
                                                self.pitch_target,
                                                self.energy_target,
                                                self.energy_length,
                                                self.mel_mask)
        
        self.assertTupleEqual(y.shape, (4, 13, 80))
        self.assertTupleEqual(pitch_pred.shape, (4, 13))
        self.assertTupleEqual(energy_pred.shape, (4, 13))

class TestConformerTTSSpeechBrain(unittest.TestCase):
    def setUp(self):
        return super().setUp()
    
    def testForward(self):
        pass

if __name__ == "__main__":
    TestConformerTTSTorchAudio.run()