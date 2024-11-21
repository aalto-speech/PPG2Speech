import unittest
import torch
from ppg_tts.models.matcha.decoder import Decoder
from ppg_tts.models.matcha.flow_matching import CFM

class TestDecoderConformer(unittest.TestCase):
    def setUp(self):
        self.model = Decoder(
            in_channels=768,
            out_channels=80,
            down_block_type='conformer',
            mid_block_type='conformer',
            up_block_type='conformer'
        )

        self.noise = torch.randn((4, 128, 8))
        self.t = torch.randn((4,))
        self.mu = torch.randn(4, 128, 8)
        self.spk_emb = torch.randn((4, 512))
        self.mask = torch.FloatTensor([
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1]
        ]).unsqueeze(1)

    def testForward(self):
        output = self.model.forward(
            x=self.noise,
            mask=self.mask,
            mu=self.mu,
            t=self.t,
            spks=self.spk_emb
        )

        self.assertTupleEqual(output.shape, (4, 80, 8))
    
class TestDecoderTransformer(unittest.TestCase):
    def setUp(self):
        self.model = Decoder(
            in_channels=768,
            out_channels=80,
            down_block_type='transformer',
            mid_block_type='transformer',
            up_block_type='transformer',
            act_fn='gelu'
        )

        self.noise = torch.randn((4, 128, 8))
        self.t = torch.randn((4,))
        self.mu = torch.randn(4, 128, 8)
        self.spk_emb = torch.randn((4, 512))
        self.mask = torch.FloatTensor([
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1]
        ]).unsqueeze(1)

    def testForward(self):
        output = self.model.forward(
            x=self.noise,
            mask=self.mask,
            mu=self.mu,
            t=self.t,
            spks=self.spk_emb
        )

        self.assertTupleEqual(output.shape, (4, 80, 8))
    

class TestCFM(unittest.TestCase):
    def setUp(self):
        return super().setUp()

if __name__ == "__main__":
    TestDecoderConformer.run()
    TestDecoderTransformer.run()
