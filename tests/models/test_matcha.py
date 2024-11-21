import unittest
import torch
from ppg_tts.models.matcha.decoder import Decoder
from ppg_tts.models.matcha.flow_matching import CFM
from ppg_tts.models.matcha.RoPE import RotaryPositionalEmbeddings
from ppg_tts.models import ConformerMatchaTTS

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
        self.model = CFM(
            in_channels=80,
            out_channel=80,
            n_spks=50,
            spk_emb_dim=512,
            cfm_params={
                'solver': None,
                'sigma_min': 1e-3
            },
            decoder_params={
                'down_block_type': 'conformer',
                'mid_block_type': 'conformer',
                'up_block_type': 'conformer'
            },
        )

        self.mu = torch.randn((4, 80, 8))
        self.mask = torch.FloatTensor([
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,1,1,1,0],
            [1,1,1,1,1,1,1,1]
        ]).unsqueeze(1)

        self.n_timesteps = 10
        self.spk_emb = torch.randn((4, 512))
        self.target = torch.randn((4, 80, 8))

    def testForward(self):
        output = self.model.forward(self.mu,
                                    self.mask,
                                    self.n_timesteps,
                                    spks=self.spk_emb)
        
        self.assertTupleEqual(output.shape, (4, 80, 8))

    def testComputeLoss(self):
        loss, y = self.model.compute_loss(self.target,
                                          self.mask,
                                          self.mu,
                                          self.spk_emb)
        
        self.assertTupleEqual(y.shape, (4, 80, 8))

class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.module = RotaryPositionalEmbeddings(d=128)
        self.x = torch.randn((4, 1, 8, 128))

    def testForward(self):
        out = self.module(self.x)

        self.assertTupleEqual(out.shape, (4,1,8,128))

class TestConformerMatchaTTS(unittest.TestCase):
    def setUp(self):
        self.model = ConformerMatchaTTS(
            ppg_dim=34,
            encode_dim=128,
            encode_heads=4,
            encode_layers=1,
            encode_ffn_dim=128,
            encode_kernel_size=9,
            spk_emb_size=512,
        )

        self.x = torch.randn((4, 8, 34))
        self.spk_emb = torch.randn((4,512))
        self.pitch = torch.randn((4, 8))
        self.v_flag = torch.randn((4,8))
        self.energy_length = torch.LongTensor([
            4, 5, 6, 8
        ])
        self.mel_target = torch.randn((4,8,80))
        self.mel_mask = torch.FloatTensor([
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,1,0,0,0],
            [1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,1,1]
        ])

    def testForward(self):
        loss, y = self.model.forward(
            x=self.x,
            spk_emb=self.spk_emb,
            pitch_target=self.pitch,
            v_flag=self.v_flag,
            energy_length=self.energy_length,
            mel_mask=self.mel_mask,
            mel_target=self.mel_target
        )

        self.assertTupleEqual(y.shape, (4, 80, 8))


if __name__ == "__main__":
    TestDecoderConformer.run()
    TestDecoderTransformer.run()
    TestCFM.run()
    TestRoPE.run()
    TestConformerMatchaTTS.run()
