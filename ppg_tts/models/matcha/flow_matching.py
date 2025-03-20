from abc import ABC

import torch
import torch.nn.functional as F

from .decoder import Decoder


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        if "sigma_min" in cfm_params:
            self.sigma_min = cfm_params['sigma_min']
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, t_span.shape[0]):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < t_span.shape[0] - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels * 2 + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)


class CFM_CFG(CFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64, cfg_prob=0.2, guidance_scale=1):
        super().__init__(in_channels, out_channel, cfm_params, decoder_params, n_spks, spk_emb_dim)

        self.cfg_prob = cfg_prob

        self.guidance_scale = guidance_scale

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, t_span.shape[0]):
            eps_cond = self.estimator(x, mask, mu, t, spks, cond)
            # Compute unconditional prediction by replacing spks with zeros
            null_spks = torch.zeros_like(spks)
            eps_uncond = self.estimator(x, mask, mu, t, null_spks, cond)
            dphi_dt = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < t_span.shape[0] - 1:
                dt = t_span[step + 1] - t

        return sol[-1]
    
    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss with classifier-free guidance training.
        
        Args:
            x1 (torch.Tensor): Target, shape (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): Target mask, shape (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): Output of encoder (text latent representations), shape (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): Condition (concat of speaker, pitch, voiced flag), 
                shape (batch_size, spk_emb_dim + pitch_emb_dim + 1, mel_timesteps). Defaults to None.
        
        Returns:
            loss: conditional flow matching loss
            y: conditional flow, shape (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # Apply classifier-free guidance dropout on the condition.
        # With probability cfg_prob, drop (zero out) the condition for the entire example.
        if spks is not None and hasattr(self, "cfg_prob") and self.cfg_prob > 0:
            # Create a dropout mask per sample
            drop_mask = (torch.rand(b, device=mu.device) < self.cfg_prob).float()  # shape (b,)
            # Reshape to broadcast to spks shape
            drop_mask = drop_mask.view(b, 1, 1)
            # Replace spks with zero for those examples
            spks = spks * (1 - drop_mask)

        # Random timestep
        t_rand = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # Sample noise p(x_0)
        z = torch.randn_like(x1)

        # Form the target (y) as a convex combination of noise and the target x1.
        y = (1 - (1 - self.sigma_min) * t_rand) * z + t_rand * x1
        # Compute the "flow" target u (a residual between x1 and a scaled noise term)
        u = x1 - (1 - self.sigma_min) * z

        # Get the estimator prediction with the (possibly dropped) condition.
        pred = self.estimator(y, mask, mu, t_rand.squeeze(), spks)
        loss = F.mse_loss(pred, u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y
