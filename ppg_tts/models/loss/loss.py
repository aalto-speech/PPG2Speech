import torch
import torch.nn.functional as F

def speaker_contrasive_loss(target_spk_emb: torch.Tensor, pred_spk_emb: torch.Tensor) -> torch.Tensor:
    """
    Args:
        target_spk_emb: torch.Tensor with shape (B, E_spk).
        pred_spk_emb: torch.Tensor with shape (B, E_spk).
    Returns:
        A scalar loss wrapped by torch.Tensor represents the speaker contrasive loss.
    """
    B, E = target_spk_emb.shape
    if pred_spk_emb.shape[0] != B or pred_spk_emb.shape[1] != E:
        raise ValueError(f"shape of pred_spk_emb of {pred_spk_emb.shape} is not compatible with target shape ({B}, {E})")
    target_emb = F.normalize(target_spk_emb)
    pred_emb = F.normalize(pred_spk_emb)

    cosine_sim = torch.mm(pred_emb, target_emb.t())

    diag_sim = torch.diag(cosine_sim)

    numerator = torch.exp(diag_sim)

    denominator = torch.sum(torch.exp(cosine_sim), dim=1) - numerator

    loss = torch.log(denominator) - torch.log(numerator)

    return loss.mean()