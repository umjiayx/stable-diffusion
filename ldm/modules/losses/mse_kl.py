import torch
import torch.nn as nn
import torch.nn.functional as F

class MSEWithKL(nn.Module):
    """
    L = MSE(recon, inputs) + kl_weight * KL
    Matches AutoencoderKL call signature:
      self.loss(inputs, reconstructions, posterior, optimizer_idx, global_step, ...)
    The optimizer_idx is ignored.
    """

    def __init__(self, kl_weight: float = 1e-6):
        super().__init__()
        self.kl_weight = float(kl_weight)

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posterior=None,
        optimizer_idx: int = 0,
        global_step: int = 0,
        last_layer=None,
        cond=None,
        split: str = "train",
        **kwargs,
    ):
        # recon term
        rec_loss = F.mse_loss(reconstructions, inputs)

        # KL term if available
        kl_loss = torch.tensor(0.0, device=reconstructions.device)
        if posterior is not None and hasattr(posterior, "kl"):
            kl = posterior.kl()
            kl_loss = kl.mean()

        loss = rec_loss + self.kl_weight * kl_loss

        log = {
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/kl_loss":  kl_loss.detach(),
            f"{split}/loss":     loss.detach(),
        }
        return loss, log