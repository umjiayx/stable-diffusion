#!/usr/bin/env python3
import os, sys, json, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

# repo root on path
sys.path.append(os.getcwd())

from ldm.util import instantiate_from_config

@torch.no_grad()
def psnr_from_mse(mse: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
    # data_range can be a scalar tensor or per-sample (B,)
    # PSNR = 20*log10(MAX) - 10*log10(MSE)
    eps = 1e-12
    return 20.0 * torch.log10(data_range.clamp_min(eps)) - 10.0 * torch.log10(mse.clamp_min(eps))

def save_image_grid(x_bchw: torch.Tensor, path: str, per_image_minmax: bool = True):
    try:
        from torchvision.utils import make_grid, save_image
    except Exception:
        print("[warn] torchvision not available; skipping PNG save.")
        return
    B = min(8, x_bchw.size(0))
    x = x_bchw[:B].clone()
    if per_image_minmax:
        for i in range(B):
            xi = x[i]
            lo, hi = xi.min(), xi.max()
            if (hi - lo) > 1e-9:
                x[i] = (xi - lo) / (hi - lo)
            else:
                x[i] = xi.clamp(0, 1)
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    grid = make_grid(x, nrow=B, padding=2)
    save_image(grid, path)


# Example command:
# CUDA_VISIBLE_DEVICES=0 python eval_AutoencoderKL_jiayx.py 
# --config configs/autoencoder/QG_v3_test_autoencoder_kl_64x64x3.yaml 
# --output_dir eval_outputs_qg_v3
def build_parser():
    p = argparse.ArgumentParser("Evaluate AutoencoderKL on a test set")
    p.add_argument("--config", "-c", type=str, required=True, help="Eval YAML (has data.test)")
    p.add_argument("--output_dir", "-o", type=str, default="eval_outputs", help="Where to save results")
    p.add_argument("--max_batches", type=int, default=None, help="Limit number of test batches")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p

def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = OmegaConf.load(args.config)

    # Instantiate model (ckpt_path should be set inside cfg.model.params if you want to restore weights)
    if "params" not in cfg.model:
        cfg.model.params = OmegaConf.create()
    model = instantiate_from_config(cfg.model).to(args.device).eval()

    # Instantiate datamodule (dataset applies normalize_const from YAML)
    data = instantiate_from_config(cfg.data)
    data.setup("test")
    if not hasattr(data, "datasets") or "test" not in data.datasets:
        raise RuntimeError("No test dataset found. Add data.params.test in the YAML.")
    test_loader = data._test_dataloader(shuffle=False)



    # Progress bar setup
    try:
        total_batches = len(test_loader)
    except TypeError:
        total_batches = None
    if args.max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, args.max_batches)


    total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    psnr_sum = 0.0
    nrmse_sum = 0.0
    saved_grids = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, total=total_batches, desc="Evaluating", dynamic_ncols=True)
        for b_idx, batch in enumerate(pbar):
            if args.max_batches is not None and b_idx >= args.max_batches:
                break

            # Dataset returns HWC float; model.get_input → BCHW float
            inputs = model.get_input(batch, getattr(model, "image_key", "image")).to(args.device)  # (B,C,H,W)

            recons, _ = model(inputs, sample_posterior=False)  # deterministic eval

            assert recons.shape == inputs.shape, f"shape mismatch: {tuple(recons.shape)} vs {tuple(inputs.shape)}"

            # Per-sample MSE/MAE
            per_elem_mse = (recons - inputs).pow(2)
            per_elem_mae = (recons - inputs).abs()
            mse = per_elem_mse.view(per_elem_mse.size(0), -1).mean(dim=1)  # (B,)
            mae = per_elem_mae.view(per_elem_mae.size(0), -1).mean(dim=1)  # (B,)

            # NRMSE: sqrt(mean((x-y)^2)) / sqrt(mean(y^2))
            per_elem_gt_squared = inputs.pow(2)
            rmse = torch.sqrt(per_elem_mse.view(per_elem_mse.size(0), -1).mean(dim=1))  # (B,)
            rms_gt = torch.sqrt(per_elem_gt_squared.view(per_elem_gt_squared.size(0), -1).mean(dim=1))  # (B,)
            nrmse = rmse / rms_gt.clamp_min(1e-8)  # (B,)

            # PSNR with data-driven range: MAX = (x.max - x.min) per-sample based on inputs
            # (You can switch to a fixed constant if you prefer.)
            x_min = inputs.view(inputs.size(0), -1).min(dim=1).values
            x_max = inputs.view(inputs.size(0), -1).max(dim=1).values
            data_range = (x_max - x_min).clamp_min(1e-6)  # (B,)
            psnr = psnr_from_mse(mse, data_range=data_range.to(mse.device))

            mse_sum += mse.sum().item()
            mae_sum += mae.sum().item()
            psnr_sum += psnr.sum().item()
            nrmse_sum += nrmse.sum().item()
            total += mse.numel()

            # Save a few visual grids
            if saved_grids < 4:
                save_image_grid(inputs.detach().cpu(), str(out_dir / f"inputs_b{b_idx:04d}.png"))
                save_image_grid(recons.detach().cpu(), str(out_dir / f"recons_b{b_idx:04d}.png"))
                torch.save(inputs.cpu(), out_dir / f"inputs_b{b_idx:04d}.pt")
                torch.save(recons.cpu(), out_dir / f"recons_b{b_idx:04d}.pt")
                saved_grids += 1

    results = {
        "num_samples": total,
        "mse": mse_sum / total if total else None,
        "mae": mae_sum / total if total else None,
        "psnr": psnr_sum / total if total else None,
        "nrmse": nrmse_sum / total if total else None,
        "psnr_range": "data-driven: (x_max - x_min) per sample on scaled inputs"
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()