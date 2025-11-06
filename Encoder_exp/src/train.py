
import argparse, os, json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from src.data.datasets import build_loader
from src.models.factory import build_dual_encoder
from src.losses.nt_xent import NTXentLoss
from src.utils.seed import set_seed
from src.utils.logger import Logger
from src.utils.common import save_checkpoint, cosine_similarity_matrix, tensor_to_numpy
from src.utils.metrics import topk_accuracy

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_dataloaders(data_path: str, model_type: str, batch_size: int, num_workers: int):
    # 'exp' 和 'trans_kine' 使用 'kine' 特征；'trans_raw' 使用 'raw'
    variant = "kine" if model_type in ["exp", "trans_kine"] else "raw"
    train_loader = build_loader(data_path, split="train", batch_size=batch_size,
                                feature_variant=variant, return_pos=False, drop_last=True)
    val_loader = build_loader(data_path, split="val", batch_size=batch_size,
                              feature_variant=variant, return_pos=False, drop_last=False, shuffle=False)
    return train_loader, val_loader

def validate(model, loader, device, temperature=0.07) -> Dict[str, float]:
    model.eval()
    zs_v, zs_a = [], []
    with torch.no_grad():
        for batch in loader:
            vis_seq = batch["vis_seq"].to(device)
            vis_mask = batch["vis_mask"].to(device)
            ais_seq = batch["ais_seq"].to(device)
            ais_mask = batch["ais_mask"].to(device)
            z_v, z_a = model(vis_seq, vis_mask, ais_seq, ais_mask)  # [B,d]
            zs_v.append(z_v); zs_a.append(z_a)
    z_v = torch.cat(zs_v, dim=0)
    z_a = torch.cat(zs_a, dim=0)
    sim = cosine_similarity_matrix(z_v, z_a)  # [N,N]
    sim_np = tensor_to_numpy(sim)
    top1 = topk_accuracy(sim_np, k=1)
    top5 = topk_accuracy(sim_np, k=5 if sim_np.shape[1] >= 5 else sim_np.shape[1])
    return {"top1": top1, "top5": top5}

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        vis_seq = batch["vis_seq"].to(device)
        vis_mask = batch["vis_mask"].to(device)
        ais_seq = batch["ais_seq"].to(device)
        ais_mask = batch["ais_mask"].to(device)

        z_v, z_a = model(vis_seq, vis_mask, ais_seq, ais_mask)
        loss = criterion(z_v, z_a)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="runs/exp")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    os.makedirs(args.workdir, exist_ok=True)

    # Seed & device
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Logging
    logger = Logger(args.workdir, use_tensorboard=True)
    logger.info(f"Config: {json.dumps(cfg, ensure_ascii=False)}")

    # Data
    batch_size = cfg.get("batch_size", 128)
    num_workers = cfg.get("num_workers", 2)
    train_loader, val_loader = build_dataloaders(args.data, cfg["model"]["type"], batch_size, num_workers)

    # Model
    model = build_dual_encoder(cfg["model"]).to(device)

    # Optimizer/Scheduler/criterion
    optimizer = optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-4))
    criterion = NTXentLoss(temperature=cfg.get("temperature", 0.07))

    best_top1 = -1.0
    best_path = os.path.join(args.workdir, "best.pt")

    epochs = cfg.get("epochs", 80)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch}/{epochs} | train loss: {train_loss:.4f}")
        logger.add_scalar("train/loss", train_loss, epoch)

        # Validate
        metrics = validate(model, val_loader, device, temperature=cfg.get("temperature", 0.07))
        logger.info(f"Epoch {epoch} | val top1: {metrics['top1']:.4f} | top5: {metrics['top5']:.4f}")
        logger.add_scalar("val/top1", metrics["top1"], epoch)
        logger.add_scalar("val/top5", metrics["top5"], epoch)

        # Save best by top1
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            save_checkpoint({"model": model.state_dict(), "cfg": cfg}, best_path)
            logger.info(f"[Best] top1={best_top1:.4f} saved to {best_path}")

    logger.close()


if __name__ == "__main__":
    main()
