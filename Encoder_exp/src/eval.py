
import argparse, os, json
import yaml
import torch
import numpy as np

from src.data.datasets import build_loader
from src.models.factory import build_dual_encoder
from src.utils.seed import set_seed
from src.utils.common import load_checkpoint, cosine_similarity_matrix, tensor_to_numpy
from src.utils.metrics import topk_accuracy, roc_auc, hungarian_id_accuracy, mofa

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_embeddings(model, loader, device):
    model.eval()
    zs_v, zs_a = [], []
    with torch.no_grad():
        for batch in loader:
            vis_seq = batch["vis_seq"].to(device)
            vis_mask = batch["vis_mask"].to(device)
            ais_seq = batch["ais_seq"].to(device)
            ais_mask = batch["ais_mask"].to(device)
            z_v, z_a = model(vis_seq, vis_mask, ais_seq, ais_mask)
            zs_v.append(z_v); zs_a.append(z_a)
    z_v = torch.cat(zs_v, dim=0)
    z_a = torch.cat(zs_a, dim=0)
    return z_v, z_a

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)  # eval.yaml
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Load model config from checkpoint
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model_cfg = ckpt["cfg"]["model"]
    model = build_dual_encoder(model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    # Data loader
    variant = "kine" if model_cfg["type"] in ["exp", "trans_kine"] else "raw"
    loader = build_loader(args.data, split=args.split, batch_size=cfg.get("batch_size", 128),
                          feature_variant=variant, return_pos=True, drop_last=False, shuffle=False)

    # Embeddings & similarity
    z_v, z_a = extract_embeddings(model, loader, device)
    sim = cosine_similarity_matrix(z_v, z_a)  # [N,N]
    sim_np = tensor_to_numpy(sim)

    # Metrics
    res = dict(
        top1=topk_accuracy(sim_np, k=1),
        top5=topk_accuracy(sim_np, k=min(5, sim_np.shape[1])),
        auc=roc_auc(sim_np),
        hungarian_acc=hungarian_id_accuracy(sim_np),
        mofa=mofa(sim_np),
    )

    print(json.dumps(res, indent=2, ensure_ascii=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
