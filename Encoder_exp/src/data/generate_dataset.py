
import argparse
import yaml
import numpy as np
from pathlib import Path

from .simulator import SimConfig, TrajectorySimulator


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate simulated dataset (npz) for cross-modal tracking")
    parser.add_argument("--config", type=str, required=True, help="configs/simulate.yaml")
    parser.add_argument("--out", type=str, required=True, help="output npz file path")
    args = parser.parse_args()

    cfg_dict = load_yaml(args.config)

    cfg = SimConfig(
        n_train=cfg_dict.get("n_train", 12000),
        n_val=cfg_dict.get("n_val", 1500),
        n_test=cfg_dict.get("n_test", 1500),
        seed=cfg_dict.get("seed", 2025),
        T_vis_max=cfg_dict.get("T_vis_max", 64),
        T_ais_max=cfg_dict.get("T_ais_max", 24),
        dt_vis=cfg_dict.get("dt_vis", 1.0),
        ais_dt_range=tuple(cfg_dict.get("ais_dt_range", [2.0, 8.0])),
        v_range=tuple(cfg_dict.get("v_range", [2.0, 6.0])),
        turn_prob=cfg_dict.get("turn_prob", 0.2),
        pix_noise_vis=cfg_dict.get("pix_noise_vis", 1.5),
        pos_noise_ais=cfg_dict.get("pos_noise_ais", 3.0),
        miss_vis_prob=cfg_dict.get("miss_vis_prob", 0.06),
        miss_ais_prob=cfg_dict.get("miss_ais_prob", 0.04),
        occ_len_range=tuple(cfg_dict.get("occ_len_range", [2, 6])),
        dt_enc_omega=cfg_dict.get("dt_enc_omega", 0.5),
    )

    sim = TrajectorySimulator(cfg)

    print(f"[Sim] Generating train={cfg.n_train}, val={cfg.n_val}, test={cfg.n_test} ...")

    train_pack = sim.simulate_split(cfg.n_train, seed=cfg.seed + 1)
    val_pack = sim.simulate_split(cfg.n_val, seed=cfg.seed + 2)
    test_pack = sim.simulate_split(cfg.n_test, seed=cfg.seed + 3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim.save_npz(dict(train=train_pack, val=val_pack, test=test_pack), str(out_path))

    def shape_of(pack):
        return {k: v.shape for k, v in pack.items()}

    print("[Done] Saved to", out_path)
    print("  Train shapes:", shape_of(train_pack))
    print("  Val   shapes:", shape_of(val_pack))
    print("  Test  shapes:", shape_of(test_pack))


if __name__ == "__main__":
    main()
