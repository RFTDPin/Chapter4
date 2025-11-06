#!/usr/bin/env bash
set -e
python -m src.data.generate_dataset --config configs/simulate.yaml --out data/sim_dataset.npz
