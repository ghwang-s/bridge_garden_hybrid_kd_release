#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from .pipeline import smoke_generate, train_experiment
from .schema import EvalConfig, ExperimentConfig, GenerationConfig, ModelConfig, TrainConfig


def build_default_config(domain: str, artifact_root: str, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        domain=domain,
        artifact_root=artifact_root,
        generation=GenerationConfig(
            train_size=512,
            val_size=128,
            test_size=128,
            analysis_size=96,
            min_len=24,
            max_len=128,
            seed=seed,
        ),
        train=TrainConfig(
            epochs=4,
            batch_size=32,
            lr=3e-4,
            teacher=ModelConfig(d_model=128, n_heads=4, n_layers=3, d_ff=256),
            student=ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128),
            hybrid_lambdas=[0.2, 0.5, 0.8],
            seed=seed,
        ),
        eval=EvalConfig(
            n_rollouts=64,
            n_window_rollouts=32,
            window_sizes=[1, 3, 5],
            kappa_n_alts=8,
            kappa_future_len=8,
            min_region_coverage=10,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="bridge_garden_v2 experiment runner")
    parser.add_argument("--domain", choices=["dialogue", "code", "math"], required=True)
    parser.add_argument("--artifact-root", default="bridge_garden_v2/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--analysis-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hybrid-lambdas", type=str, help="Comma-separated lambda list, e.g. 0.1,0.2,0.5,0.8")
    args = parser.parse_args()

    config = build_default_config(args.domain, args.artifact_root, args.seed)
    if args.train_size is not None:
        config.generation.train_size = args.train_size
    if args.val_size is not None:
        config.generation.val_size = args.val_size
    if args.test_size is not None:
        config.generation.test_size = args.test_size
    if args.analysis_size is not None:
        config.generation.analysis_size = args.analysis_size
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.hybrid_lambdas:
        config.train.hybrid_lambdas = [float(x.strip()) for x in args.hybrid_lambdas.split(",") if x.strip()]
    if args.mode == "smoke":
        report = smoke_generate(args.domain, config)
    else:
        report = train_experiment(args.domain, config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
