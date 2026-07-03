# Manifest

## Synthetic Bridge-Garden Code

- `bridge_garden_v2/domains/`: domain definitions for dialogue, math, and code.
- `bridge_garden_v2/exact_oracle.py`, `exact_regions.py`, `scripted_oracle.py`: exact oracle and exact-region utilities.
- `bridge_garden_v2/oracle_dataset.py`, `oracle_student.py`, `oracle_training.py`, `oracle_eval.py`: synthetic dataset, student model, training, and evaluation code.
- `bridge_garden_v2/synthetic_*.py`: synthetic domain/evaluator/manifest/visualization support.
- `bridge_garden_v2/tests/`: regression tests for metrics, exact regions, oracle behavior, manifests, checks, visualizations, and seed verification.
- `scripts/run_synthetic_mini_pipeline.py`: small end-to-end synthetic pipeline.
- `scripts/render_synthetic_readable_heatmaps.py`: readable token-level kappa heatmap rendering.
- `scripts/check_synthetic_exact_cost.py`, `check_synthetic_oracle_cost.py`, `build_synthetic_cost_summary.py`: exact-cost and oracle-cost checks for the controlled synthetic setup.
- `scripts/build_synthetic_manifest.py`, `check_synthetic_config.py`: experiment integrity checks and manifest utilities.

## LLM Hybrid KD Code

- `llm_hybrid_kd/trl_trainer/hybrid_gkd_trainer.py`: Hybrid KD trainer combining soft KD and teacher-argmax hard-label supervision.
- `llm_hybrid_kd/trl_trainer/gkd_trainer.py`, `gkd_config.py`: soft/on-policy GKD base trainer and configuration.
- `llm_hybrid_kd/trl_trainer/sft_trainer.py`, `sft_config.py`: SFT/KD trainer extensions used for the control studies.
- `llm_hybrid_kd/trl_trainer/on_policy_distill_trainer.py`, `on_policy_distill_config.py`: on-policy distillation compatibility code.
- `llm_hybrid_kd/train_scripts/`: training entrypoints for SFT, soft GKD, Hybrid GKD, and on-policy distillation.
- `llm_hybrid_kd/examples/`: launch examples.
- `llm_hybrid_kd/eval/`: AlpacaEval-style judge utilities.
