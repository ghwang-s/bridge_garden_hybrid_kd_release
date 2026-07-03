from __future__ import annotations

import argparse
import json

import torch

from bridge_garden_v2.synthetic_domains import build_synthetic_domain
from bridge_garden_v2.synthetic_evaluator import evaluate_oracle_states
from bridge_garden_v2.modeling import CausalTransformerLM
from bridge_garden_v2.oracle_student import NeuralStudentLoss
from bridge_garden_v2.schema import ModelConfig
from scripts.run_synthetic_mini_pipeline import _evaluate_methods
from scripts.run_synthetic_mini_pipeline import _evaluate_ref_kappa
from scripts.run_synthetic_mini_pipeline import _method_eval_devices


def test_method_eval_devices_defaults_to_fallback_device() -> None:
    assert _method_eval_devices("", torch.device("cpu")) == ["cpu"]
    assert _method_eval_devices("auto", torch.device("cpu")) == ["cpu"]
    assert _method_eval_devices("auto", torch.device("cuda:0")) == ["cuda:0"]
    assert _method_eval_devices("cuda:1,auto", torch.device("cuda:0")) == ["cuda:1", "cuda:0"]
    assert _method_eval_devices("cuda:0, cuda:1", torch.device("cpu")) == ["cuda:0", "cuda:1"]


def test_parallel_method_eval_matches_serial_cpu(tmp_path) -> None:
    bundle = build_synthetic_domain("code", sample_count=1, script_len=47, vocab_size=64)
    model_cfg = ModelConfig(d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0)
    torch.manual_seed(7)
    ce_ref = CausalTransformerLM(len(bundle.vocab), model_cfg, max_len=64, pad_id=bundle.oracle.pad_id)
    torch.manual_seed(11)
    hard_kd = CausalTransformerLM(len(bundle.vocab), model_cfg, max_len=64, pad_id=bundle.oracle.pad_id)
    models = {"ce_ref": ce_ref, "hard_kd": hard_kd}
    histories = {
        method: {
            "train_loss": [1.0],
            "val_teacher_forced_kl": [0.5],
            "val_early_stopping_objective": [0.5],
            "early_stopping_metric": "teacher_forced_kl",
            "early_stopping_relative_min_delta": 0.0,
            "best_epoch": 0,
            "best_val_early_stopping_objective": 0.5,
            "best_teacher_forced_kl_epoch": 0,
            "best_val_teacher_forced_kl": 0.5,
        }
        for method in models
    }
    states = _tail_clean_states(bundle.oracle, count=2)
    ref_loss = NeuralStudentLoss(ce_ref, vocab_size=len(bundle.vocab), violation_penalty=2.0, device=torch.device("cpu"))
    kappa_eval = evaluate_oracle_states(bundle.oracle, ref_loss, states, continuation="expectation")

    common = {
        "domain": "code",
        "sample_count": 1,
        "script_len": 47,
        "vocab_size": 64,
        "kappa_continuation": "expectation",
        "rollout_repeats": 1,
        "rollout_seed": 3,
        "violation_penalty": 2.0,
        "progress_json": None,
        "seed_dir_name": "seed_0",
        "train_sample_seed": 0,
        "analysis_sample_seed": 0,
        "model_seed": 0,
    }
    serial = _evaluate_methods(
        args=argparse.Namespace(**common, method_eval_devices=""),
        out_dir=tmp_path / "serial",
        bundle=bundle,
        model_cfg=model_cfg,
        model_max_len=64,
        models=models,
        histories=histories,
        kappa_eval=kappa_eval,
        analysis_ids=[0],
        fallback_device=torch.device("cpu"),
    )
    parallel = _evaluate_methods(
        args=argparse.Namespace(**common, method_eval_devices="cpu,cpu"),
        out_dir=tmp_path / "parallel",
        bundle=bundle,
        model_cfg=model_cfg,
        model_max_len=64,
        models=models,
        histories=histories,
        kappa_eval=kappa_eval,
        analysis_ids=[0],
        fallback_device=torch.device("cpu"),
    )

    assert json.dumps(parallel, sort_keys=True) == json.dumps(serial, sort_keys=True)


def test_parallel_ref_kappa_matches_serial_cpu(tmp_path) -> None:
    bundle = build_synthetic_domain("code", sample_count=1, script_len=47, vocab_size=64)
    model_cfg = ModelConfig(d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0)
    torch.manual_seed(13)
    model = CausalTransformerLM(len(bundle.vocab), model_cfg, max_len=64, pad_id=bundle.oracle.pad_id)
    states = _tail_clean_states(bundle.oracle, count=2)
    common = {
        "domain": "code",
        "sample_count": 1,
        "script_len": 47,
        "vocab_size": 64,
        "kappa_continuation": "expectation",
        "violation_penalty": 2.0,
        "progress_json": None,
        "seed_dir_name": "seed_0",
        "train_sample_seed": 0,
        "analysis_sample_seed": 0,
        "model_seed": 0,
    }

    serial = _evaluate_ref_kappa(
        args=argparse.Namespace(**common, method_eval_devices=""),
        out_dir=tmp_path / "serial_ref",
        bundle=bundle,
        model_cfg=model_cfg,
        model_max_len=64,
        ref_model=model,
        states=states,
        fallback_device=torch.device("cpu"),
    )
    parallel = _evaluate_ref_kappa(
        args=argparse.Namespace(**common, method_eval_devices="cpu,cpu"),
        out_dir=tmp_path / "parallel_ref",
        bundle=bundle,
        model_cfg=model_cfg,
        model_max_len=64,
        ref_model=model,
        states=states,
        fallback_device=torch.device("cpu"),
    )

    assert [score.state_id for score in parallel.scores] == [score.state_id for score in serial.scores]
    assert [score.kappa for score in parallel.scores] == [score.kappa for score in serial.scores]
    assert [result.q_values for result in parallel.kappa_results] == [result.q_values for result in serial.kappa_results]
    assert parallel.kappa_split == serial.kappa_split
    assert parallel.confidence_split == serial.confidence_split


def _tail_clean_states(oracle, *, count: int) -> list:
    state = oracle.initial_state(0)
    states = []
    while not oracle.is_terminal(state):
        states.append(state)
        dist = oracle.next_dist(state)
        action = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
        state = oracle.step(state, int(action))
    return states[-count:]
