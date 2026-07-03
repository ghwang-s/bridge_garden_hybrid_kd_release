from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional


RoleList = List[str]


@dataclass
class SpecialTokenIds:
    pad: int
    bos: int
    eos: int


@dataclass
class SequenceRecord:
    """One generated sample with explicit oracle roles for later verifications."""

    token_ids: List[int]
    tokens: List[str]
    oracle_roles: RoleList
    latent_trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationConfig:
    train_size: int
    val_size: int
    test_size: int
    analysis_size: int
    min_len: int
    max_len: int
    seed: int = 42


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    teacher: ModelConfig
    student: ModelConfig
    hybrid_lambdas: List[float]
    seed: int = 42


@dataclass
class EvalConfig:
    n_rollouts: int
    n_window_rollouts: int
    window_sizes: List[int]
    kappa_n_alts: int
    kappa_future_len: int
    min_region_coverage: int = 10


@dataclass
class ExperimentConfig:
    domain: str
    artifact_root: str
    generation: GenerationConfig
    train: TrainConfig
    eval: EvalConfig

    def artifact_path(self) -> Path:
        return Path(self.artifact_root).expanduser().resolve()


@dataclass
class DomainDefinition:
    name: str
    vocab: List[str]
    token_to_id: Dict[str, int]
    special_ids: SpecialTokenIds
    default_config: Mapping[str, Any]
    generate_split: Callable[[str, GenerationConfig], List[SequenceRecord]]
    sample_alternatives: Callable[[SequenceRecord, int, int], List[int]]
    teacher_action_ids: Optional[Callable[[SequenceRecord, int], List[int]]] = None
    build_example_payload: Optional[Callable[[SequenceRecord], Dict[str, Any]]] = None


@dataclass
class RegionMaskBundle:
    coverage_counts: List[int]
    valid_positions: List[int]
    bridge_positions: List[int]
    garden_positions: List[int]
    bridge_threshold: float
    garden_threshold: float
    position_mean_kappa: List[float]


@dataclass
class MethodEpochSummary:
    overall_eb: float
    bridge_eb: float
    garden_eb: float
    overall_tf_kl: float
    overall_rollout_kl: float
    valid_steps: int


@dataclass
class ExperimentSummary:
    config: Dict[str, Any]
    bridge_garden: Dict[str, Any]
    methods: Dict[str, Any]
    example_payloads: List[Dict[str, Any]]
