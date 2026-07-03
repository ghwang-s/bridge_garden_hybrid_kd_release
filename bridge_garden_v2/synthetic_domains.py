from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .scripted_oracle import ScriptedOracle, ScriptStep


SPECIAL = ("<pad>", "<bos>", "<eos>")


@dataclass(frozen=True)
class SyntheticDomainBundle:
    name: str
    vocab: tuple[str, ...]
    oracle: ScriptedOracle
    high_risk_tags: tuple[str, ...]
    flexible_tags: tuple[str, ...]


DOMAIN_TAGS = {
    "code": {
        "high": ("operator", "branch_guard", "return_semantics"),
        "flex": ("equivalent_implementation",),
    },
    "math": {
        "high": ("computed_value", "operator", "substitution", "final_answer"),
        "flex": ("equivalent_representation",),
    },
    "dialogue": {
        "high": ("required_fact", "date_time", "recipient", "forbidden_constraint"),
        "flex": ("tone_paraphrase",),
    },
    "negative_control": {
        "high": (),
        "flex": ("surface_ngram",),
    },
}

DOMAIN_SCRIPT_LENS = {
    "code": 47,
    "math": 43,
    "dialogue": 35,
    "negative_control": 35,
}

FLEXIBLE_SUPPORT_SIZE = 8
FLEXIBLE_SPREAD_LOGIT_RANGE = 2.5
HIGH_RISK_MAIN_PROB_START = 0.98
HIGH_RISK_MAIN_PROB_DECAY = 0.01
HIGH_RISK_MAIN_PROB_FLOOR = 0.94


BASE_TOKENS = {
    "code": (
        "def", "solve", "compute", "return", "if", "else", "for", "in", "range",
        "x", "y", "z", "a", "b", "tmp", "result", "answer", "value", "score", "limit", "count", "idx", "0", "#", "+", "-", "*", ">",
        "<", "==", "!=", "(", ")", ":", "=", "NEWLINE", "INDENT", "DEDENT",
        ",",
        "inline", "staged", "guarded", "direct", "branch", "accumulate", "clamp",
        "max", "min", "helper", "clear", "stepwise", "iterative", "early_return",
        "local_var", "cached", "vectorized", "loop", "conditional", "normalized",
        "bounded", "safe_default", "fast_path", "slow_path", "fallback", "compose",
        "extract", "rename", "reuse", "wrap", "unroll", "fold", "scan", "filter", "map",
    ),
    "math": (
        "let", "f", "g", "h", "x", "y", "n", "a", "b", "c", "compute", "done", "=", "+", "-",
        "/",
        "*", "(", ")", "compose", "expand", "factor", "substitute", "evaluate",
        "matrix", "vector", "sum", "dot", "final", "form", "direct", "intermediate",
        ".", "then", "so", "with", "and", "check", "write",
        "2", "3", "4", "5", "7", "9", "10", "12", "15", "17", "23", "27", "33",
        "simplify", "collect", "canonical", "distributed", "factored", "scalar",
        "product", "isolate", "reduce", "rewrite", "equivalent", "symbolic",
        "numeric", "closed_form", "recursive", "linear", "quadratic", "inverse",
        "transpose", "normalize", "cancel",
    ),
    "dialogue": (
        "Hi", "client", "assistant", "asks", "constraint", "done", "Mia", "Noah", "Ava", "Liam", "Friday", "Monday", "Tuesday", "3pm", "4pm", "morning", "meeting", "budget", "thanks",
        "friendly", "brief", "reminder", "confirm", "schedule", "call", "dinner",
        "project", "deadline", "please", "tomorrow", "do",
        "not", "mention", "invite", "reply", "warmly", "clearly", "short",
        "concise", "kindly", "avoid", "include", "team", "client",
        ",", ".", "the", "and", "about", "at", "I", "will", "keep", "it", "for", "with",
        "polite", "direct", "gentle", "formal", "casual", "empathetic", "neutral",
        "specific", "acknowledge", "decline", "summarize", "rephrase", "soften",
        "clarify", "professional", "supportive", "simple", "detailed", "calm", "helpful",
    ),
    "negative_control": tuple(f"ng{i}" for i in range(64)),
}


FLEXIBLE_POOLS = {
    "code": (
        "direct", "iterative", "local_var", "cached", "inline", "staged",
        "guarded", "helper", "clear", "stepwise", "accumulate", "clamp",
        "early_return",
        "cached", "vectorized", "loop", "conditional", "normalized", "bounded",
        "safe_default", "fast_path", "slow_path", "fallback", "compose", "extract",
        "rename", "reuse", "wrap", "unroll", "fold", "scan", "filter", "map",
    ),
    "math": (
        "expand", "factor", "direct", "intermediate", "simplify", "collect",
        "substitute", "compose", "evaluate", "canonical", "distributed", "factored",
        "matrix", "vector", "scalar", "dot", "sum", "product", "isolate", "reduce",
        "rewrite", "equivalent", "symbolic", "numeric", "closed_form", "recursive",
        "linear", "quadratic", "inverse", "transpose", "normalize", "cancel",
    ),
    "dialogue": (
        "friendly", "warmly", "kindly", "clearly", "brief", "short", "concise",
        "polite", "direct", "gentle", "formal", "casual", "empathetic", "neutral",
        "specific", "acknowledge", "confirm", "reminder", "reply", "invite",
        "decline", "include", "summarize", "rephrase", "soften", "clarify",
        "professional", "supportive", "simple", "detailed", "calm", "helpful",
    ),
}


def build_synthetic_domain(
    domain: str,
    *,
    sample_count: int = 8,
    script_len: int | None = None,
    vocab_size: int = 64,
) -> SyntheticDomainBundle:
    if domain not in DOMAIN_TAGS:
        raise ValueError(f"unknown synthetic domain: {domain}")
    expected_len = DOMAIN_SCRIPT_LENS[domain]
    if script_len is None:
        script_len = expected_len
    if script_len != expected_len:
        raise ValueError(f"script_len must be {expected_len} for {domain}")
    if vocab_size != 64:
        raise ValueError("vocab_size must be 64 for the synthetic protocol")

    vocab = _build_vocab(domain, vocab_size)
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    scripts = [
        _build_script(domain, token_to_id, script_len=script_len, sample_id=sample_id)
        for sample_id in range(sample_count)
    ]
    oracle = ScriptedOracle(
        scripts=scripts,
        eval_token_ids=tuple(idx for idx in range(len(vocab)) if idx not in {0, 1}),
        pad_id=0,
        bos_id=1,
        eos_id=2,
    )
    tags = DOMAIN_TAGS[domain]
    return SyntheticDomainBundle(
        name=domain,
        vocab=vocab,
        oracle=oracle,
        high_risk_tags=tuple(tags["high"]),
        flexible_tags=tuple(tags["flex"]),
    )


def _build_vocab(domain: str, vocab_size: int) -> tuple[str, ...]:
    tokens = list(SPECIAL)
    for token in BASE_TOKENS[domain]:
        if token not in tokens:
            tokens.append(token)
    filler_idx = 0
    filler_prefix = {
        "code": "alias",
        "math": "form",
        "dialogue": "phrase",
        "negative_control": "ng",
    }[domain]
    while len(tokens) < vocab_size:
        token = f"{filler_prefix}_{filler_idx:03d}"
        if token not in tokens:
            tokens.append(token)
        filler_idx += 1
    return tuple(tokens[:vocab_size])


def _build_script(
    domain: str,
    token_to_id: dict[str, int],
    *,
    script_len: int,
    sample_id: int,
) -> tuple[ScriptStep, ...]:
    if domain == "negative_control":
        return _negative_control_script(token_to_id, script_len, sample_id)

    normal_ids = [idx for token, idx in token_to_id.items() if token not in SPECIAL]
    violation_token = normal_ids[-1]
    decision_specs = _domain_decision_specs(domain, token_to_id, sample_id)
    surface_items = _domain_surface_items(domain)
    deterministic_tokens = _layout_token_ids(domain, token_to_id)
    steps: list[ScriptStep] = []
    high_seen = 0
    for pos in range(script_len - 1):
        item = surface_items[pos] if pos < len(surface_items) else None
        decision_idx = _decision_index(item)
        if decision_idx is None:
            token = _fixed_surface_token(item, domain, token_to_id, deterministic_tokens, pos, sample_id)
            structural = _is_structural_surface_item(domain, item)
            steps.append(
                ScriptStep(
                    dist=((token, 1.0),),
                    semantic_tag="syntax_layout" if structural else "context",
                    expected_role="structural" if structural else "context",
                    violation_dist=((token, 1.0),),
                    off_policy_violates=False,
                    clean_token_ids=(token,),
                )
            )
        else:
            tag, role, choices = decision_specs[decision_idx]
            if role == "high_risk":
                high_ids = tuple(choices[:4])
                main = high_ids[0]
                main_prob = _high_risk_main_prob(high_seen)
                alt_prob = (1.0 - main_prob) / (len(high_ids) - 1)
                high_seen += 1
                dist = ((main, main_prob),) + tuple((alt, alt_prob) for alt in high_ids[1:])
                steps.append(
                    ScriptStep(
                        dist=dist,
                        semantic_tag=tag,
                        expected_role="high_risk",
                        violation_dist=((violation_token, 1.0),),
                        off_policy_violates=True,
                        clean_token_ids=high_ids,
                        eval_token_ids=high_ids,
                        style_dists=_style_conditioned_high_risk_dists(domain, tag, high_seen, high_ids, dist),
                    )
                )
            elif role == "flexible":
                flex_ids = _flexible_token_ids(domain, token_to_id, choices, target_count=FLEXIBLE_SUPPORT_SIZE)
                style_dists = _style_spread_dists(flex_ids)
                steps.append(
                    ScriptStep(
                        dist=style_dists[0],
                        semantic_tag=tag,
                        expected_role="flexible",
                        violation_dist=((choices[0], 1.0),),
                        off_policy_violates=False,
                        clean_token_ids=tuple(flex_ids),
                        eval_token_ids=tuple(flex_ids),
                        style_dists=style_dists,
                        style_token_ids=(tuple(flex_ids[:4]), tuple(flex_ids[4:8])),
                        style_canonical_token_ids=(int(flex_ids[0]), int(flex_ids[4])),
                    )
                )
            else:
                raise ValueError(f"unknown decision role for {domain}:{tag}: {role}")
    steps.append(
        ScriptStep(
            dist=((token_to_id["<eos>"], 1.0),),
            semantic_tag="terminal",
            expected_role="terminal",
            violation_dist=((token_to_id["<eos>"], 1.0),),
            off_policy_violates=False,
            eval_token_ids=(token_to_id["<eos>"],),
        )
    )
    return tuple(steps)


def _high_risk_main_prob(high_seen: int) -> float:
    return max(HIGH_RISK_MAIN_PROB_FLOOR, HIGH_RISK_MAIN_PROB_START - HIGH_RISK_MAIN_PROB_DECAY * high_seen)


def _decision_index(item: object) -> int | None:
    if not isinstance(item, str):
        return None
    if len(item) >= 2 and item[0] == "D" and item[1:].isdigit():
        return int(item[1:])
    return None


def _fixed_surface_token(
    item: object,
    domain: str,
    token_to_id: dict[str, int],
    deterministic_tokens: tuple[int, ...],
    pos: int,
    sample_id: int,
) -> int:
    if isinstance(item, str) and item in token_to_id:
        return token_to_id[item]
    return deterministic_tokens[(pos + sample_id) % len(deterministic_tokens)]


STRUCTURAL_SURFACE_TOKENS = {
    "code": frozenset({
        "def",
        "(",
        ")",
        ":",
        "NEWLINE",
        "INDENT",
        "DEDENT",
        "if",
        "return",
        "=",
        ",",
        "#",
    }),
    "math": frozenset({
        "(",
        ")",
        "=",
        "+",
        "*",
        ".",
        "then",
        "so",
        "final",
    }),
    "dialogue": frozenset({
        "constraint",
        "do",
        "not",
        "mention",
    }),
}


def _is_structural_surface_item(domain: str, item: object) -> bool:
    return isinstance(item, str) and item in STRUCTURAL_SURFACE_TOKENS[domain]


def _domain_surface_items(domain: str) -> tuple[str, ...]:
    if domain == "code":
        return (
            "def", "compute", "(", "x", ",", "limit", ")", ":", "NEWLINE",
            "INDENT",
            "#", "D0", "D1", "D2", "direct", "clear", "helper", "inline", "NEWLINE",
            "result", "=", "0", "NEWLINE",
            "score", "=", "x", "D3", "limit", "NEWLINE",
            "if", "score", "D4", "limit", ":", "NEWLINE",
            "INDENT", "result", "=", "result", "D5", "score", "NEWLINE", "DEDENT",
            "return", "D6", "NEWLINE",
        )
    if domain == "math":
        return (
            "let", "f", "(", "x", ")", "=", "2", "*", "x", "+", "3", ".",
            "substitute", "x", "=", "D0", "then", "D1", ".",
            "compute", "2", "D2", "x", "+", "3", "=", "D3", ".",
            "so", "final", "=", "D4", ".",
            "check", "simplify", "and", "write", "D5", "form", ".", "done", ".",
        )
    if domain == "dialogue":
        return (
            "please", "keep", "it", "D0", "and", "brief", ".",
            "client", "D1", "asks", "about", "the", "D2", "reminder", "D3",
            "for", "the", "team", "for", "D4", ".",
            "constraint", "do", "not", "mention", "D5", ".",
            "assistant", "reply", "with", "the", "schedule", ".", "thanks",
        )
    raise ValueError(domain)


def _ids(token_to_id: dict[str, int], tokens: Sequence[str]) -> tuple[int, ...]:
    return tuple(token_to_id[token] for token in tokens if token in token_to_id)


def _flexible_token_ids(
    domain: str,
    token_to_id: dict[str, int],
    choices: Sequence[int],
    *,
    target_count: int,
) -> tuple[int, ...]:
    ids = list(dict.fromkeys(int(token_id) for token_id in choices))
    if len(ids) >= target_count:
        return tuple(ids[:target_count])
    for token in FLEXIBLE_POOLS[domain]:
        token_id = token_to_id.get(token)
        if token_id is not None and int(token_id) not in ids:
            ids.append(int(token_id))
        if len(ids) >= target_count:
            break
    filler_prefix = {
        "code": "alias_",
        "math": "form_",
        "dialogue": "phrase_",
    }[domain]
    for token, token_id in sorted(token_to_id.items(), key=lambda item: item[1]):
        if token.startswith(filler_prefix) and int(token_id) not in ids:
            ids.append(int(token_id))
        if len(ids) >= target_count:
            break
    return tuple(ids)


def _spread_probs(count: int) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    logits = [-(FLEXIBLE_SPREAD_LOGIT_RANGE * idx / max(1, count - 1)) for idx in range(count)]
    weights = [math.exp(value) for value in logits]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _style_spread_dists(token_ids: Sequence[int]) -> tuple[tuple[tuple[int, float], ...], ...]:
    if len(token_ids) != FLEXIBLE_SUPPORT_SIZE:
        raise ValueError(f"expected {FLEXIBLE_SUPPORT_SIZE} flexible token ids")
    probs = _spread_probs(len(token_ids))
    style0_order = tuple(int(token_id) for token_id in token_ids)
    style1_order = tuple(int(token_id) for token_id in token_ids[4:8]) + tuple(int(token_id) for token_id in token_ids[:4])
    return (
        tuple((token_id, prob) for token_id, prob in zip(style0_order, probs)),
        tuple((token_id, prob) for token_id, prob in zip(style1_order, probs)),
    )


def _style_conditioned_high_risk_dists(
    domain: str,
    tag: str,
    high_seen: int,
    token_ids: Sequence[int],
    default_dist: Sequence[tuple[int, float]],
) -> tuple[tuple[tuple[int, float], ...], ...]:
    if domain != "code":
        return ()
    if tag != "return_semantics" and not (tag == "operator" and high_seen >= 3):
        return ()
    if len(token_ids) < 2:
        return ()
    probs = [float(prob) for _, prob in default_dist]
    style0_order = tuple(int(token_id) for token_id in token_ids)
    style1_order = style0_order[1:] + style0_order[:1]
    return (
        tuple((token_id, prob) for token_id, prob in zip(style0_order, probs)),
        tuple((token_id, prob) for token_id, prob in zip(style1_order, probs)),
    )


def _layout_token_ids(domain: str, token_to_id: dict[str, int]) -> tuple[int, ...]:
    layouts = {
        "code": ("def", "solve", "(", "x", ",", "y", ")", ":", "NEWLINE", "INDENT", "if", "value", ">", "limit", ":", "NEWLINE", "return"),
        "math": ("let", "f", "(", "x", ")", "=", "2", "*", "x", "+", "3", ".", "evaluate", "compose", "form"),
        "dialogue": ("Hi", "please", "reply", "brief", "and", "friendly", "include", "the", "meeting", "and", "avoid", "budget"),
    }
    tokens = _ids(token_to_id, layouts[domain])
    if not tokens:
        return tuple(idx for token, idx in token_to_id.items() if token not in SPECIAL)
    return tokens


def _domain_decision_specs(
    domain: str,
    token_to_id: dict[str, int],
    sample_id: int,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if domain == "code":
        specs = (
            ("equivalent_implementation", "flexible", ("direct", "iterative", "local_var", "cached")),
            ("equivalent_implementation", "flexible", ("clear", "stepwise", "helper", "inline")),
            ("equivalent_implementation", "flexible", ("branch", "accumulate", "clamp", "max")),
            ("operator", "high_risk", ("+", "-", "*", "==")),
            ("branch_guard", "high_risk", (">", "<", "==", "!=")),
            ("operator", "high_risk", ("+", "-", "*", "==")),
            ("return_semantics", "high_risk", ("result", "score", "tmp", "limit")),
        )
    elif domain == "math":
        specs = (
            ("substitution", "high_risk", ("12", "10", "15", "7")),
            ("equivalent_representation", "flexible", ("expand", "factor", "direct", "intermediate")),
            ("operator", "high_risk", ("*", "+", "-", "/")),
            ("computed_value", "high_risk", ("27", "23", "33", "17")),
            ("final_answer", "high_risk", ("27", "23", "33", "17")),
            ("equivalent_representation", "flexible", ("canonical", "distributed", "factored", "closed_form")),
        )
    elif domain == "dialogue":
        specs = (
            ("tone_paraphrase", "flexible", ("polite", "friendly", "formal", "gentle")),
            ("recipient", "high_risk", ("Mia", "Noah", "Ava", "Liam")),
            ("required_fact", "high_risk", ("meeting", "deadline", "project", "call")),
            ("tone_paraphrase", "flexible", ("kindly", "clearly", "warmly", "friendly")),
            ("date_time", "high_risk", ("3pm", "4pm", "morning", "Friday")),
            ("forbidden_constraint", "high_risk", ("budget", "dinner", "schedule", "tomorrow")),
        )
    else:
        raise ValueError(domain)
    # Rotate paired choices across samples to avoid identical surface scripts.
    rotated = []
    for tag, role, tokens in specs:
        ids = list(_ids(token_to_id, tokens))
        if not ids:
            raise ValueError(f"empty decision choices for {domain}:{tag}")
        shift = 0 if role == "high_risk" else sample_id % len(ids)
        ids = ids[shift:] + ids[:shift]
        rotated.append((tag, role, tuple(ids)))
    return tuple(rotated)


def _decision_positions(script_len: int) -> tuple[int, ...]:
    # Helper retained for negative-control spacing.
    count = 10
    start = 2
    stop = script_len - 3
    if stop <= start:
        return tuple(range(max(0, script_len - 1)))
    raw = [round(start + i * (stop - start) / (count - 1)) for i in range(count)]
    positions = []
    for pos in raw:
        pos = max(0, min(script_len - 2, int(pos)))
        if pos not in positions:
            positions.append(pos)
    return tuple(positions)


def _negative_control_script(
    token_to_id: dict[str, int],
    script_len: int,
    sample_id: int,
) -> tuple[ScriptStep, ...]:
    normal_ids = [idx for token, idx in token_to_id.items() if token not in SPECIAL]
    decision_positions = set(_control_decision_positions(script_len))
    steps: list[ScriptStep] = []
    for pos in range(script_len - 1):
        if pos in decision_positions:
            start = (sample_id + pos) % max(1, len(normal_ids) - 4)
            choices = normal_ids[start : start + 4]
            steps.append(
                ScriptStep(
                    dist=tuple((tok, 1.0 / len(choices)) for tok in choices),
                    semantic_tag="surface_ngram",
                    expected_role="control",
                    off_policy_violates=False,
                    eval_token_ids=tuple(choices),
                )
            )
        else:
            token = normal_ids[(sample_id + pos) % len(normal_ids)]
            steps.append(
                ScriptStep(
                    dist=((token, 1.0),),
                    semantic_tag="syntax_layout",
                    expected_role="structural",
                    violation_dist=((token, 1.0),),
                    off_policy_violates=True,
                    clean_token_ids=(token,),
                )
            )
    steps.append(
        ScriptStep(
            dist=((token_to_id["<eos>"], 1.0),),
            semantic_tag="terminal",
            expected_role="terminal",
            off_policy_violates=False,
            eval_token_ids=(token_to_id["<eos>"],),
        )
    )
    return tuple(steps)


def _control_decision_positions(script_len: int) -> tuple[int, ...]:
    main_positions = _decision_positions(script_len)
    return main_positions[::2]
