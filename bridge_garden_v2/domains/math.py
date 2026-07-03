from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from bridge_garden_v2.schema import DomainDefinition, GenerationConfig, SequenceRecord, SpecialTokenIds


SPECIAL = ["<pad>", "<bos>", "<eos>"]
FUNCTION_NAMES = ["f", "g", "h", "q"]
PRIMARY_VARS = ["x", "n", "t", "y"]
BINDING_VARS = ["a", "b", "c", "u", "v"]

KEYWORDS = [
    "let",
    "then",
    "next",
    "note",
    "observe",
    "since",
    "therefore",
    "thus",
    "so",
    "hence",
    "compute",
    "check",
    "recheck",
    "bind",
    "using",
    "after",
    "at",
    "point",
    "value",
    "inner",
    "outer",
    "result",
    "equals",
    "same",
    "larger",
    "smaller",
    "than",
    "of",
    "means",
    "now",
    "again",
    "plainly",
    "carefully",
    "directly",
    "briefly",
    "noted",
]
OPERATORS = ["=", "+", "-", "*", ">", "<"]
PARENS = ["(", ")"]
PUNCT = [".", ","]

GARDEN_OPENERS = [
    ("note",),
    ("observe",),
    ("compute",),
    ("check",),
    ("recheck",),
]
GARDEN_TRANSITIONS = [
    ("then",),
    ("next",),
    ("now",),
]
GARDEN_CONCLUSIONS = [
    ("therefore",),
    ("thus",),
    ("so",),
    ("hence",),
    ("plainly",),
    ("briefly",),
]
GARDEN_RESTATE = [
    ("result", "equals"),
    ("that", "means"),
    ("same", "result"),
    ("again", "same"),
    ("carefully", "noted"),
]
GARDEN_FILLERS = [
    ("carefully",),
    ("again",),
    ("plainly",),
    ("directly",),
]

GARDEN_SYNONYM_GROUPS = [
    {token for variant in GARDEN_OPENERS for token in variant},
    {token for variant in GARDEN_TRANSITIONS for token in variant},
    {token for variant in GARDEN_CONCLUSIONS for token in variant},
    {token for variant in GARDEN_RESTATE for token in variant},
    {token for variant in GARDEN_FILLERS for token in variant},
    {"at", "point", "value"},
    {"after", "using", "of"},
    {"larger", "smaller", "same"},
]


def _number_tokens(lo: int = -40, hi: int = 100) -> List[str]:
    toks = []
    for value in range(lo, hi + 1):
        toks.append(str(value))
    return toks


def _build_vocab() -> Tuple[List[str], Dict[str, int]]:
    tokens = list(SPECIAL)
    for group in [
        FUNCTION_NAMES,
        PRIMARY_VARS,
        BINDING_VARS,
        KEYWORDS,
        ["that"],
        OPERATORS,
        PARENS,
        PUNCT,
        _number_tokens(),
    ]:
        for tok in group:
            if tok not in tokens:
                tokens.append(tok)
    return tokens, {tok: idx for idx, tok in enumerate(tokens)}


VOCAB, TOKEN_TO_ID = _build_vocab()
SPECIAL_IDS = SpecialTokenIds(
    pad=TOKEN_TO_ID["<pad>"],
    bos=TOKEN_TO_ID["<bos>"],
    eos=TOKEN_TO_ID["<eos>"],
)
NUMBER_TOKENS = set(_number_tokens())
FUNCTION_TOKEN_SET = set(FUNCTION_NAMES)
PRIMARY_VAR_SET = set(PRIMARY_VARS)
BINDING_VAR_SET = set(BINDING_VARS)
KEYWORD_SET = set(KEYWORDS)
OPERATOR_SET = set(OPERATORS)
PAREN_SET = set(PARENS)
PUNCT_SET = set(PUNCT)


def _linear(a: int, b: int, value: int) -> int:
    return a * value + b


def _quadratic(a: int, b: int, c: int, value: int) -> int:
    return a * value * value + b * value + c


@dataclass
class _LatentState:
    functions: Dict[str, Dict[str, int]]
    bindings: Dict[str, int]
    queries: Dict[str, int]
    derived: Dict[str, int]

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            "functions": copy.deepcopy(self.functions),
            "bindings": copy.deepcopy(self.bindings),
            "queries": copy.deepcopy(self.queries),
            "derived": copy.deepcopy(self.derived),
        }


class _SequenceBuilder:
    def __init__(self) -> None:
        self.tokens: List[str] = ["<bos>"]
        self.roles: List[str] = ["middle"]
        self.trace: List[Dict[str, object]] = []
        self.metadata: Dict[str, object] = {}

    def emit(self, token: str, role: str) -> None:
        self.tokens.append(token)
        self.roles.append(role)

    def extend(self, parts: Iterable[Tuple[str, str]]) -> None:
        for token, role in parts:
            self.emit(token, role)

    def log(self, kind: str, summary: str, reads: Sequence[str], writes: Sequence[str], state: _LatentState) -> None:
        self.trace.append(
            {
                "kind": kind,
                "summary": summary,
                "reads": list(reads),
                "writes": list(writes),
                "state_snapshot": state.snapshot(),
                "token_count": len(self.tokens) - 1,
            }
        )

    def finish(self) -> SequenceRecord:
        self.tokens.append("<eos>")
        self.roles.append("middle")
        token_ids = [TOKEN_TO_ID[token] for token in self.tokens]
        return SequenceRecord(
            token_ids=token_ids,
            tokens=self.tokens,
            oracle_roles=self.roles,
            latent_trace=self.trace,
            metadata=self.metadata,
        )


def _garden_phrase(rng: random.Random, variants: Sequence[Tuple[str, ...]]) -> List[Tuple[str, str]]:
    return [(token, "garden") for token in rng.choice(list(variants))]


def _garden_filler(rng: random.Random) -> List[Tuple[str, str]]:
    return [(token, "garden") for token in rng.choice(GARDEN_FILLERS)]


def _emit_function_definition(
    builder: _SequenceBuilder,
    state: _LatentState,
    fn: str,
    var: str,
    coeffs: Sequence[int],
) -> None:
    if len(coeffs) == 2:
        a, b = coeffs
        builder.extend(
            [
                ("let", "middle"),
                (fn, "commit"),
                ("(", "middle"),
                (var, "commit"),
                (")", "middle"),
                ("=", "middle"),
                (str(a), "commit"),
                ("*", "middle"),
                (var, "bridge"),
                ("+", "middle"),
                (str(b), "commit"),
                (".", "middle"),
            ]
        )
        summary = f"Commit linear rule {fn}({var}) = {a}*{var}+{b}"
        state.functions[fn] = {"type": 0, "a": a, "b": b}
    else:
        a, b, c = coeffs
        builder.extend(
            [
                ("let", "middle"),
                (fn, "commit"),
                ("(", "middle"),
                (var, "commit"),
                (")", "middle"),
                ("=", "middle"),
                (str(a), "commit"),
                ("*", "middle"),
                (var, "bridge"),
                ("*", "middle"),
                (var, "bridge"),
                ("+", "middle"),
                (str(b), "commit"),
                ("*", "middle"),
                (var, "bridge"),
                ("+", "middle"),
                (str(c), "commit"),
                (".", "middle"),
            ]
        )
        summary = f"Commit quadratic rule {fn}({var}) = {a}*{var}*{var}+{b}*{var}+{c}"
        state.functions[fn] = {"type": 1, "a": a, "b": b, "c": c}
    builder.log("commit_function", summary, [], [fn], state)


def _emit_eval_clause(
    builder: _SequenceBuilder,
    state: _LatentState,
    rng: random.Random,
    opener: Sequence[Tuple[str, str]],
    fn: str,
    value_label: str,
    value: int,
    result_label: str,
    result: int,
    *,
    restate: bool = False,
) -> None:
    builder.extend(opener)
    builder.extend(_garden_filler(rng))
    builder.extend(
        [
            (fn, "bridge"),
            ("at", "garden"),
            (str(value), "commit"),
            ("=", "middle"),
            (str(result), "bridge"),
            (".", "middle"),
        ]
    )
    state.queries[value_label] = value
    state.derived[result_label] = result
    builder.log(
        "evaluate",
        f"Read {fn} at {value} and derive {result}",
        [fn],
        [value_label, result_label],
        state,
    )
    if restate:
        builder.extend(_garden_phrase(rng, GARDEN_RESTATE))
        builder.extend(_garden_filler(rng))
        builder.extend(
            [
                (fn, "bridge"),
                ("at", "garden"),
                (str(value), "bridge"),
                ("=", "middle"),
                (str(result), "bridge"),
                (".", "middle"),
            ]
        )
        builder.log(
            "restate",
            f"Restate exact readout {fn}({value}) = {result}",
            [fn, value_label, result_label],
            [],
            state,
        )


def _emit_compare_clause(
    builder: _SequenceBuilder,
    state: _LatentState,
    rng: random.Random,
    left_label: str,
    right_label: str,
    *,
    compare_word: str,
) -> None:
    left = state.derived[left_label]
    right = state.derived[right_label]
    builder.extend(_garden_phrase(rng, GARDEN_CONCLUSIONS))
    builder.extend(_garden_filler(rng))
    builder.extend(
        [
            (str(left), "bridge"),
            (compare_word, "garden"),
            ("than", "garden"),
            (str(right), "bridge"),
            (".", "middle"),
        ]
    )
    builder.log(
        "compare",
        f"Compare derived values {left_label}={left} and {right_label}={right}",
        [left_label, right_label],
        [],
        state,
    )


def _pattern_linear_compare(rng: random.Random) -> SequenceRecord:
    builder = _SequenceBuilder()
    fn = rng.choice(FUNCTION_NAMES[:3])
    var = rng.choice(PRIMARY_VARS[:3])
    a = rng.randint(2, 4)
    b = rng.randint(1, 8)
    x1 = rng.randint(1, 4)
    x2 = rng.randint(5, 8)
    y1 = _linear(a, b, x1)
    y2 = _linear(a, b, x2)
    state = _LatentState(functions={}, bindings={}, queries={}, derived={})

    _emit_function_definition(builder, state, fn, var, [a, b])
    _emit_eval_clause(
        builder,
        state,
        rng,
        _garden_phrase(rng, GARDEN_OPENERS),
        fn,
        "query_left",
        x1,
        "result_left",
        y1,
        restate=rng.random() < 0.4,
    )
    _emit_eval_clause(
        builder,
        state,
        rng,
        _garden_phrase(rng, GARDEN_TRANSITIONS),
        fn,
        "query_right",
        x2,
        "result_right",
        y2,
        restate=rng.random() < 0.5,
    )
    _emit_compare_clause(builder, state, rng, "result_right", "result_left", compare_word="larger")
    builder.metadata["pattern"] = "linear_compare"
    builder.metadata["expected_bridge_reads"] = [fn, str(y1), str(y2)]
    return builder.finish()


def _pattern_composition(rng: random.Random) -> SequenceRecord:
    builder = _SequenceBuilder()
    outer = rng.choice(FUNCTION_NAMES[:3])
    inner = rng.choice([name for name in FUNCTION_NAMES if name != outer])
    var = rng.choice(PRIMARY_VARS)
    a1, b1 = rng.randint(1, 3), rng.randint(0, 6)
    a2, b2 = rng.randint(1, 3), rng.randint(1, 5)
    x0 = rng.randint(1, 4)
    inner_value = _linear(a1, b1, x0)
    outer_value = _linear(a2, b2, inner_value)
    state = _LatentState(functions={}, bindings={}, queries={}, derived={})

    _emit_function_definition(builder, state, inner, var, [a1, b1])
    _emit_function_definition(builder, state, outer, var, [a2, b2])
    _emit_eval_clause(
        builder,
        state,
        rng,
        _garden_phrase(rng, GARDEN_OPENERS),
        inner,
        "inner_query",
        x0,
        "inner_result",
        inner_value,
        restate=False,
    )
    builder.extend(_garden_phrase(rng, GARDEN_CONCLUSIONS))
    builder.extend(
        [
            (outer, "bridge"),
            ("after", "garden"),
            (inner, "bridge"),
            ("at", "garden"),
            (str(x0), "bridge"),
            ("=", "middle"),
            (outer, "bridge"),
            ("(", "middle"),
            (str(inner_value), "bridge"),
            (")", "middle"),
            ("=", "middle"),
            (str(outer_value), "bridge"),
            (".", "middle"),
        ]
    )
    state.derived["outer_result"] = outer_value
    builder.log(
        "compose",
        f"Compose {outer} after {inner} using inner={inner_value} and outer={outer_value}",
        [outer, inner, "inner_query", "inner_result"],
        ["outer_result"],
        state,
    )
    builder.metadata["pattern"] = "composition"
    builder.metadata["expected_bridge_reads"] = [inner, outer, str(inner_value), str(outer_value)]
    return builder.finish()


def _pattern_binding_chain(rng: random.Random) -> SequenceRecord:
    builder = _SequenceBuilder()
    fn = rng.choice(FUNCTION_NAMES[:3])
    var = rng.choice(PRIMARY_VARS)
    bind_name = rng.choice(BINDING_VARS)
    a, b = rng.randint(2, 4), rng.randint(1, 7)
    query = rng.randint(2, 6)
    bound_value = _linear(a, b, query)
    threshold = rng.randint(8, 16)
    final_value = bound_value - rng.randint(1, 4)
    state = _LatentState(functions={}, bindings={}, queries={}, derived={})

    _emit_function_definition(builder, state, fn, var, [a, b])
    builder.extend(_garden_phrase(rng, GARDEN_OPENERS))
    builder.extend(
        [
            ("bind", "garden"),
            (bind_name, "commit"),
            ("=", "middle"),
            (fn, "bridge"),
            ("at", "garden"),
            (str(query), "commit"),
            ("=", "middle"),
            (str(bound_value), "bridge"),
            (".", "middle"),
        ]
    )
    state.queries["binding_query"] = query
    state.bindings[bind_name] = bound_value
    builder.log(
        "bind_value",
        f"Bind {bind_name} to {fn}({query}) = {bound_value}",
        [fn],
        [bind_name, "binding_query"],
        state,
    )

    builder.extend(_garden_phrase(rng, [("since",)]))
    builder.extend(
        [
            (bind_name, "bridge"),
            (">", "middle"),
            (str(threshold), "commit"),
            (",", "middle"),
        ]
    )
    builder.log(
        "threshold_check",
        f"Check binding {bind_name}={bound_value} against threshold {threshold}",
        [bind_name],
        ["threshold"],
        state,
    )

    builder.extend(_garden_phrase(rng, GARDEN_CONCLUSIONS))
    builder.extend(
        [
            (bind_name, "bridge"),
            ("-", "middle"),
            (str(bound_value - final_value), "commit"),
            ("=", "middle"),
            (str(final_value), "bridge"),
            (".", "middle"),
        ]
    )
    state.derived["final_result"] = final_value
    builder.log(
        "derive_from_binding",
        f"Use binding {bind_name} to derive final result {final_value}",
        [bind_name],
        ["final_result"],
        state,
    )
    builder.metadata["pattern"] = "binding_chain"
    builder.metadata["expected_bridge_reads"] = [bind_name, str(bound_value), str(final_value)]
    return builder.finish()


def _pattern_quadratic_check(rng: random.Random) -> SequenceRecord:
    builder = _SequenceBuilder()
    fn = rng.choice(["q", "h"])
    var = rng.choice(PRIMARY_VARS)
    a = 1
    b = rng.randint(1, 4)
    c = rng.randint(0, 5)
    query = rng.randint(1, 4)
    result = _quadratic(a, b, c, query)
    state = _LatentState(functions={}, bindings={}, queries={}, derived={})

    _emit_function_definition(builder, state, fn, var, [a, b, c])
    _emit_eval_clause(
        builder,
        state,
        rng,
        _garden_phrase(rng, GARDEN_OPENERS),
        fn,
        "quad_query",
        query,
        "quad_result",
        result,
        restate=True,
    )
    builder.extend(_garden_phrase(rng, GARDEN_CONCLUSIONS))
    builder.extend(
        [
            (str(result), "bridge"),
            (">", "middle"),
            ("0", "commit"),
            (".", "middle"),
        ]
    )
    builder.log(
        "sign_check",
        f"Check positive sign of exact quadratic result {result}",
        ["quad_result"],
        ["sign_target"],
        state,
    )
    builder.metadata["pattern"] = "quadratic_check"
    builder.metadata["expected_bridge_reads"] = [fn, str(result)]
    return builder.finish()


PATTERN_BUILDERS = [
    _pattern_linear_compare,
    _pattern_composition,
    _pattern_binding_chain,
    _pattern_quadratic_check,
]


def _generate_one(rng: random.Random, min_len: int, max_len: int) -> SequenceRecord:
    for _ in range(64):
        record = rng.choice(PATTERN_BUILDERS)(rng)
        length = len(record.token_ids)
        if min_len <= length <= max_len:
            return record
    return rng.choice(PATTERN_BUILDERS)(rng)


def generate_split(split_name: str, config: GenerationConfig) -> List[SequenceRecord]:
    split_sizes = {
        "train": config.train_size,
        "val": config.val_size,
        "test": config.test_size,
        "analysis": config.analysis_size,
    }
    if split_name not in split_sizes:
        raise ValueError(f"Unknown split: {split_name}")
    split_offsets = {"train": 11, "val": 23, "test": 37, "analysis": 53}
    rng = random.Random(config.seed + split_offsets[split_name])
    records: List[SequenceRecord] = []
    for idx in range(split_sizes[split_name]):
        record = _generate_one(rng, config.min_len, config.max_len)
        record.metadata["split"] = split_name
        record.metadata["sample_index"] = idx
        records.append(record)
    return records


def _payload_tokens_and_breaks(record: SequenceRecord) -> Tuple[List[str], List[str], List[int]]:
    tokens: List[str] = []
    roles: List[str] = []
    line_breaks: List[int] = []
    for token, role in zip(record.tokens, record.oracle_roles):
        if token in SPECIAL:
            continue
        tokens.append(token)
        roles.append(role)
        if token == ".":
            line_breaks.append(len(tokens) - 1)
    return tokens, roles, line_breaks


def _trace_rows(record: SequenceRecord) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for idx, event in enumerate(record.latent_trace, start=1):
        snapshot = event.get("state_snapshot", {})
        functions = ",".join(sorted(snapshot.get("functions", {}).keys())) or "-"
        bindings = ",".join(f"{k}={v}" for k, v in sorted(snapshot.get("bindings", {}).items())) or "-"
        derived = ",".join(f"{k}={v}" for k, v in sorted(snapshot.get("derived", {}).items())) or "-"
        reads = ",".join(str(x) for x in event.get("reads", [])) or "-"
        writes = ",".join(str(x) for x in event.get("writes", [])) or "-"
        text = (
            f"{event.get('summary', '')}; reads={reads}; writes={writes}; "
            f"state[fns={functions} | bind={bindings} | derived={derived}]"
        )
        rows.append({"label": f"S{idx} {event.get('kind', 'step')}", "text": text})
    return rows


def build_example_payload(record: SequenceRecord) -> Dict[str, object]:
    tokens, roles, line_breaks = _payload_tokens_and_breaks(record)
    pattern = str(record.metadata.get("pattern", "math_derivation"))
    title = f"Math Derivation: {pattern.replace('_', ' ').title()}"
    return {
        "tokens": tokens,
        "roles": roles,
        "trace_rows": _trace_rows(record),
        "line_breaks": line_breaks,
        "title": title,
        "caption": (
            "Commit tokens introduce rules or query values; Bridge tokens read exact "
            "derivation state; Garden tokens vary phrasing or local connective words "
            "without changing the derivation semantics."
        ),
    }


def _category_alternatives(token: str) -> List[str]:
    for group in GARDEN_SYNONYM_GROUPS:
        if token in group:
            return sorted(group)
    if token in FUNCTION_TOKEN_SET:
        return FUNCTION_NAMES
    if token in PRIMARY_VAR_SET:
        return PRIMARY_VARS
    if token in BINDING_VAR_SET:
        return BINDING_VARS
    if token in NUMBER_TOKENS:
        return [str(v) for v in range(-12, 25)]
    if token in OPERATOR_SET:
        return OPERATORS
    if token in KEYWORD_SET:
        return list(KEYWORD_SET)
    if token in PAREN_SET:
        return list(PAREN_SET)
    if token in PUNCT_SET:
        return list(PUNCT_SET)
    return [token]


def sample_alternatives(record: SequenceRecord, step_idx: int, n_alts: int) -> List[int]:
    """
    Return same-category / same-slot alternatives for counterfactual kappa.

    `step_idx` follows the pipeline convention: it indexes prediction steps, so the
    actual token is `record.tokens[step_idx + 1]`.
    """
    token_index = step_idx + 1
    if token_index <= 0 or token_index >= len(record.tokens) - 1:
        return []
    token = record.tokens[token_index]
    role = record.oracle_roles[token_index]
    if token in SPECIAL or role == "middle" and token not in OPERATOR_SET | PAREN_SET | PUNCT_SET:
        return []
    candidates = [alt for alt in _category_alternatives(token) if alt != token and alt in TOKEN_TO_ID]
    return [TOKEN_TO_ID[alt] for alt in candidates[:n_alts]]


def teacher_action_ids(record: SequenceRecord, step_idx: int) -> List[int]:
    token_index = step_idx + 1
    if token_index <= 0 or token_index >= len(record.tokens) - 1:
        return []
    token = record.tokens[token_index]
    candidates = [alt for alt in _category_alternatives(token) if alt in TOKEN_TO_ID]
    return [TOKEN_TO_ID[alt] for alt in candidates]


def build_domain() -> DomainDefinition:
    return DomainDefinition(
        name="math",
        vocab=VOCAB,
        token_to_id=TOKEN_TO_ID,
        special_ids=SPECIAL_IDS,
        default_config={
            "patterns": [builder.__name__.removeprefix("_pattern_") for builder in PATTERN_BUILDERS],
            "query_range": [1, 8],
            "coefficient_range": [1, 4],
            "goal": "Readable derivation sequences with explicit commit/bridge/garden traces",
        },
        generate_split=generate_split,
        sample_alternatives=sample_alternatives,
        teacher_action_ids=teacher_action_ids,
        build_example_payload=build_example_payload,
    )


def _smoke_check() -> None:
    cfg = GenerationConfig(
        train_size=4,
        val_size=2,
        test_size=2,
        analysis_size=2,
        min_len=24,
        max_len=96,
        seed=42,
    )
    domain = build_domain()
    train_records = domain.generate_split("train", cfg)
    assert len(train_records) == 4
    for record in train_records:
        assert len(record.token_ids) == len(record.tokens) == len(record.oracle_roles)
        assert record.tokens[0] == "<bos>" and record.tokens[-1] == "<eos>"
        assert any(role == "commit" for role in record.oracle_roles)
        assert any(role == "bridge" for role in record.oracle_roles)
        assert any(role == "garden" for role in record.oracle_roles)
        assert record.latent_trace, "latent_trace should not be empty"
        assert record.metadata.get("pattern") in {
            "linear_compare",
            "composition",
            "binding_chain",
            "quadratic_check",
        }
        payload = build_example_payload(record)
        assert isinstance(payload["tokens"], list) and payload["tokens"]
        assert isinstance(payload["roles"], list) and len(payload["tokens"]) == len(payload["roles"])
        assert isinstance(payload["trace_rows"], list) and payload["trace_rows"]
        assert "caption" in payload and "title" in payload
    print(f"math smoke check OK: generated {len(train_records)} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-check", action="store_true")
    args = parser.parse_args()
    if args.smoke_check:
        _smoke_check()
