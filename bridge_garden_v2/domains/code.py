from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from bridge_garden_v2.schema import DomainDefinition, GenerationConfig, SequenceRecord, SpecialTokenIds


SPECIAL = ["<pad>", "<bos>", "<eos>"]
KEYWORDS = [
    "def",
    "if",
    "else",
    "return",
    "for",
    "in",
    "range",
    "True",
    "False",
]
STRUCTURE = [
    "(",
    ")",
    ":",
    ",",
    "=",
    "+",
    "-",
    ">",
    "<",
    "==",
    "!=",
    "NEWLINE",
    "INDENT",
    "DEDENT",
    "#",
]
FUNC_NAMES = ["solve", "compute", "check", "update", "collect", "merge", "trace", "review"]
PARAM_NAMES = ["item", "seed", "count", "value", "budget", "cursor", "index", "limit", "signal", "target"]
LOCAL_NAMES = [
    "base", "delta", "total", "cache", "marker", "result", "window", "offset", "alias",
    "buffer", "shadow", "anchor", "choice", "branch", "state", "current", "final",
]
NUMBERS = [str(i) for i in range(11)]
COMMENT_WORDS = [
    "note",
    "keep",
    "safe",
    "trace",
    "fast",
    "stable",
    "clean",
    "local",
    "guard",
    "path",
    "reuse",
    "state",
    "cache",
    "final",
    "branch",
    "trace",
    "steady",
    "simple",
    "guarded",
    "careful",
    "nearby",
    "detail",
    "context",
    "helper",
    "review",
    "variant",
    "signal",
    "smooth",
    "guided",
    "reason",
]


def _build_vocab() -> Tuple[List[str], Dict[str, int]]:
    vocab = list(SPECIAL)
    for group in [KEYWORDS, STRUCTURE, FUNC_NAMES, PARAM_NAMES, LOCAL_NAMES, NUMBERS, COMMENT_WORDS]:
        for token in group:
            if token not in vocab:
                vocab.append(token)
    return vocab, {token: idx for idx, token in enumerate(vocab)}


VOCAB, TOKEN_TO_ID = _build_vocab()
SPECIAL_IDS = SpecialTokenIds(
    pad=TOKEN_TO_ID["<pad>"],
    bos=TOKEN_TO_ID["<bos>"],
    eos=TOKEN_TO_ID["<eos>"],
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "comment_probability": 0.95,
    "comment_lengths": [3, 4],
    "patterns": ["branch_return", "loop_accumulator", "fallback_cache", "alias_guard_return"],
    "max_retries": 128,
}

TOKEN_GROUPS: Dict[str, List[str]] = {
    "comment": list(COMMENT_WORDS),
    "function_name": list(FUNC_NAMES),
    "identifier": list(PARAM_NAMES + LOCAL_NAMES),
    "number": list(NUMBERS),
    "arith_op": ["+", "-"],
    "cmp_op": [">", "<", "==", "!="],
    "bool": ["True", "False"],
}


@dataclass
class SymbolInfo:
    name: str
    origin: str
    value_hint: str


class ProgramBuilder:
    def __init__(self) -> None:
        self.tokens: List[str] = ["<bos>"]
        self.roles: List[str] = ["middle"]
        self.latent_trace: List[Dict[str, Any]] = []
        self.indent = 0
        self.line_no = 0
        self._known_symbols: Dict[str, SymbolInfo] = {}

    def add(self, token: str, role: str = "middle", trace: Dict[str, Any] | None = None) -> None:
        self.tokens.append(token)
        self.roles.append(role)
        if trace is not None:
            item = dict(trace)
            item.setdefault("token", token)
            item.setdefault("token_index", len(self.tokens) - 1)
            item.setdefault("line_no", self.line_no)
            item.setdefault("indent", self.indent)
            self.latent_trace.append(item)

    def emit_line_break(self) -> None:
        self.add("NEWLINE", "middle", {"kind": "line_break"})
        self.line_no += 1

    def emit_indent(self) -> None:
        self.add("INDENT", "middle", {"kind": "indent"})
        self.indent += 1

    def emit_dedent(self) -> None:
        self.indent = max(0, self.indent - 1)
        self.add("DEDENT", "middle", {"kind": "dedent"})

    def define_symbol(self, name: str, origin: str, value_hint: str) -> None:
        self._known_symbols[name] = SymbolInfo(name=name, origin=origin, value_hint=value_hint)

    def snapshot_symbols(self) -> Dict[str, Dict[str, str]]:
        return {
            name: {"origin": info.origin, "value_hint": info.value_hint}
            for name, info in sorted(self._known_symbols.items())
        }


def _pick_comment(rng: random.Random, hint: str) -> List[str]:
    hint_map = {
        "guard": ["guard", "safe", "path"],
        "branch": ["branch", "state", "trace"],
        "return": ["final", "stable", "return"],
        "loop": ["reuse", "local", "state"],
        "cache": ["cache", "keep", "clean"],
    }
    choices = hint_map.get(hint, ["note", "trace", "state"])
    n_words = rng.choice(DEFAULT_CONFIG["comment_lengths"])
    words = list(choices[: min(len(choices), n_words)])
    while len(words) < n_words:
        words.append(rng.choice(COMMENT_WORDS))
    return words


def _emit_comment(builder: ProgramBuilder, words: Sequence[str], reason: str) -> None:
    builder.add("#", "middle", {"kind": "comment_start", "reason": reason})
    for word in words:
        builder.add(word, "garden", {"kind": "comment_word", "reason": reason})
    builder.emit_line_break()


def _emit_signature(builder: ProgramBuilder, func_name: str, params: Sequence[str]) -> None:
    builder.add("def")
    builder.add(func_name, "middle", {"kind": "function_name"})
    builder.add("(")
    for idx, param in enumerate(params):
        builder.add(
            param,
            "commit",
            {
                "kind": "param_commit",
                "writes": [param],
                "state_after": builder.snapshot_symbols(),
            },
        )
        builder.define_symbol(param, origin="param", value_hint=f"input:{param}")
        if idx != len(params) - 1:
            builder.add(",")
    builder.add(")")
    builder.add(":")
    builder.emit_line_break()
    builder.emit_indent()


def _emit_assignment(
    builder: ProgramBuilder,
    target: str,
    read_a: str,
    op: str,
    read_b: str,
    numeric_literal: str,
    origin: str,
) -> None:
    builder.add(
        target,
        "commit",
        {
            "kind": "assignment_target",
            "writes": [target],
            "reads": [read_a, read_b] if read_b else [read_a],
            "origin": origin,
        },
    )
    builder.add("=")
    builder.add(
        read_a,
        "bridge",
        {
            "kind": "assignment_read",
            "reads": [read_a],
            "depends_on": read_a,
            "state_before": builder.snapshot_symbols(),
        },
    )
    builder.add(op)
    if read_b:
        builder.add(
            read_b,
            "bridge",
            {
                "kind": "assignment_read",
                "reads": [read_b],
                "depends_on": read_b,
            },
        )
    else:
        builder.add(
            numeric_literal,
            "commit",
            {
                "kind": "literal_commit",
                "writes": [target],
                "value_hint": numeric_literal,
            },
        )
    builder.emit_line_break()
    value_hint = f"{read_a}{op}{read_b or numeric_literal}"
    builder.define_symbol(target, origin=origin, value_hint=value_hint)


def _emit_if_header(builder: ProgramBuilder, lhs: str, cmp_op: str, rhs: str) -> None:
    builder.add("if")
    builder.add(
        lhs,
        "bridge",
        {"kind": "branch_read", "reads": [lhs], "state_before": builder.snapshot_symbols()},
    )
    builder.add(cmp_op)
    builder.add(
        rhs,
        "bridge",
        {"kind": "branch_read", "reads": [rhs]},
    )
    builder.add(":")
    builder.emit_line_break()
    builder.emit_indent()


def _emit_else_header(builder: ProgramBuilder) -> None:
    builder.emit_dedent()
    builder.add("else")
    builder.add(":")
    builder.emit_line_break()
    builder.emit_indent()


def _emit_return(builder: ProgramBuilder, value: str) -> None:
    builder.add("return")
    builder.add(
        value,
        "bridge",
        {
            "kind": "return_read",
            "reads": [value],
            "depends_on": value,
            "state_before": builder.snapshot_symbols(),
        },
    )
    builder.emit_line_break()


def _emit_for_loop(builder: ProgramBuilder, loop_var: str, upper_ref: str, acc_var: str, delta_var: str) -> None:
    builder.add("for")
    builder.add(
        loop_var,
        "commit",
        {"kind": "loop_commit", "writes": [loop_var], "origin": "loop_index"},
    )
    builder.define_symbol(loop_var, origin="loop_index", value_hint="range_index")
    builder.add("in")
    builder.add("range")
    builder.add("(")
    builder.add(
        upper_ref,
        "bridge",
        {"kind": "loop_bound_read", "reads": [upper_ref], "state_before": builder.snapshot_symbols()},
    )
    builder.add(")")
    builder.add(":")
    builder.emit_line_break()
    builder.emit_indent()

    builder.add(
        acc_var,
        "bridge",
        {"kind": "loop_acc_read", "reads": [acc_var], "depends_on": acc_var},
    )
    builder.add("=")
    builder.add(
        acc_var,
        "bridge",
        {"kind": "loop_acc_self_read", "reads": [acc_var], "depends_on": acc_var},
    )
    builder.add("+")
    builder.add(
        delta_var,
        "bridge",
        {"kind": "loop_delta_read", "reads": [delta_var], "depends_on": delta_var},
    )
    builder.emit_line_break()
    builder.define_symbol(acc_var, origin="loop_update", value_hint=f"{acc_var}+{delta_var}")
    builder.emit_dedent()


def _render_lines(tokens: Sequence[str]) -> List[str]:
    lines: List[str] = []
    current: List[str] = []
    indent = 0
    for token in tokens:
        if token in {"<bos>", "<eos>"}:
            continue
        if token == "INDENT":
            indent += 1
            continue
        if token == "DEDENT":
            indent = max(0, indent - 1)
            continue
        if token == "NEWLINE":
            if current:
                lines.append("    " * indent + " ".join(current))
            current = []
            continue
        current.append(token)
    if current:
        lines.append("    " * indent + " ".join(current))
    return lines


def _build_example_payload(record: SequenceRecord) -> Dict[str, Any]:
    visible_tokens: List[str] = []
    visible_roles: List[str] = []
    line_breaks: List[int] = []
    current_indent = 0
    emitted_on_line = False

    for token, role in zip(record.tokens, record.oracle_roles):
        if token == "<bos>":
            continue
        if token == "<eos>":
            visible_tokens.append(token)
            visible_roles.append("special")
            emitted_on_line = True
            continue
        if token == "INDENT":
            current_indent += 1
            continue
        if token == "DEDENT":
            current_indent = max(0, current_indent - 1)
            continue
        if token == "NEWLINE":
            if emitted_on_line and visible_tokens:
                line_breaks.append(len(visible_tokens) - 1)
            emitted_on_line = False
            continue
        if not emitted_on_line and current_indent > 0:
            indent_token = "»" * current_indent
            visible_tokens.append(indent_token)
            visible_roles.append("middle")
        visible_tokens.append(token)
        visible_roles.append(role)
        emitted_on_line = True

    rendered_lines = _render_lines(record.tokens)
    trace_rows: List[Dict[str, str]] = []
    pattern = record.metadata.get("pattern", "code")
    params = ", ".join(record.metadata.get("params", []))
    trace_rows.append({"label": "Pattern", "text": f"{pattern}; params={params}"})

    state_targets = ", ".join(record.metadata.get("state_targets", []))
    if state_targets:
        trace_rows.append({"label": "Writes", "text": f"commit vars -> {state_targets}"})

    symbol_state: Dict[str, str] = {}
    branch_desc: List[str] = []
    return_desc: List[str] = []
    for item in record.latent_trace:
        writes = item.get("writes", [])
        if writes:
            origin = item.get("origin") or item.get("kind", "write")
            for name in writes:
                symbol_state[name] = origin
        if item.get("kind") == "branch_read":
            reads = ",".join(item.get("reads", []))
            branch_desc.append(f"{reads} {item.get('token', '')}".strip())
        if item.get("kind") == "return_read":
            reads = item.get("reads", [])
            if reads:
                return_desc.append(reads[0])

    if symbol_state:
        ordered = ", ".join(f"{name}:{origin}" for name, origin in symbol_state.items())
        trace_rows.append({"label": "State", "text": ordered})
    if branch_desc:
        trace_rows.append({"label": "Branch", "text": " / ".join(branch_desc[:2])})
    if return_desc:
        trace_rows.append({"label": "Return", "text": " -> ".join(return_desc)})

    role_counts = {
        key: record.oracle_roles.count(key) for key in ["commit", "bridge", "garden", "middle"]
    }
    trace_rows.append(
        {
            "label": "Role mix",
            "text": ", ".join(f"{name}={count}" for name, count in role_counts.items()),
        }
    )
    trace_rows.append(
        {
            "label": "Garden note",
            "text": "comment words are safe local variants; symbol reads and returns carry bridge risk",
        }
    )

    return {
        "tokens": visible_tokens,
        "roles": visible_roles,
        "trace_rows": trace_rows,
        "line_breaks": line_breaks,
        "caption": "Commit tokens write new symbol state, bridge tokens read or return that state, and garden tokens live in comments that do not change execution.",
        "title": f"Code Example: {pattern}",
        "rendered_lines": rendered_lines,
        "latent_trace": record.latent_trace,
        "metadata": record.metadata,
    }


def _infer_token_group(token: str) -> str | None:
    if token in COMMENT_WORDS:
        return "comment"
    if token in FUNC_NAMES:
        return "function_name"
    if token in PARAM_NAMES or token in LOCAL_NAMES:
        return "identifier"
    if token in NUMBERS:
        return "number"
    if token in {"+", "-"}:
        return "arith_op"
    if token in {">", "<", "==", "!="}:
        return "cmp_op"
    if token in {"True", "False"}:
        return "bool"
    return None


def sample_alternatives(record: SequenceRecord, step_idx: int, n_alts: int) -> List[int]:
    """Return same-family token alternatives for kappa estimation.

    The step index follows the evaluator convention: `step_idx` predicts `tokens[step_idx + 1]`.
    Alternatives stay within a local semantic class so the perturbation is still code-like.
    """
    token_idx = step_idx + 1
    if token_idx >= len(record.tokens):
        return []
    token = record.tokens[token_idx]
    group = _infer_token_group(token)
    if group is None:
        return []
    alts = [tok for tok in TOKEN_GROUPS[group] if tok != token and tok in TOKEN_TO_ID]
    return [TOKEN_TO_ID[tok] for tok in alts[:n_alts]]


def teacher_action_ids(record: SequenceRecord, step_idx: int) -> List[int]:
    token_idx = step_idx + 1
    if token_idx >= len(record.tokens):
        return []
    token = record.tokens[token_idx]
    group = _infer_token_group(token)
    if group is None:
        return [TOKEN_TO_ID[token]] if token in TOKEN_TO_ID else []
    return [TOKEN_TO_ID[tok] for tok in TOKEN_GROUPS[group] if tok in TOKEN_TO_ID]


def _branch_return_program(rng: random.Random) -> SequenceRecord:
    builder = ProgramBuilder()
    params = [rng.choice(PARAM_NAMES), rng.choice(PARAM_NAMES)]
    func_name = rng.choice(FUNC_NAMES)
    while params[0] == params[1]:
        params[1] = rng.choice(PARAM_NAMES)

    _emit_signature(builder, func_name, params)

    base_var, delta_var, score_var = rng.sample(LOCAL_NAMES, 3)
    first_num = rng.choice(NUMBERS[1:6])
    second_num = rng.choice(NUMBERS[1:6])

    _emit_assignment(builder, base_var, params[0], "+", "", first_num, "base_from_param")
    _emit_comment(builder, _pick_comment(rng, "cache"), "base_assignment")

    _emit_assignment(builder, delta_var, params[1], "-", "", second_num, "delta_from_limit")
    _emit_comment(builder, _pick_comment(rng, "branch"), "delta_assignment")

    builder.add(
        score_var,
        "commit",
        {"kind": "assignment_target", "writes": [score_var], "reads": [base_var, delta_var], "origin": "score_merge"},
    )
    builder.add("=")
    builder.add(base_var, "bridge", {"kind": "assignment_read", "reads": [base_var]})
    builder.add("+")
    builder.add(delta_var, "bridge", {"kind": "assignment_read", "reads": [delta_var]})
    builder.emit_line_break()
    builder.define_symbol(score_var, origin="score_merge", value_hint=f"{base_var}+{delta_var}")
    _emit_comment(builder, _pick_comment(rng, "guard"), "score_assignment")

    _emit_if_header(builder, score_var, ">", params[1])
    _emit_comment(builder, _pick_comment(rng, "guard"), "guard_comment")
    _emit_return(builder, score_var)
    _emit_else_header(builder)
    _emit_comment(builder, _pick_comment(rng, "return"), "else_comment")
    _emit_return(builder, base_var)
    builder.emit_dedent()
    builder.add("<eos>")

    metadata = {
        "pattern": "branch_return",
        "function_name": func_name,
        "params": params,
        "state_targets": [base_var, delta_var, score_var],
    }
    return _to_record(builder, metadata)


def _loop_accumulator_program(rng: random.Random) -> SequenceRecord:
    builder = ProgramBuilder()
    params = [rng.choice(PARAM_NAMES), rng.choice(PARAM_NAMES)]
    while params[0] == params[1]:
        params[1] = rng.choice(PARAM_NAMES)
    func_name = rng.choice(FUNC_NAMES)
    _emit_signature(builder, func_name, params)

    acc_var, step_var, loop_var = rng.sample(LOCAL_NAMES, 3)
    literal = rng.choice(NUMBERS[1:5])
    _emit_assignment(builder, acc_var, params[0], "+", "", literal, "acc_init")
    _emit_comment(builder, _pick_comment(rng, "loop"), "acc_assignment")

    _emit_assignment(builder, step_var, params[1], "+", "", rng.choice(NUMBERS[1:4]), "step_init")
    _emit_comment(builder, _pick_comment(rng, "cache"), "step_assignment")

    _emit_for_loop(builder, loop_var, params[1], acc_var, step_var)
    _emit_comment(builder, _pick_comment(rng, "loop"), "loop_comment")

    _emit_if_header(builder, acc_var, ">", params[1])
    _emit_return(builder, acc_var)
    _emit_else_header(builder)
    _emit_comment(builder, _pick_comment(rng, "return"), "loop_else_comment")
    _emit_return(builder, step_var)
    builder.emit_dedent()
    builder.add("<eos>")

    metadata = {
        "pattern": "loop_accumulator",
        "function_name": func_name,
        "params": params,
        "state_targets": [acc_var, step_var, loop_var],
    }
    return _to_record(builder, metadata)


def _fallback_cache_program(rng: random.Random) -> SequenceRecord:
    builder = ProgramBuilder()
    params = [rng.choice(PARAM_NAMES), rng.choice(PARAM_NAMES)]
    while params[0] == params[1]:
        params[1] = rng.choice(PARAM_NAMES)
    func_name = rng.choice(FUNC_NAMES)
    _emit_signature(builder, func_name, params)

    cache_var, marker_var, result_var = rng.sample(LOCAL_NAMES, 3)
    _emit_assignment(builder, cache_var, params[0], "+", "", rng.choice(NUMBERS[1:4]), "cache_commit")
    _emit_comment(builder, _pick_comment(rng, "cache"), "cache_assignment")

    _emit_assignment(builder, marker_var, params[1], "+", "", rng.choice(NUMBERS[2:6]), "marker_commit")
    _emit_comment(builder, _pick_comment(rng, "branch"), "marker_assignment")

    builder.add(result_var, "commit", {"kind": "assignment_target", "writes": [result_var], "reads": [cache_var, marker_var]})
    builder.add("=")
    builder.add(marker_var, "bridge", {"kind": "assignment_read", "reads": [marker_var]})
    builder.add("-")
    builder.add(cache_var, "bridge", {"kind": "assignment_read", "reads": [cache_var]})
    builder.emit_line_break()
    builder.define_symbol(result_var, origin="difference", value_hint=f"{marker_var}-{cache_var}")
    _emit_comment(builder, _pick_comment(rng, "guard"), "result_assignment")

    _emit_if_header(builder, result_var, "!=", params[0])
    _emit_return(builder, result_var)
    _emit_else_header(builder)
    _emit_comment(builder, _pick_comment(rng, "cache"), "fallback_comment")
    _emit_return(builder, cache_var)
    builder.emit_dedent()
    builder.add("<eos>")

    metadata = {
        "pattern": "fallback_cache",
        "function_name": func_name,
        "params": params,
        "state_targets": [cache_var, marker_var, result_var],
    }
    return _to_record(builder, metadata)


def _alias_guard_return_program(rng: random.Random) -> SequenceRecord:
    builder = ProgramBuilder()
    params = [rng.choice(PARAM_NAMES), rng.choice(PARAM_NAMES)]
    while params[0] == params[1]:
        params[1] = rng.choice(PARAM_NAMES)
    func_name = rng.choice(FUNC_NAMES)
    _emit_signature(builder, func_name, params)

    source_var, alias_var, branch_var, final_var = rng.sample(LOCAL_NAMES, 4)
    first_num = rng.choice(NUMBERS[2:8])
    second_num = rng.choice(NUMBERS[1:6])

    _emit_assignment(builder, source_var, params[0], "+", "", first_num, "source_commit")
    _emit_comment(builder, _pick_comment(rng, "cache"), "source_assignment")

    builder.add(
        alias_var,
        "commit",
        {"kind": "assignment_target", "writes": [alias_var], "reads": [source_var], "origin": "alias_copy"},
    )
    builder.add("=")
    builder.add(
        source_var,
        "bridge",
        {"kind": "assignment_read", "reads": [source_var], "depends_on": source_var, "state_before": builder.snapshot_symbols()},
    )
    builder.emit_line_break()
    builder.define_symbol(alias_var, origin="alias_copy", value_hint=source_var)
    _emit_comment(builder, _pick_comment(rng, "branch"), "alias_assignment")

    _emit_assignment(builder, branch_var, alias_var, "-", "", second_num, "branch_commit")
    _emit_comment(builder, _pick_comment(rng, "guard"), "branch_assignment")

    builder.add(
        final_var,
        "commit",
        {"kind": "assignment_target", "writes": [final_var], "reads": [alias_var, branch_var], "origin": "final_merge"},
    )
    builder.add("=")
    builder.add(alias_var, "bridge", {"kind": "assignment_read", "reads": [alias_var], "depends_on": alias_var})
    builder.add("+")
    builder.add(branch_var, "bridge", {"kind": "assignment_read", "reads": [branch_var], "depends_on": branch_var})
    builder.emit_line_break()
    builder.define_symbol(final_var, origin="final_merge", value_hint=f"{alias_var}+{branch_var}")
    _emit_comment(builder, _pick_comment(rng, "return"), "final_assignment")

    _emit_if_header(builder, final_var, ">", alias_var)
    _emit_comment(builder, _pick_comment(rng, "guard"), "guard_comment")
    _emit_return(builder, final_var)
    _emit_else_header(builder)
    _emit_comment(builder, _pick_comment(rng, "return"), "else_comment")
    _emit_return(builder, alias_var)
    builder.emit_dedent()
    builder.add("<eos>")

    metadata = {
        "pattern": "alias_guard_return",
        "function_name": func_name,
        "params": params,
        "state_targets": [source_var, alias_var, branch_var, final_var],
    }
    return _to_record(builder, metadata)


def _to_record(builder: ProgramBuilder, metadata: Dict[str, Any]) -> SequenceRecord:
    token_ids = [TOKEN_TO_ID[token] for token in builder.tokens]
    oracle_roles = list(builder.roles)
    if len(token_ids) != len(oracle_roles):
        raise ValueError("token/role length mismatch in code domain generator")
    return SequenceRecord(
        token_ids=token_ids,
        tokens=list(builder.tokens),
        oracle_roles=oracle_roles,
        latent_trace=list(builder.latent_trace),
        metadata=metadata,
    )


def _generate_one(rng: random.Random) -> SequenceRecord:
    pattern = rng.choice(DEFAULT_CONFIG["patterns"])
    if pattern == "branch_return":
        return _branch_return_program(rng)
    if pattern == "loop_accumulator":
        return _loop_accumulator_program(rng)
    if pattern == "alias_guard_return":
        return _alias_guard_return_program(rng)
    return _fallback_cache_program(rng)


def _split_size(split_name: str, config: GenerationConfig) -> int:
    mapping = {
        "train": config.train_size,
        "val": config.val_size,
        "test": config.test_size,
        "analysis": config.analysis_size,
    }
    if split_name not in mapping:
        raise ValueError(f"unknown split: {split_name}")
    return mapping[split_name]


def _split_seed(split_name: str, config: GenerationConfig) -> int:
    offset = {"train": 0, "val": 10_000, "test": 20_000, "analysis": 30_000}[split_name]
    return config.seed + offset


def generate_split(split_name: str, config: GenerationConfig) -> List[SequenceRecord]:
    rng = random.Random(_split_seed(split_name, config))
    target_size = _split_size(split_name, config)
    records: List[SequenceRecord] = []
    retries = 0

    while len(records) < target_size:
        record = _generate_one(rng)
        seq_len = len(record.token_ids)
        if config.min_len <= seq_len <= config.max_len:
            records.append(record)
            retries = 0
            continue
        retries += 1
        if retries > DEFAULT_CONFIG["max_retries"]:
            raise RuntimeError(
                f"code domain generator could not satisfy length bounds {config.min_len}-{config.max_len}"
            )
    return records


def build_domain() -> DomainDefinition:
    return DomainDefinition(
        name="code",
        vocab=list(VOCAB),
        token_to_id=dict(TOKEN_TO_ID),
        special_ids=SPECIAL_IDS,
        default_config=DEFAULT_CONFIG,
        generate_split=generate_split,
        sample_alternatives=sample_alternatives,
        teacher_action_ids=teacher_action_ids,
        build_example_payload=_build_example_payload,
    )


def _smoke_check() -> None:
    domain = build_domain()
    cfg = GenerationConfig(
        train_size=4,
        val_size=2,
        test_size=2,
        analysis_size=2,
        min_len=24,
        max_len=96,
        seed=123,
    )
    analysis = domain.generate_split("analysis", cfg)
    assert len(analysis) == 2
    for record in analysis:
        assert len(record.token_ids) == len(record.tokens) == len(record.oracle_roles)
        assert "commit" in record.oracle_roles
        assert "bridge" in record.oracle_roles
        assert "garden" in record.oracle_roles
        payload = domain.build_example_payload(record) if domain.build_example_payload else {}
        assert payload.get("rendered_lines")


if __name__ == "__main__":
    _smoke_check()
    demo = build_domain().generate_split(
        "analysis",
        GenerationConfig(
            train_size=4,
            val_size=2,
            test_size=2,
            analysis_size=1,
            min_len=24,
            max_len=96,
            seed=7,
        ),
    )[0]
    print("Generated code example:")
    for line in _build_example_payload(demo)["rendered_lines"]:
        print(line)
