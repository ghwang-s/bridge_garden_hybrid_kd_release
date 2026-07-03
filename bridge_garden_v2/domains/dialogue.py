from __future__ import annotations

import sys
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_CURRENT_DIR) in sys.path:
    sys.path.remove(str(_CURRENT_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import random
from typing import Any, Dict, List, Sequence, Tuple

from bridge_garden_v2.schema import (
    DomainDefinition,
    GenerationConfig,
    SequenceRecord,
    SpecialTokenIds,
)


FEMALE_NAMES = ["Grace", "Maya", "Nora", "Lena", "Chloe", "Iris"]
MALE_NAMES = ["Liam", "Noah", "Ethan", "Miles", "Owen", "Lucas"]
ROLES = ["doctor", "painter", "teacher", "clerk", "writer", "baker", "nurse", "guard"]
TRAITS = ["calm", "bright", "careful", "quiet", "kind", "steady", "curious", "gentle"]
ITEMS = ["map", "notebook", "key", "letter", "schedule", "lantern", "ticket", "badge"]
LOCATIONS = ["harbor", "studio", "garden", "library", "station", "market", "bridge", "hall"]
COLORS = ["blue", "silver", "small", "folded", "plain", "marked", "neat", "worn"]
CONTAINERS = ["bag", "drawer", "box", "desk", "pouch", "shelf"]
ADVERBS = [
    "softly", "calmly", "briefly", "quietly", "warmly", "gently", "plainly", "slowly",
    "politely", "clearly", "lightly", "carefully", "kindly", "firmly",
]
SPEECH_VERBS = ["said", "added", "noted", "whispered", "replied", "murmured", "explained", "answered", "agreed"]
REPLIES = ["yes", "thanks", "good", "ready", "right", "sure", "please", "fine", "okay", "maybe", "agreed"]
FEELINGS = ["relieved", "glad", "calm", "proud", "safe", "steady", "hopeful", "settled", "eased"]
DISCOURSE = ["Later", "Then", "Afterward", "Soon", "Meanwhile", "Eventually", "Gently", "Quietly"]
CLOSE_MODIFIERS = ["quiet", "warm", "narrow", "nearby", "still", "open", "restful", "shaded"]
SCENE_WORDS = ["again", "today", "outside", "nearby", "together", "carefully", "plainly", "briefly", "just"]
TONE_WORDS = ["softly", "warmly", "plainly", "gently", "briefly", "carefully", "kindly", "lightly"]
FUNCTION_WORDS = [
    "the",
    "a",
    "at",
    "near",
    "inside",
    "about",
    "and",
    "that",
    "felt",
    "placed",
    "showed",
    "kept",
    "met",
    "thanked",
    "would",
    "guide",
    "trip",
    "stayed",
    "safe",
    "in",
]
PUNCT = [".", ","]
SPECIAL = ["<pad>", "<bos>", "<eos>"]


def _build_vocab() -> Tuple[List[str], Dict[str, int]]:
    vocab = list(SPECIAL)
    for group in [
        FEMALE_NAMES,
        MALE_NAMES,
        ROLES,
        TRAITS,
        ITEMS,
        LOCATIONS,
        COLORS,
        CONTAINERS,
        ADVERBS,
        SPEECH_VERBS,
        REPLIES,
        FEELINGS,
        DISCOURSE,
        CLOSE_MODIFIERS,
        SCENE_WORDS,
        TONE_WORDS,
        FUNCTION_WORDS,
        ["he", "she", "him", "her"],
        PUNCT,
    ]:
        for token in group:
            if token not in vocab:
                vocab.append(token)
    return vocab, {tok: idx for idx, tok in enumerate(vocab)}


VOCAB, TOKEN_TO_ID = _build_vocab()
SPECIAL_IDS = SpecialTokenIds(
    pad=TOKEN_TO_ID["<pad>"],
    bos=TOKEN_TO_ID["<bos>"],
    eos=TOKEN_TO_ID["<eos>"],
)


def _pronouns(gender: str) -> Dict[str, str]:
    if gender == "f":
        return {"subj": "she", "obj": "her"}
    return {"subj": "he", "obj": "him"}


class _SequenceBuilder:
    def __init__(self) -> None:
        self.tokens: List[str] = ["<bos>"]
        self.roles: List[str] = ["middle"]
        self.latent_trace: List[Dict[str, Any]] = []

    def append(self, token: str, role: str) -> None:
        self.tokens.append(token)
        self.roles.append(role)

    def extend(self, items: Sequence[Tuple[str, str]]) -> None:
        for token, role in items:
            self.append(token, role)

    def mark_clause(
        self,
        clause_name: str,
        start_idx: int,
        state_reads: List[str],
        state_updates: List[str],
        note: str,
    ) -> None:
        end_idx = len(self.tokens) - 1
        self.latent_trace.append(
            {
                "clause": clause_name,
                "span": [start_idx, end_idx],
                "text": " ".join(self.tokens[start_idx : end_idx + 1]),
                "state_reads": state_reads,
                "state_updates": state_updates,
                "note": note,
            }
        )


def _emit_intro(builder: _SequenceBuilder, state: Dict[str, str]) -> None:
    start = len(builder.tokens)
    builder.extend(
        [
            (state["name1"], "commit"),
            ("the", "middle"),
            (state["trait1"], "garden"),
            (state["role1"], "commit"),
            ("met", "middle"),
            (state["name2"], "commit"),
            ("the", "middle"),
            (state["trait2"], "garden"),
            (state["role2"], "commit"),
            ("at", "middle"),
            ("the", "middle"),
            (state["location"], "commit"),
            (".", "middle"),
        ]
    )
    builder.mark_clause(
        "intro_commit",
        start,
        state_reads=[],
        state_updates=[
            f"entity_1={state['name1']}/{state['role1']}",
            f"entity_2={state['name2']}/{state['role2']}",
            f"location={state['location']}",
        ],
        note="Commits both entities and the shared location.",
    )


def _emit_item_commit(builder: _SequenceBuilder, state: Dict[str, str]) -> None:
    start = len(builder.tokens)
    builder.extend(
        [
            (state["marker1"], "garden"),
            (state["scene1"], "garden"),
            (state["pron1_subj"], "garden"),
            ("placed", "middle"),
            ("the", "middle"),
            (state["item_color"], "garden"),
            (state["item"], "commit"),
            ("inside", "middle"),
            ("the", "middle"),
            (state["container"], "garden"),
            (".", "middle"),
        ]
    )
    builder.mark_clause(
        "item_commit",
        start,
        state_reads=["entity_1"],
        state_updates=[f"item={state['item']}", f"container={state['container']}"],
        note="The first entity introduces the shared item. Color/container stay local.",
    )


def _emit_dialogue(builder: _SequenceBuilder, state: Dict[str, str]) -> None:
    start = len(builder.tokens)
    builder.extend(
        [
            (state["marker2"], "garden"),
            (state["scene2"], "garden"),
            (state["name2"], "garden"),
            (state["dialogue_adverb"], "garden"),
            (state["speech_verb"], "garden"),
            (state["reply"], "garden"),
            (state["reply_tone"], "garden"),
            ("about", "middle"),
            ("the", "middle"),
            (state["item"], "bridge"),
            (".", "middle"),
        ]
    )
    builder.mark_clause(
        "dialogue_reference",
        start,
        state_reads=["entity_2", "item"],
        state_updates=[],
        note="Garden wording varies, but the dialogue still references the committed item.",
    )


def _emit_role_recall(builder: _SequenceBuilder, state: Dict[str, str]) -> None:
    start = len(builder.tokens)
    builder.extend(
        [
            (state["marker3"], "garden"),
            (state["scene3"], "garden"),
            ("the", "middle"),
            (state["role1"], "bridge"),
            ("showed", "middle"),
            (state["pron2_obj"], "bridge"),
            ("the", "middle"),
            (state["item"], "bridge"),
            ("near", "middle"),
            ("the", "middle"),
            (state["location"], "bridge"),
            (".", "middle"),
        ]
    )
    builder.mark_clause(
        "role_recall",
        start,
        state_reads=["role1", "entity_2", "item", "location"],
        state_updates=[],
        note="Reads previously committed state using role and location aliases.",
    )


def _emit_reaction(builder: _SequenceBuilder, state: Dict[str, str]) -> None:
    start = len(builder.tokens)
    builder.extend(
        [
            (state["marker4"], "garden"),
            (state["scene4"], "garden"),
            (state["pron2_subj"], "garden"),
            ("kept", "middle"),
            ("the", "middle"),
            (state["item"], "bridge"),
            ("and", "middle"),
            (state["pron1_subj"], "garden"),
            ("felt", "middle"),
            (state["feeling_tone"], "garden"),
            (state["feeling"], "garden"),
            (".", "middle"),
        ]
    )
    builder.mark_clause(
        "reaction_bridge",
        start,
        state_reads=["entity_2", "entity_1", "item"],
        state_updates=[],
        note="Bridge tokens track who keeps the item; feeling is purely local.",
    )


def _emit_close(builder: _SequenceBuilder, state: Dict[str, str]) -> None:
    start = len(builder.tokens)
    builder.extend(
        [
            (state["marker5"], "garden"),
            (state["scene5"], "garden"),
            ("in", "middle"),
            ("the", "middle"),
            (state["close_modifier"], "garden"),
            (state["location"], "bridge"),
            ("the", "middle"),
            (state["item"], "bridge"),
            ("stayed", "middle"),
            ("safe", "garden"),
            (".", "middle"),
        ]
    )
    builder.mark_clause(
        "close_summary",
        start,
        state_reads=["location", "item"],
        state_updates=[],
        note="Final summary reuses the same location and item while keeping modifiers flexible.",
    )


def _make_state(rng: random.Random) -> Dict[str, str]:
    genders = [("f", rng.choice(FEMALE_NAMES)), ("m", rng.choice(MALE_NAMES))]
    rng.shuffle(genders)
    (g1, name1), (g2, name2) = genders
    pron1 = _pronouns(g1)
    pron2 = _pronouns(g2)
    role1, role2 = rng.sample(ROLES, 2)
    return {
        "name1": name1,
        "name2": name2,
        "gender1": g1,
        "gender2": g2,
        "role1": role1,
        "role2": role2,
        "trait1": rng.choice(TRAITS),
        "trait2": rng.choice(TRAITS),
        "location": rng.choice(LOCATIONS),
        "item": rng.choice(ITEMS),
        "item_color": rng.choice(COLORS),
        "container": rng.choice(CONTAINERS),
        "dialogue_adverb": rng.choice(ADVERBS),
        "speech_verb": rng.choice(SPEECH_VERBS),
        "reply": rng.choice(REPLIES),
        "feeling": rng.choice(FEELINGS),
        "close_modifier": rng.choice(CLOSE_MODIFIERS),
        "marker1": rng.choice(DISCOURSE),
        "marker2": rng.choice(DISCOURSE),
        "marker3": rng.choice(DISCOURSE),
        "marker4": rng.choice(DISCOURSE),
        "marker5": rng.choice(DISCOURSE),
        "scene1": rng.choice(SCENE_WORDS),
        "scene2": rng.choice(SCENE_WORDS),
        "scene3": rng.choice(SCENE_WORDS),
        "scene4": rng.choice(SCENE_WORDS),
        "scene5": rng.choice(SCENE_WORDS),
        "reply_tone": rng.choice(TONE_WORDS),
        "feeling_tone": rng.choice(TONE_WORDS),
        "pron1_subj": pron1["subj"],
        "pron1_obj": pron1["obj"],
        "pron2_subj": pron2["subj"],
        "pron2_obj": pron2["obj"],
    }


def _finalize_record(builder: _SequenceBuilder, state: Dict[str, str]) -> SequenceRecord:
    builder.append("<eos>", "middle")
    token_ids = [TOKEN_TO_ID[token] for token in builder.tokens]
    metadata = {
        "text": " ".join(token for token in builder.tokens[1:-1]),
        "state": {
            "entity_1": {"name": state["name1"], "role": state["role1"], "gender": state["gender1"]},
            "entity_2": {"name": state["name2"], "role": state["role2"], "gender": state["gender2"]},
            "item": state["item"],
            "location": state["location"],
        },
    }
    return SequenceRecord(
        token_ids=token_ids,
        tokens=list(builder.tokens),
        oracle_roles=list(builder.roles),
        latent_trace=list(builder.latent_trace),
        metadata=metadata,
    )


def generate_dialogue_sequence(rng: random.Random) -> SequenceRecord:
    state = _make_state(rng)
    builder = _SequenceBuilder()
    _emit_intro(builder, state)
    _emit_item_commit(builder, state)
    _emit_dialogue(builder, state)
    _emit_role_recall(builder, state)
    _emit_reaction(builder, state)
    if rng.random() < 0.85:
        _emit_close(builder, state)
    return _finalize_record(builder, state)


def _sequence_within_bounds(record: SequenceRecord, config: GenerationConfig) -> bool:
    length = len(record.token_ids)
    return config.min_len <= length <= config.max_len


def generate_split(split_name: str, config: GenerationConfig) -> List[SequenceRecord]:
    split_sizes = {
        "train": config.train_size,
        "val": config.val_size,
        "test": config.test_size,
        "analysis": config.analysis_size,
    }
    if split_name not in split_sizes:
        raise ValueError(f"Unknown split: {split_name}")

    split_seed_offset = {"train": 0, "val": 10_000, "test": 20_000, "analysis": 30_000}[split_name]
    rng = random.Random(config.seed + split_seed_offset)
    records: List[SequenceRecord] = []
    target = split_sizes[split_name]
    while len(records) < target:
        record = generate_dialogue_sequence(rng)
        if _sequence_within_bounds(record, config):
            records.append(record)
    return records


def build_example_payload(record: SequenceRecord) -> Dict[str, Any]:
    spans = []
    for event in record.latent_trace:
        spans.append(
            {
                "clause": event["clause"],
                "span": event["span"],
                "state_reads": event["state_reads"],
                "state_updates": event["state_updates"],
                "text": event["text"],
            }
        )
    trace_rows = []
    for idx, event in enumerate(spans[:6], start=1):
        reads = ",".join(event["state_reads"]) or "-"
        writes = ",".join(event["state_updates"]) or "-"
        trace_rows.append(
            {
                "label": f"C{idx} {event['clause']}",
                "text": f"reads={reads}; writes={writes}; {event['text']}",
            }
        )
    content_tokens = record.tokens[1:-1]
    line_breaks = []
    for event in spans[:-1]:
        end_idx = int(event["span"][1]) - 1
        if 0 <= end_idx < len(content_tokens):
            line_breaks.append(end_idx)
    return {
        "tokens": content_tokens,
        "roles": record.oracle_roles[1:-1],
        "trace_rows": trace_rows,
        "line_breaks": line_breaks,
        "caption": "Commit tokens introduce entities, item, or location; Bridge tokens recover that latent state through pronouns, role recall, and item/location references; Garden tokens vary local wording.",
        "title": "Dialogue Example",
        "latent_trace": spans,
        "state": record.metadata["state"],
    }


def _token_alternatives(record: SequenceRecord, token: str, role: str) -> List[str]:
    state = record.metadata.get("state", {})
    entity_1 = state.get("entity_1", {})
    entity_2 = state.get("entity_2", {})

    if token == entity_1.get("name"):
        return [entity_2["name"]] if entity_2.get("name") else []
    if token == entity_2.get("name"):
        return [entity_1["name"]] if entity_1.get("name") else []
    if token == entity_1.get("role"):
        return [entity_2["role"]] if entity_2.get("role") else []
    if token == entity_2.get("role"):
        return [entity_1["role"]] if entity_1.get("role") else []
    if token == "he":
        return ["she"]
    if token == "she":
        return ["he"]
    if token == "him":
        return ["her"]
    if token == "her":
        return ["him"]
    if token == state.get("item"):
        return [t for t in ITEMS if t != token]
    if token == state.get("location"):
        return [t for t in LOCATIONS if t != token]
    if role == "garden":
        garden_groups = [TRAITS, COLORS, CONTAINERS, ADVERBS, SPEECH_VERBS, REPLIES, FEELINGS, CLOSE_MODIFIERS, DISCOURSE, SCENE_WORDS, TONE_WORDS]
        for group in garden_groups:
            if token in group:
                return [t for t in group if t != token]
    if role == "commit":
        commit_groups = [FEMALE_NAMES + MALE_NAMES, ROLES, ITEMS, LOCATIONS]
        for group in commit_groups:
            if token in group:
                return [t for t in group if t != token]
    return []


def sample_alternatives(record: SequenceRecord, step_idx: int, n_alts: int) -> List[int]:
    token_idx = step_idx + 1
    if token_idx <= 0 or token_idx >= len(record.tokens) - 1:
        return []
    token = record.tokens[token_idx]
    role = record.oracle_roles[token_idx]
    candidates = _token_alternatives(record, token, role)
    return [TOKEN_TO_ID[t] for t in candidates if t in TOKEN_TO_ID and t != token][:n_alts]


def teacher_action_ids(record: SequenceRecord, step_idx: int) -> List[int]:
    token_idx = step_idx + 1
    if token_idx <= 0 or token_idx >= len(record.tokens) - 1:
        return []
    token = record.tokens[token_idx]
    role = record.oracle_roles[token_idx]
    candidates = _token_alternatives(record, token, role)
    if token in TOKEN_TO_ID:
        return [TOKEN_TO_ID[token]] + [TOKEN_TO_ID[t] for t in candidates if t in TOKEN_TO_ID and t != token]
    return [TOKEN_TO_ID[t] for t in candidates if t in TOKEN_TO_ID]


def build_domain() -> DomainDefinition:
    return DomainDefinition(
        name="dialogue",
        vocab=VOCAB,
        token_to_id=TOKEN_TO_ID,
        special_ids=SPECIAL_IDS,
        default_config={
            "description": "Mini-story dialogue domain with explicit entity/item/location state.",
            "min_recommended_length": 48,
            "max_recommended_length": 80,
        },
        generate_split=generate_split,
        sample_alternatives=sample_alternatives,
        teacher_action_ids=teacher_action_ids,
        build_example_payload=build_example_payload,
    )


def _smoke_check() -> None:
    config = GenerationConfig(
        train_size=2,
        val_size=1,
        test_size=1,
        analysis_size=1,
        min_len=30,
        max_len=90,
        seed=7,
    )
    records = generate_split("analysis", config)
    assert len(records) == 1
    record = records[0]
    assert record.tokens[0] == "<bos>"
    assert record.tokens[-1] == "<eos>"
    assert len(record.tokens) == len(record.oracle_roles)
    assert "commit" in record.oracle_roles
    assert "bridge" in record.oracle_roles
    assert "garden" in record.oracle_roles
    assert record.latent_trace
    payload = build_example_payload(record)
    assert payload["tokens"] == record.tokens[1:-1]
    assert payload["roles"] == record.oracle_roles[1:-1]
    assert isinstance(sample_alternatives(record, 0, 3), list)


if __name__ == "__main__":
    _smoke_check()
    sample = generate_split(
        "analysis",
        GenerationConfig(
            train_size=1,
            val_size=1,
            test_size=1,
            analysis_size=1,
            min_len=30,
            max_len=90,
            seed=19,
        ),
    )[0]
    print(sample.metadata["text"])
    print(sample.oracle_roles)
