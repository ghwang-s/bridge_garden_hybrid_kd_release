from __future__ import annotations

from bridge_garden_v2.synthetic_domains import (
    DOMAIN_SCRIPT_LENS,
    FLEXIBLE_SUPPORT_SIZE,
    HIGH_RISK_MAIN_PROB_FLOOR,
    HIGH_RISK_MAIN_PROB_START,
    build_synthetic_domain,
    _domain_surface_items,
)
from bridge_garden_v2.synthetic_evaluator import collect_sampled_path_states
from bridge_garden_v2.exact_oracle import exact_kappa_for_state


class _ZeroLoss:
    def loss_many(self, states, oracle):
        return [0.0 for _ in states]

    def loss_at_state(self, state, oracle):
        return 0.0


def test_synthetic_domains_meet_basic_setup_bounds() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=3)
        assert bundle.name == domain
        assert len(bundle.vocab) == 64
        assert bundle.oracle.eos_id in bundle.oracle.eval_token_ids
        assert bundle.oracle.pad_id not in bundle.oracle.eval_token_ids
        assert bundle.oracle.bos_id not in bundle.oracle.eval_token_ids
        for sample_id in range(3):
            state = bundle.oracle.initial_state(sample_id)
            assert bundle.oracle.remaining_horizon(state) == DOMAIN_SCRIPT_LENS[domain]
            seen_tags = set()
            while not bundle.oracle.is_terminal(state):
                seen_tags.add(bundle.oracle.semantic_tag(state))
                dist = bundle.oracle.next_dist(state)
                eval_ids = bundle.oracle.eval_token_ids_for_state(state)
                assert set(dist).issubset(set(eval_ids))
                action = max(dist, key=dist.get)
                state = bundle.oracle.step(state, action)
            assert seen_tags & set(bundle.high_risk_tags)
            assert seen_tags & set(bundle.flexible_tags)
            assert "syntax_layout" in seen_tags
            assert "context" in seen_tags


def test_synthetic_declared_tags_match_real_decision_tags() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        high_tags = {
            step.semantic_tag
            for step in bundle.oracle.scripts[0]
            if step.expected_role == "high_risk"
        }
        flexible_tags = {
            step.semantic_tag
            for step in bundle.oracle.scripts[0]
            if step.expected_role == "flexible"
        }
        assert high_tags == set(bundle.high_risk_tags)
        assert flexible_tags == set(bundle.flexible_tags)
    assert "variable_binding" not in build_synthetic_domain("code").high_risk_tags


def test_synthetic_surface_template_lengths_match_contract() -> None:
    for domain in ["code", "math", "dialogue"]:
        assert len(_domain_surface_items(domain)) == DOMAIN_SCRIPT_LENS[domain] - 1


def test_synthetic_main_domains_use_semantic_flex_and_multichoice_bridge() -> None:
    filler_prefixes = {
        "code": "alias_",
        "math": "form_",
        "dialogue": "phrase_",
    }
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        high_risk_steps = []
        flexible_steps = []
        for step in bundle.oracle.scripts[0]:
            if step.expected_role == "high_risk":
                high_risk_steps.append(step)
            if step.expected_role == "flexible":
                flexible_steps.append(step)
        assert high_risk_steps
        assert flexible_steps
        assert all(len(step.dist) >= 4 for step in high_risk_steps)
        for step in flexible_steps:
            assert len(step.clean_token_ids) == FLEXIBLE_SUPPORT_SIZE
            assert len(step.eval_token_ids) == FLEXIBLE_SUPPORT_SIZE
            assert len(step.style_dists) == 2
            assert step.style_token_ids == (step.clean_token_ids[:4], step.clean_token_ids[4:8])
            assert step.style_canonical_token_ids == (step.clean_token_ids[0], step.clean_token_ids[4])
            assert all(tuple(token_id for token_id, _ in dist) != () for dist in step.style_dists)
            assert all(set(token_id for token_id, _ in dist) == set(step.clean_token_ids) for dist in step.style_dists)
            probs = [prob for _, prob in step.dist]
            assert probs == sorted(probs, reverse=True)
            assert probs[0] > probs[-1]
            assert abs(sum(probs) - 1.0) < 1e-12
            clean_tokens = [bundle.vocab[token_id] for token_id in step.eval_token_ids]
            assert not any(token.startswith(filler_prefixes[domain]) for token in clean_tokens)


def test_synthetic_high_risk_argmax_is_consistent_across_samples() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=4)
        by_position = {}
        for sample_id, script in enumerate(bundle.oracle.scripts):
            for position, step in enumerate(script):
                if step.expected_role != "high_risk":
                    continue
                argmax = max(step.dist, key=lambda item: item[1])[0]
                by_position.setdefault(position, []).append(argmax)
        assert by_position
        for argmaxes in by_position.values():
            assert len(argmaxes) == 4
            assert len(set(argmaxes)) == 1


def test_code_high_risk_distribution_depends_on_flexible_style() -> None:
    bundle = build_synthetic_domain("code", sample_count=1)
    state = bundle.oracle.initial_state(0)
    saw_plain_high_risk = False
    while not bundle.oracle.is_terminal(state):
        if bundle.oracle.expected_role(state) == "flexible":
            step = bundle.oracle.scripts[0][getattr(state, "position")]
            style1_token = step.style_token_ids[1][0]
            state = bundle.oracle.step(state, style1_token)
            break
        action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
        state = bundle.oracle.step(state, action)
    while not bundle.oracle.is_terminal(state):
        if bundle.oracle.expected_role(state) == "high_risk":
            step = bundle.oracle.scripts[0][getattr(state, "position")]
            if not step.style_dists:
                saw_plain_high_risk = True
            else:
                assert saw_plain_high_risk
                assert bundle.oracle.semantic_tag(state) in {"operator", "return_semantics"}
                assert bundle.oracle.next_dist(state) == dict(step.style_dists[1])
                default_argmax = max(step.dist, key=lambda item: item[1])[0]
                style_argmax = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
                assert style_argmax != default_argmax
                assert bundle.oracle.semantic_tag(state) in bundle.high_risk_tags
                return
        action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
        state = bundle.oracle.step(state, action)
    raise AssertionError("no style-conditioned high-risk state after code flexible style switch")


def test_code_early_high_risk_tokens_are_not_style_conditioned() -> None:
    bundle = build_synthetic_domain("code", sample_count=1)
    steps = [
        step for step in bundle.oracle.scripts[0]
        if step.expected_role == "high_risk"
    ]
    assert [(step.semantic_tag, bool(step.style_dists)) for step in steps] == [
        ("operator", False),
        ("branch_guard", False),
        ("operator", True),
        ("return_semantics", True),
    ]


def test_code_domain_has_enough_flexible_states_for_bottom_split() -> None:
    bundle = build_synthetic_domain("code", sample_count=1)
    flexible_steps = [
        step for step in bundle.oracle.scripts[0]
        if step.expected_role == "flexible"
    ]
    assert len(flexible_steps) >= 3
    assert len(flexible_steps) / DOMAIN_SCRIPT_LENS["code"] > 0.06


def test_synthetic_main_high_risk_teacher_is_sharp_but_not_deterministic() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        high_risk_steps = [
            step
            for step in bundle.oracle.scripts[0]
            if step.expected_role == "high_risk"
        ]
        assert high_risk_steps
        for step in high_risk_steps:
            probs = [prob for _, prob in step.dist]
            assert probs[0] <= HIGH_RISK_MAIN_PROB_START
            assert probs[0] >= HIGH_RISK_MAIN_PROB_FLOOR
            assert probs[0] < 1.0
            assert all(prob > 0.0 for prob in probs[1:])
            assert len(set(round(prob, 12) for prob in probs[1:])) == 1
            assert abs(sum(probs) - 1.0) < 1e-12


def test_synthetic_flexible_overrides_preserve_clean_status() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        state = bundle.oracle.initial_state(0)
        while not bundle.oracle.is_terminal(state):
            if bundle.oracle.expected_role(state) == "flexible":
                off_support = next(token_id for token_id in bundle.oracle.eval_token_ids if token_id not in bundle.oracle.next_dist(state))
                next_state = bundle.oracle.step(state, off_support)
                assert getattr(next_state, "status") == "clean"
                assert getattr(next_state, "prefix")[-1] == bundle.oracle.scripts[0][getattr(state, "position")].clean_token_ids[0]
                return
            action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
            state = bundle.oracle.step(state, action)
        raise AssertionError(f"no flexible state found for {domain}")


def test_synthetic_flexible_style_is_clean_and_visible_in_prefix() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=2)
        for sample_id in range(2):
            state = bundle.oracle.initial_state(sample_id)
            assert getattr(state, "style") == 0
            while not bundle.oracle.is_terminal(state):
                if bundle.oracle.expected_role(state) == "flexible":
                    step = bundle.oracle.scripts[sample_id][getattr(state, "position")]
                    style0_dist = bundle.oracle.next_dist(state)
                    assert style0_dist == dict(step.style_dists[0])
                    style1_token = step.style_token_ids[1][-1]
                    next_state = bundle.oracle.step(state, style1_token)
                    assert getattr(next_state, "status") == "clean"
                    assert getattr(next_state, "style") == 1
                    assert getattr(next_state, "prefix")[-1] == style1_token
                    if not bundle.oracle.is_terminal(next_state):
                        next_step = bundle.oracle.scripts[sample_id][getattr(next_state, "position")]
                        if next_step.style_dists:
                            assert bundle.oracle.next_dist(next_state) == dict(next_step.style_dists[1])
                    break
                action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
                state = bundle.oracle.step(state, action)
            else:
                raise AssertionError(f"no flexible state found for {domain}")


def test_synthetic_high_risk_offsupport_overrides_violate() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        state = bundle.oracle.initial_state(0)
        while not bundle.oracle.is_terminal(state):
            if bundle.oracle.expected_role(state) == "high_risk":
                off_support = next(token_id for token_id in bundle.oracle.eval_token_ids if token_id not in bundle.oracle.next_dist(state))
                next_state = bundle.oracle.step(state, off_support)
                assert getattr(next_state, "status") == "violation"
                return
            action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
            state = bundle.oracle.step(state, action)
        raise AssertionError(f"no high-risk state found for {domain}")


def test_synthetic_structural_offsupport_overrides_preserve_clean_status() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        state = bundle.oracle.initial_state(0)
        while not bundle.oracle.is_terminal(state):
            if bundle.oracle.expected_role(state) == "structural":
                correct_token = next(iter(bundle.oracle.next_dist(state)))
                off_support = next(token_id for token_id in bundle.oracle.eval_token_ids if token_id != correct_token)
                next_state = bundle.oracle.step(state, off_support)
                assert getattr(next_state, "status") == "clean"
                assert getattr(next_state, "prefix")[-1] == correct_token
                assert correct_token in bundle.oracle.eval_token_ids_for_state(state)
                assert len(bundle.oracle.eval_token_ids_for_state(state)) == len(bundle.oracle.eval_token_ids)
                return
            action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
            state = bundle.oracle.step(state, action)
        raise AssertionError(f"no structural state found for {domain}")


def test_synthetic_context_offsupport_overrides_preserve_clean_status() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        state = bundle.oracle.initial_state(0)
        while not bundle.oracle.is_terminal(state):
            if bundle.oracle.expected_role(state) == "context":
                correct_token = next(iter(bundle.oracle.next_dist(state)))
                off_support = next(token_id for token_id in bundle.oracle.eval_token_ids if token_id != correct_token)
                next_state = bundle.oracle.step(state, off_support)
                assert getattr(next_state, "status") == "clean"
                assert getattr(next_state, "prefix")[-1] == correct_token
                assert correct_token in bundle.oracle.eval_token_ids_for_state(state)
                return
            action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
            state = bundle.oracle.step(state, action)
        raise AssertionError(f"no context state found for {domain}")


def test_synthetic_context_kappa_reuses_identical_canonical_prefixes() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        state = next(
            state
            for state in collect_sampled_path_states(bundle.oracle, [0], seed=0)
            if bundle.oracle.expected_role(state) == "context"
        )
        result = exact_kappa_for_state(
            bundle.oracle,
            _ZeroLoss(),
            state,
            continuation="expectation",
            max_expansions_per_action=250_000,
        )
        stats = result.computation_stats or {}
        assert stats["constant_next_state_shortcut"] is True
        assert stats["expanded_state_total"] == 0
        assert result.state_kappa == 0.0


def test_synthetic_structural_kappa_shortcuts_identical_canonical_prefixes() -> None:
    for domain in ["code", "math", "dialogue"]:
        bundle = build_synthetic_domain(domain, sample_count=1)
        state = next(
            state
            for state in collect_sampled_path_states(bundle.oracle, [0], seed=0)
            if bundle.oracle.expected_role(state) == "structural"
        )
        result = exact_kappa_for_state(
            bundle.oracle,
            _ZeroLoss(),
            state,
            continuation="expectation",
            max_expansions_per_action=250_000,
        )
        stats = result.computation_stats or {}
        assert stats["constant_next_state_shortcut"] is True
        assert stats["expanded_state_total"] == 0
        assert result.state_kappa == 0.0


def test_negative_control_domain_has_no_high_risk_tags() -> None:
    bundle = build_synthetic_domain("negative_control", sample_count=2)
    assert bundle.high_risk_tags == ()
    state = bundle.oracle.initial_state(0)
    seen_control = False
    seen_excluded = False
    while not bundle.oracle.is_terminal(state):
        tag = bundle.oracle.semantic_tag(state)
        if tag == "surface_ngram":
            seen_control = True
            assert len(bundle.oracle.next_dist(state)) == 4
        if tag == "syntax_layout":
            seen_excluded = True
            assert len(bundle.oracle.next_dist(state)) == 1
        action = max(bundle.oracle.next_dist(state), key=bundle.oracle.next_dist(state).get)
        state = bundle.oracle.step(state, action)
    assert seen_control
    assert seen_excluded


def test_dialogue_forbidden_constraint_stays_out_of_assistant_reply() -> None:
    bundle = build_synthetic_domain("dialogue", sample_count=8)
    for sample_id in range(8):
        script = bundle.oracle.scripts[sample_id]
        tokens = [bundle.vocab[max(step.dist, key=lambda item: item[1])[0]] for step in script[:-1]]
        forbidden_tokens = [
            bundle.vocab[max(step.dist, key=lambda item: item[1])[0]]
            for step in script
            if step.semantic_tag == "forbidden_constraint"
        ]
        assert len(forbidden_tokens) == 1
        assistant_start = tokens.index("assistant")
        assert forbidden_tokens[0] not in tokens[assistant_start:]


def test_synthetic_mode_surfaces_are_human_readable() -> None:
    expected_fragments = {
        "code": [
            "result = 0",
            "# direct",
            "score = x + limit",
            "if score > limit",
            "result = result + score",
            "return result",
        ],
        "math": [
            "substitute x = 12",
            "compute 2 * x + 3 = 27",
            "so final = 27",
        ],
        "dialogue": [
            "please keep it polite and brief",
            "client Mia asks about the meeting",
            "do not mention budget",
            "assistant reply with the schedule",
        ],
    }
    forbidden_fragments = {
        "code": ["result = guarded", "return guarded"],
        "math": [],
        "dialogue": ["keep it confirm", "keep it reminder", "keep it reply", "keep it invite"],
    }
    for domain in ["code", "math", "dialogue"]:
        surface = _mode_surface(domain)
        for fragment in expected_fragments[domain]:
            assert fragment in surface
        for fragment in forbidden_fragments[domain]:
            assert fragment not in surface


def test_synthetic_domain_rejects_out_of_contract_sizes() -> None:
    try:
        build_synthetic_domain("code", script_len=32)
    except ValueError as exc:
        assert "script_len" in str(exc)
    else:
        raise AssertionError("expected script_len ValueError")

    try:
        build_synthetic_domain("code", vocab_size=128)
    except ValueError as exc:
        assert "vocab_size" in str(exc)
    else:
        raise AssertionError("expected vocab_size ValueError")


def _mode_surface(domain: str) -> str:
    bundle = build_synthetic_domain(domain, sample_count=1)
    state = bundle.oracle.initial_state(0)
    tokens = []
    while not bundle.oracle.is_terminal(state):
        dist = bundle.oracle.next_dist(state)
        if not dist:
            break
        action = max(dist, key=lambda token_id: (dist[token_id], -int(token_id)))
        tokens.append(bundle.vocab[action])
        state = bundle.oracle.step(state, action)
    return " ".join(tokens).replace(" NEWLINE ", " ").replace(" INDENT ", " ").replace(" DEDENT ", " ")
