"""
Quantum Parser - Main parsing engine with parallel hypothesis exploration.

Implements the quantum parsing algorithm that maintains multiple parse
interpretations simultaneously.
"""

import itertools
import time
from typing import List, Optional
from copy import deepcopy

from .data_structures import (
    Word, Node, Edge, Hypothesis, ParseChart,
    create_initial_chart, ParserConfig
)
from .dsl import Grammar, Rule, load_grammar
from .matcher import find_matches, Match
from .scorer import score_hypothesis
from .enums import ConnectionType


class ParseResourceExceeded(Exception):
    """Raised by :meth:`QuantumParser.parse` when a resource cap set on
    ``ParserConfig`` (``max_ruleset_hypotheses`` / ``max_parse_seconds``,
    both ``None``/disabled by default) is hit mid-parse.

    Opt-in only: this can never be raised unless a caller explicitly passes
    a ``config_override`` (or otherwise builds a ``ParserConfig``) with one
    of those fields set -- every existing caller that relies on the default
    ``ParserConfig()`` sees byte-identical behavior, cap or no cap.
    """

    def __init__(self, message: str, ruleset_name: str, hypothesis_count: int):
        super().__init__(message)
        self.ruleset_name = ruleset_name
        self.hypothesis_count = hypothesis_count


class QuantumParser:
    """
    Quantum parser with parallel hypothesis exploration.

    Attributes:
        grammar: Loaded grammar rules
        config: Parser configuration
    """

    def __init__(self, grammar_path: str, config: Optional[ParserConfig] = None):
        """
        Initialize parser with grammar file.

        Args:
            grammar_path: Path to grammar JSON file
            config: Parser configuration (uses default if None)
        """
        self.grammar = load_grammar(grammar_path)
        self.config = config if config is not None else ParserConfig()

    def parse(self, words: List[Word], config_override: Optional[ParserConfig] = None) -> ParseChart:
        """
        Parse a list of words into a ParseChart with multiple hypotheses.

        Args:
            words: Input sentence as list of Word objects
            config_override: use this config instead of ``self.config`` for
                this call only (``self.config``/every other caller is
                untouched). Additive, opt-in hook for
                ``max_ruleset_hypotheses``/``max_parse_seconds`` -- passing
                ``None`` (the default) is byte-identical to before this
                parameter existed.

        Returns:
            ParseChart containing all viable parse hypotheses, ranked by score

        Raises:
            ParseResourceExceeded: only possible when ``config_override`` (or
                ``self.config``) sets ``max_ruleset_hypotheses`` or
                ``max_parse_seconds`` -- both ``None``/disabled by default.
        """
        config = config_override if config_override is not None else self.config

        # Validate input
        if not words:
            raise ValueError("Cannot parse empty sentence")

        if len(words) > config.max_sentence_length:
            raise ValueError(f"Sentence too long ({len(words)} > {config.max_sentence_length})")

        # Create initial chart
        chart = create_initial_chart(words, config)

        start_time = time.monotonic()
        deadline = start_time + config.max_parse_seconds if config.max_parse_seconds is not None else None

        # Apply rulesets in order
        for ruleset_name in self.grammar.order:
            ruleset = self.grammar.rulesets[ruleset_name]

            if config.max_parse_seconds is not None and time.monotonic() - start_time > config.max_parse_seconds:
                raise ParseResourceExceeded(
                    f"parse exceeded {config.max_parse_seconds}s wall-clock cap "
                    f"(at ruleset {ruleset_name!r})",
                    ruleset_name, len(chart.hypotheses),
                )

            # Generate new hypotheses by applying rules
            new_hypotheses = []

            for current_hyp in chart.hypotheses:
                # Per-hypothesis time check (not just per-ruleset/per-combo):
                # a single non-combinatorial hypothesis can still be slow on
                # its own (e.g. apply_ruleset_recursively's up-to-100-iteration
                # inner loop against a large/complex hypothesis), with no
                # itertools.product combo ever entered for the time check
                # below to catch -- measured on a real 60-word corpus
                # sentence that ran well past this ruleset's time budget with
                # zero ambiguous branching involved.
                if (config.max_parse_seconds is not None
                        and time.monotonic() - start_time > config.max_parse_seconds):
                    raise ParseResourceExceeded(
                        f"parse exceeded {config.max_parse_seconds}s wall-clock cap "
                        f"(mid-ruleset {ruleset_name!r}, per-hypothesis check)",
                        ruleset_name, len(new_hypotheses),
                    )

                # Collect ALL possible rule matches for this hypothesis
                all_matches = []

                # Try each unconsumed node as potential anchor
                for unconsumed_idx in current_hyp.get_unconsumed():
                    # Try each rule in ruleset
                    for rule in ruleset.rules:
                        # Find all ways this rule can match
                        matches = find_matches(current_hyp, unconsumed_idx, rule)
                        all_matches.extend(matches)

                # SMARTER QUANTUM BRANCHING: Only branch on actual ambiguity
                if len(all_matches) > 1:
                    # Group by anchor: an anchor with exactly one match is a
                    # deterministic, independent transformation; an anchor
                    # with several matches is genuine ambiguity (multiple
                    # ways to parse THAT anchor). Previously, the presence of
                    # ANY ambiguous anchor discarded every OTHER anchor's
                    # independent match entirely (each branch applied only
                    # the one conflicting match and nothing else) -- so e.g.
                    # "mary thinks [the ball] [is in the shed]": "thinks"
                    # being ambiguous (intransitive vs. transitive-object)
                    # silently ate the independent, unambiguous "is -> PREDICATE"
                    # transformation on every branch. Now: independent matches
                    # are applied on top of every branch, and only the
                    # Cartesian product of the genuinely ambiguous anchors'
                    # alternatives is explored.
                    by_anchor: dict = {}
                    for m in all_matches:
                        by_anchor.setdefault(m.anchor_idx, []).append(m)

                    independent_matches = [ms[0] for ms in by_anchor.values() if len(ms) == 1]
                    ambiguous_groups = [ms for ms in by_anchor.values() if len(ms) > 1]

                    def _apply_all(base_hyp, matches):
                        result = base_hyp
                        for match in matches:
                            # Per-match deadline check (opt-in, None by default):
                            # the per-combo checks below this closure only fire
                            # once a whole _apply_all call returns, so a single
                            # call chewing through a long `matches` list (e.g.
                            # independent_matches on a large hypothesis, each
                            # apply_rule deep-copying via Hypothesis.copy) can
                            # run well past max_parse_seconds before either
                            # check downstream ever runs. Checking here bounds
                            # that single call too.
                            if (deadline is not None
                                    and time.monotonic() > deadline):
                                raise ParseResourceExceeded(
                                    f"parse exceeded {config.max_parse_seconds}s wall-clock cap "
                                    f"(mid-_apply_all, ruleset {ruleset_name!r})",
                                    ruleset_name, len(new_hypotheses),
                                )
                            result = apply_rule(result, match)
                            if match.rule.recursive:
                                result = apply_ruleset_recursively(result, ruleset, deadline=deadline,
                                                                    ruleset_name=ruleset_name)
                        return result

                    if not ambiguous_groups:
                        # ALL DIFFERENT ANCHORS: independent transformations
                        new_hypotheses.append(_apply_all(current_hyp, independent_matches))
                    else:
                        # Branch only over the ambiguous anchors' alternatives;
                        # independent matches still land on every branch.
                        # Bounded (opt-in): the Cartesian product itself is
                        # what runs away on a long/highly-ambiguous sentence
                        # (a handful of ambiguous anchors each with a few
                        # alternatives multiplies out fast -- see
                        # ParseResourceExceeded's docstring); checking the
                        # running count INSIDE this loop, not after building
                        # the full product, is what keeps this from ever
                        # materializing more than max_ruleset_hypotheses+1
                        # deep-copied Hypothesis objects for one ruleset pass.
                        for combo in itertools.product(*ambiguous_groups):
                            new_hyp = _apply_all(current_hyp, independent_matches)
                            new_hyp = _apply_all(new_hyp, combo)
                            new_hypotheses.append(new_hyp)
                            if (config.max_ruleset_hypotheses is not None
                                    and len(new_hypotheses) > config.max_ruleset_hypotheses):
                                raise ParseResourceExceeded(
                                    f"ruleset {ruleset_name!r} exceeded "
                                    f"max_ruleset_hypotheses={config.max_ruleset_hypotheses} "
                                    "(unbounded ambiguous-anchor combinatorics)",
                                    ruleset_name, len(new_hypotheses),
                                )
                            if (config.max_parse_seconds is not None
                                    and time.monotonic() - start_time > config.max_parse_seconds):
                                raise ParseResourceExceeded(
                                    f"parse exceeded {config.max_parse_seconds}s wall-clock cap "
                                    f"(mid-ruleset {ruleset_name!r})",
                                    ruleset_name, len(new_hypotheses),
                                )

                elif len(all_matches) == 1:
                    # SINGLE MATCH: No ambiguity, just transform in-place
                    match = all_matches[0]
                    new_hyp = apply_rule(current_hyp, match)

                    # If recursive rule, keep applying until no more matches
                    if match.rule.recursive:
                        new_hyp = apply_ruleset_recursively(new_hyp, ruleset, deadline=deadline,
                                                             ruleset_name=ruleset_name)

                    new_hypotheses.append(new_hyp)

                else:
                    # NO MATCHES: Keep hypothesis unchanged
                    new_hypotheses.append(current_hyp)

            # Score all hypotheses
            for hyp in new_hypotheses:
                hyp.score = score_hypothesis(hyp, chart.embeddings)

            # DEDUPLICATION: Remove structurally equivalent hypotheses.
            #
            # Keyed by Hypothesis.equivalence_key() (constructed from exactly
            # the fields is_equivalent compares) instead of the original
            # pairwise "for hyp: for existing in deduplicated: is_equivalent"
            # scan: two hypotheses share a key iff is_equivalent(...) would
            # have returned True for them, so this dict produces the same
            # surviving set (same score-tie-break rule: keep the earlier one
            # unless a strictly-better-scored equivalent shows up) in O(n)
            # instead of O(n^2) -- final order doesn't matter since every
            # consumer (sort_hypotheses/prune_hypotheses, and the final sort
            # below) sorts by score before use, never relies on list order.
            # This mattered because is_equivalent rebuilds two edge sets
            # every call: even a new_hypotheses list that stayed within
            # max_ruleset_hypotheses (capped only during generation, above)
            # could spend far longer than a parse's whole max_parse_seconds
            # budget on this one loop alone -- the deadline check below is
            # kept as a backstop for whatever's still O(n) here.
            dedup_index: dict = {}
            for hyp in new_hypotheses:
                if (config.max_parse_seconds is not None
                        and time.monotonic() - start_time > config.max_parse_seconds):
                    raise ParseResourceExceeded(
                        f"parse exceeded {config.max_parse_seconds}s wall-clock cap "
                        f"(mid-deduplication, ruleset {ruleset_name!r})",
                        ruleset_name, len(new_hypotheses),
                    )
                key = hyp.equivalence_key()
                existing = dedup_index.get(key)
                if existing is None or hyp.score > existing.score:
                    dedup_index[key] = hyp
            deduplicated = list(dedup_index.values())

            # Update chart hypotheses
            chart.hypotheses = deduplicated

            # Prune if configured to score continuously
            if chart.config.score_continuously:
                chart.prune_hypotheses()

        # Final sort by score
        chart.sort_hypotheses()

        # Filter for complete parses only (exactly 1 unconsumed root node)
        complete_hypotheses = [h for h in chart.hypotheses if len(h.get_unconsumed()) == 1]
        if complete_hypotheses:
            chart.hypotheses = complete_hypotheses

        return chart


def apply_rule(hypothesis: Hypothesis, match: Match) -> Hypothesis:
    """
    Apply a rule match to create a new hypothesis.

    Args:
        hypothesis: Original hypothesis
        match: Successful rule match

    Returns:
        New hypothesis with rule applied
    """
    # Create a copy of the hypothesis
    new_hyp = hypothesis.copy()

    # Transform anchor node type
    anchor = new_hyp.nodes[match.anchor_idx]
    anchor.type = match.rule.result

    # Pull categories if specified
    if match.rule.pull_categories:
        for subcat in match.rule.pull_categories:
            # Find a child with this subcategory value
            # (Simplified: just pull from first child)
            for child_idx in match.before_indices + match.after_indices:
                child = new_hyp.nodes[child_idx]
                value = child.get_subcategory_value(subcat)
                if value and value not in anchor.flags:
                    anchor.flags.append(value)
                    break

    # Pop categories if specified
    if match.rule.pop_categories:
        for subcat in match.rule.pop_categories:
            # Remove all flags of this subcategory
            from .enums import SUBTYPE_TO_SUBCAT
            anchor.flags = [
                flag for flag in anchor.flags
                if SUBTYPE_TO_SUBCAT.get(flag) != subcat
            ]

    # Push subtypes onto result node if specified
    if match.rule.push_subtypes:
        for subtype in match.rule.push_subtypes:
            if subtype not in anchor.flags:
                anchor.flags.append(subtype)

    # Create connections
    for conn_spec in match.rule.connections:
        # Resolve node references
        from_indices = resolve_reference(conn_spec.from_ref, match)
        to_indices = resolve_reference(conn_spec.to_ref, match)

        # Create edges for all combinations
        for from_idx in from_indices:
            for to_idx in to_indices:
                edge = Edge(
                    type=conn_spec.type,
                    parent=from_idx,
                    child=to_idx,
                    source_rule=match.rule.note
                )
                new_hyp.add_edge(edge)

    # Mark nodes as consumed
    if "before" in match.rule.consume:
        for idx in match.before_indices:
            new_hyp.consume(idx)

    if "after" in match.rule.consume:
        for idx in match.after_indices:
            new_hyp.consume(idx)

    if "anchor" in match.rule.consume:
        new_hyp.consume(match.anchor_idx)

    return new_hyp


def resolve_reference(ref: str, match: Match) -> List[int]:
    """
    Resolve a node reference to list of indices.

    Args:
        ref: Reference string ("anchor", "before[0]", "after[*]", etc.)
        match: Rule match containing indices

    Returns:
        List of node indices
    """
    if ref == "anchor":
        return [match.anchor_idx]

    elif ref.startswith("before["):
        # Extract index
        idx_str = ref[7:-1]  # Remove "before[" and "]"

        if idx_str == "*":
            # All before elements
            return match.before_indices
        else:
            # Specific index
            idx = int(idx_str)
            if 0 <= idx < len(match.before_indices):
                return [match.before_indices[idx]]
            else:
                return []

    elif ref.startswith("after["):
        # Extract index
        idx_str = ref[6:-1]  # Remove "after[" and "]"

        if idx_str == "*":
            # All after elements
            return match.after_indices
        else:
            # Specific index
            idx = int(idx_str)
            if 0 <= idx < len(match.after_indices):
                return [match.after_indices[idx]]
            else:
                return []

    else:
        raise ValueError(f"Invalid node reference: {ref}")


def apply_ruleset_recursively(hypothesis: Hypothesis, ruleset,
                               deadline: Optional[float] = None,
                               ruleset_name: str = "") -> Hypothesis:
    """
    Keep applying a ruleset until no more matches are found.

    Args:
        hypothesis: Starting hypothesis
        ruleset: Ruleset to apply recursively
        deadline: opt-in wall-clock cap (``time.monotonic()`` timestamp,
            ``None``/disabled by default -- every caller not passing this
            gets byte-identical behavior). Checked once per outer iteration
            (up to ``max_iterations`` of them, each itself doing up to
            ``len(unconsumed) * len(ruleset.rules)`` ``find_matches`` calls)
            because THIS loop, not just the Cartesian-product branching in
            :meth:`QuantumParser.parse`, measured as the actual site of a
            real corpus sentence running well past its parse deadline with
            no ambiguous branching involved at all -- a long/complex
            hypothesis's ``find_matches`` cost across many iterations, not
            any combinatorial blowup.
        ruleset_name: only used in the ``ParseResourceExceeded`` message.

    Returns:
        Hypothesis after exhaustive rule application

    Raises:
        ParseResourceExceeded: only possible when ``deadline`` is given.
    """
    max_iterations = 100  # Prevent infinite loops
    iterations = 0

    current_hyp = hypothesis

    while iterations < max_iterations:
        iterations += 1
        if deadline is not None and time.monotonic() > deadline:
            raise ParseResourceExceeded(
                f"parse exceeded wall-clock cap inside apply_ruleset_recursively "
                f"(ruleset {ruleset_name!r}, iteration {iterations})",
                ruleset_name, iterations,
            )
        matched = False

        # Try to find a match
        for unconsumed_idx in current_hyp.get_unconsumed():
            for rule in ruleset.rules:
                matches = find_matches(current_hyp, unconsumed_idx, rule)

                if matches:
                    # Apply first match
                    current_hyp = apply_rule(current_hyp, matches[0])
                    matched = True
                    break

            if matched:
                break

        if not matched:
            # No more matches, done
            break

    return current_hyp
