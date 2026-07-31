"""Tests for ExplicationStore and its integration with NSMMeaningResolver.

Uses a tiny inline fixture store — no HuggingFace download required for CI.

PROVENANCE NOTE: The "snake" explication below is an illustrative example
derived from the DeepNSM paper description (arxiv 2505.11764). In real use,
explications come from the AI-GENERATED DeepNSM corpus (Gemini-2.0-Flash);
they are always labeled ``ai_generated`` in provenance.
"""

from __future__ import annotations

import pytest

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.explications import ExplicationStore
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.thought import meaning_prime_ids


# ---------------------------------------------------------------------------
# Inline fixture data (do NOT download the real dataset in these tests)
# ---------------------------------------------------------------------------

# The canonical DeepNSM example explication for "snake" (from arxiv 2505.11764)
_SNAKE_EXPLICATION = (
    "creatures / there are many kinds of such creatures / "
    "the bodies of creatures of this kind are long / "
    "their bodies touch the ground at all times / "
    "when these creatures move they do this with their bodies / "
    "these creatures can do bad things to people / "
    "when this happens something bad can happen to these people"
)

_RIVER_EXPLICATION = (
    "something / water moves through this place / "
    "it moves for a long time / people can see this"
)


def _make_fixture_store() -> ExplicationStore:
    """Build a tiny ExplicationStore from hard-coded rows (no I/O)."""
    store = ExplicationStore()
    # Inject DeepNSM rows directly (bypasses file I/O for CI)
    store._deepnsm["snake"] = {
        "explication": _SNAKE_EXPLICATION,
        "provenance": "ai_generated:DeepNSM hand-refined (Gemini, cc-by-nc-sa-4.0)",
        "score": 0.95,
    }
    store._deepnsm["river"] = {
        "explication": _RIVER_EXPLICATION,
        "provenance": "ai_generated:DeepNSM (Gemini, cc-by-nc-sa-4.0)",
        "score": 0.80,
    }
    return store


def _make_gold_store() -> ExplicationStore:
    """Build a store where 'snake' has a gold entry (overrides DeepNSM)."""
    store = _make_fixture_store()
    store._gold["snake"] = {
        "explication": "a creature / this creature has a long body",
        "provenance": "gold:literature (Wierzbicka 1996)",
        "score": None,
    }
    return store


# ---------------------------------------------------------------------------
# 1. Basic get() and provenance
# ---------------------------------------------------------------------------

class TestExplicationStoreGet:
    def test_get_snake_returns_entry(self):
        store = _make_fixture_store()
        result = store.get("snake")
        assert result is not None

    def test_get_snake_has_ai_generated_provenance(self):
        store = _make_fixture_store()
        result = store.get("snake")
        assert result["provenance"].startswith("ai_generated")

    def test_get_snake_hand_refined_provenance(self):
        store = _make_fixture_store()
        result = store.get("snake")
        assert "hand-refined" in result["provenance"]

    def test_get_snake_has_cc_attribution(self):
        store = _make_fixture_store()
        result = store.get("snake")
        assert "cc-by-nc-sa" in result["provenance"]

    def test_get_river_ai_generated_not_hand_refined(self):
        store = _make_fixture_store()
        result = store.get("river")
        assert result["provenance"].startswith("ai_generated")
        assert "hand-refined" not in result["provenance"]

    def test_get_unknown_returns_none(self):
        store = _make_fixture_store()
        assert store.get("zxqwerty_unknown") is None

    def test_get_case_insensitive(self):
        store = _make_fixture_store()
        assert store.get("SNAKE") is not None
        assert store.get("Snake") is not None

    def test_len_reflects_entries(self):
        store = _make_fixture_store()
        # 2 DeepNSM + 0 gold
        assert len(store) == 2


# ---------------------------------------------------------------------------
# 2. Gold precedence
# ---------------------------------------------------------------------------

class TestGoldPrecedence:
    def test_gold_entry_overrides_deepnsm(self):
        store = _make_gold_store()
        result = store.get("snake")
        assert result is not None
        assert result["provenance"].startswith("gold:literature")

    def test_gold_provenance_contains_source(self):
        store = _make_gold_store()
        result = store.get("snake")
        assert "Wierzbicka" in result["provenance"]

    def test_deepnsm_still_accessible_for_non_gold_word(self):
        store = _make_gold_store()
        result = store.get("river")
        assert result is not None
        assert result["provenance"].startswith("ai_generated")

    def test_gold_is_never_labeled_ai_generated(self):
        store = _make_gold_store()
        result = store.get("snake")
        assert "ai_generated" not in result["provenance"]


# ---------------------------------------------------------------------------
# 3. explication_to_tree — structure checks
# ---------------------------------------------------------------------------

class TestExplicationToTree:
    def test_returns_parse_tree(self):
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        assert isinstance(tree, ParseTree)

    def test_root_label_is_explication(self):
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        assert tree.root.label == "EXPLICATION"

    def test_tree_has_children(self):
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        assert len(tree.root.children) > 0

    def test_tree_contains_kind_prime(self):
        """The snake explication contains 'kind' -> KIND prime."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = {n.label for n in tree.iter_preorder()}
        assert "KIND" in labels

    def test_tree_contains_people_prime(self):
        """The snake explication contains 'people' -> PEOPLE prime."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = {n.label for n in tree.iter_preorder()}
        assert "PEOPLE" in labels

    def test_tree_contains_touch_prime(self):
        """The snake explication contains 'touch' -> TOUCH prime."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = {n.label for n in tree.iter_preorder()}
        assert "TOUCH" in labels

    def test_tree_contains_long_molecule(self):
        """The snake explication contains 'long' -> LONG molecule."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = {n.label for n in tree.iter_preorder()}
        assert "LONG" in labels

    def test_tree_contains_ground_molecule(self):
        """The snake explication contains 'ground' -> GROUND molecule."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = {n.label for n in tree.iter_preorder()}
        assert "GROUND" in labels

    def test_tree_contains_do_prime(self):
        """The snake explication contains 'do' -> DO prime."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = {n.label for n in tree.iter_preorder()}
        assert "DO" in labels

    def test_tree_contains_bad_prime(self):
        """The snake explication contains 'bad' -> BAD prime."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = {n.label for n in tree.iter_preorder()}
        assert "BAD" in labels

    def test_nodes_are_deduplicated(self):
        """Duplicate prime/molecule tokens should appear only once in the tree."""
        store = _make_fixture_store()
        tree = store.explication_to_tree(_SNAKE_EXPLICATION)
        labels = [n.label for n in tree.root.children]
        assert len(labels) == len(set(labels)), "duplicate labels in children"

    def test_empty_text_gives_root_only(self):
        store = _make_fixture_store()
        tree = store.explication_to_tree("")
        assert tree.root.label == "EXPLICATION"
        assert tree.root.children == []


# ---------------------------------------------------------------------------
# 4. meaning_prime_ids on the snake explication tree
# ---------------------------------------------------------------------------

class TestMeaningPrimeIds:
    def test_snake_meaning_prime_ids_nontrivial(self):
        """meaning_prime_ids for 'snake' must be non-trivial (not just SOMETHING)."""
        store = _make_fixture_store()
        entry = store.get("snake")
        tree = store.explication_to_tree(entry["explication"])
        ids = meaning_prime_ids(tree)
        assert len(ids) >= 1, f"Expected >=1 prime ids, got {ids!r}"

    def test_snake_meaning_prime_ids_contains_known_primes(self):
        """The prime ids must include known primes from the snake explication."""
        from nsm_ct.thought import meaning_prime_id
        store = _make_fixture_store()
        entry = store.get("snake")
        tree = store.explication_to_tree(entry["explication"])
        ids = meaning_prime_ids(tree)
        # KIND, DO, BAD, TOUCH, PEOPLE are confirmed in the snake explication
        # (capped at MAX_MEANING_PRIMES=4)
        kind_id = meaning_prime_id("KIND")
        do_id = meaning_prime_id("DO")
        bad_id = meaning_prime_id("BAD")
        touch_id = meaning_prime_id("TOUCH")
        assert any(i in ids for i in [kind_id, do_id, bad_id, touch_id]), (
            f"Expected at least one of KIND/DO/BAD/TOUCH in {ids!r}"
        )

    def test_snake_is_not_just_something(self):
        """The snake tree should NOT resolve to just [SOMETHING_id]."""
        from nsm_ct.thought import meaning_prime_id
        store = _make_fixture_store()
        entry = store.get("snake")
        tree = store.explication_to_tree(entry["explication"])
        ids = meaning_prime_ids(tree)
        something_id = meaning_prime_id("SOMETHING")
        # Either not just SOMETHING, or multiple primes present
        assert ids != [something_id], "Snake meaning should not reduce to just SOMETHING"


# ---------------------------------------------------------------------------
# 5. Integration with NSMMeaningResolver
# ---------------------------------------------------------------------------

class TestResolverIntegration:
    def test_resolver_with_store_returns_explication_tree_for_snake(self):
        """Resolver wired with fixture store returns explication tree for 'snake'."""
        store = _make_fixture_store()
        resolver = NSMMeaningResolver(explication_store=store)
        tree = resolver.resolve("snake")
        assert tree is not None
        # The root should be EXPLICATION (from explication_to_tree)
        assert tree.root.label == "EXPLICATION"

    def test_resolver_provenance_is_ai_generated(self):
        """tree.text carries the ai_generated provenance string."""
        store = _make_fixture_store()
        resolver = NSMMeaningResolver(explication_store=store)
        tree = resolver.resolve("snake")
        assert tree.text.startswith("ai_generated"), (
            f"Expected ai_generated provenance, got {tree.text!r}"
        )

    def test_resolver_snake_has_grounded_prime_ids(self):
        """meaning_prime_ids from resolver for 'snake' is grounded (not trivially SOMETHING)."""
        from nsm_ct.thought import meaning_prime_id
        store = _make_fixture_store()
        resolver = NSMMeaningResolver(explication_store=store)
        tree = resolver.resolve("snake")
        ids = meaning_prime_ids(tree)
        assert len(ids) >= 1
        something_id = meaning_prime_id("SOMETHING")
        assert ids != [something_id], f"'snake' should not resolve to just SOMETHING; got {ids!r}"

    def test_resolver_gold_takes_precedence_over_deepnsm(self):
        """When gold entry exists, resolver uses it and provenance is gold:literature."""
        store = _make_gold_store()
        resolver = NSMMeaningResolver(explication_store=store)
        tree = resolver.resolve("snake")
        assert tree is not None
        assert tree.text.startswith("gold:literature"), (
            f"Expected gold:literature provenance, got {tree.text!r}"
        )

    def test_resolver_falls_through_to_wordnet_for_unknown(self):
        """Word not in explication store still resolves via WordNet / fallback."""
        store = _make_fixture_store()
        resolver = NSMMeaningResolver(explication_store=store)
        # "zzzfoo_absent" is not in the fixture store -> falls through
        tree = resolver.resolve("zzzfoo_absent")
        assert tree is not None
        # fallback is SOMETHING
        assert tree.root.label == "SOMETHING"

    def test_resolver_prime_words_not_intercepted_by_store(self):
        """Words that are NSM primes should still be resolved as primes, not via store."""
        store = _make_fixture_store()
        # Inject a fake explication for "think" (a prime) — resolver should NOT use it
        store._deepnsm["think"] = {
            "explication": "something happens in someone",
            "provenance": "ai_generated:DeepNSM (Gemini, cc-by-nc-sa-4.0)",
            "score": 0.5,
        }
        resolver = NSMMeaningResolver(explication_store=store)
        tree = resolver.resolve("think")
        # Prime resolution wins over explication store
        assert tree.root.label == "THINK"

    def test_resolver_molecule_words_not_intercepted_by_store(self):
        """Words that are molecule exponents should still resolve as molecules."""
        store = _make_fixture_store()
        store._deepnsm["water"] = {
            "explication": "something / people drink this",
            "provenance": "ai_generated:DeepNSM (Gemini, cc-by-nc-sa-4.0)",
            "score": 0.5,
        }
        resolver = NSMMeaningResolver(explication_store=store)
        tree = resolver.resolve("water")
        assert tree.root.label == "WATER"

    def test_resolver_empty_store_falls_back_gracefully(self):
        """Resolver with empty store behaves identically to resolver without store."""
        empty_store = ExplicationStore()  # no data loaded
        resolver_with_empty = NSMMeaningResolver(explication_store=empty_store)
        # "zzzfoo_absent" -> SOMETHING (fallback), same as without store
        tree = resolver_with_empty.resolve("zzzfoo_absent")
        assert tree.root.label == "SOMETHING"

    def test_caching_still_works_with_store(self):
        """Resolver with store still caches: second call returns same object."""
        store = _make_fixture_store()
        resolver = NSMMeaningResolver(explication_store=store)
        t1 = resolver.resolve("snake")
        t2 = resolver.resolve("snake")
        assert t1 is t2


# ---------------------------------------------------------------------------
# 6. Graceful loading when data file absent
# ---------------------------------------------------------------------------

class TestGracefulLoad:
    def test_load_missing_file_gives_empty_store(self, tmp_path):
        store = ExplicationStore.load(
            path=str(tmp_path / "nonexistent.jsonl"),
            gold_path=str(tmp_path / "nonexistent_gold.json"),
        )
        assert store.is_empty()

    def test_load_empty_jsonl_gives_empty_store(self, tmp_path):
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("")
        store = ExplicationStore.load(path=str(jsonl))
        assert len(store._deepnsm) == 0

    def test_load_valid_jsonl(self, tmp_path):
        import json
        jsonl = tmp_path / "test.jsonl"
        rows = [
            {"word": "cat", "syn": "cat.n.01", "explication": "creatures / small / people keep them", "score": 0.9, "split": "test"},
            {"word": "dog", "syn": "dog.n.01", "explication": "creatures / they live with people", "score": 0.8, "split": "train"},
        ]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows))
        store = ExplicationStore.load(path=str(jsonl))
        assert store.get("cat") is not None
        assert store.get("dog") is not None
        # cat is from test split -> hand-refined provenance
        assert "hand-refined" in store.get("cat")["provenance"]
        # dog is from train -> non-hand-refined provenance
        assert "hand-refined" not in store.get("dog")["provenance"]

    def test_load_synset_lookup(self, tmp_path):
        import json
        jsonl = tmp_path / "syn.jsonl"
        rows = [
            {"word": "snake", "syn": "snake.n.01", "explication": _SNAKE_EXPLICATION, "score": 0.9, "split": "test"},
        ]
        jsonl.write_text(json.dumps(rows[0]))
        store = ExplicationStore.load(path=str(jsonl))
        # Lookup by synset id
        assert store.get("snake.n.01") is not None

    def test_resolver_with_default_load_when_file_absent(self, tmp_path, monkeypatch):
        """NSMMeaningResolver with default ExplicationStore.load() when JSONL absent
        should degrade gracefully (not crash) and return SOMETHING for unknowns."""
        import nsm_ct.explications as expl_mod
        # Point default paths to non-existent tmp files
        monkeypatch.setattr(expl_mod, "_DEFAULT_JSONL", tmp_path / "absent.jsonl")
        monkeypatch.setattr(expl_mod, "_DEFAULT_GOLD", tmp_path / "absent_gold.json")
        resolver = NSMMeaningResolver()
        tree = resolver.resolve("zzzabsent_word")
        assert tree.root.label == "SOMETHING"

    def test_resolver_water_still_molecule_when_store_absent(self, tmp_path, monkeypatch):
        """Even with no data file, water should resolve as WATER molecule."""
        import nsm_ct.explications as expl_mod
        monkeypatch.setattr(expl_mod, "_DEFAULT_JSONL", tmp_path / "absent.jsonl")
        monkeypatch.setattr(expl_mod, "_DEFAULT_GOLD", tmp_path / "absent_gold.json")
        resolver = NSMMeaningResolver()
        tree = resolver.resolve("water")
        assert tree.root.label == "WATER"
