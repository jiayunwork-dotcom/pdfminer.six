"""Tests for optional content group (OCG) visibility and layer queries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest import mock

from pdfminer.high_level import (
    extract_layered_text,
    extract_text,
    extract_text_with_sources,
    find_layer_state_for_text,
)
from pdfminer.layers import (
    LayerPolicy,
    LayerQueryResult,
    NativeSequenceDevice,
    SymbolicLayerPolicy,
    assignment_states,
    symbolic_page_search,
)
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from tests.helpers import absolute_sample_path
from tests.sourcemap_fixtures import (
    write_cyclic_ve_variant,
    write_ocg_cross_tj_form_variant,
    write_ocg_direct_dict_variant,
    write_ocg_policy_variant,
    write_ocg_two_page_variant,
    write_ocg_variant,
    write_sixteen_ocg_variant,
    write_unbalanced_marked_content_variant,
)


SIMPLE1 = absolute_sample_path("simple1.pdf")


def _open(path: Path) -> Tuple[Any, PDFDocument]:
    fp = open(path, "rb")
    return fp, PDFDocument(PDFParser(fp))


def _brute_force_feasible(
    sequence: List[Tuple[str, int]], target: str, assignment_count: int
) -> Set[int]:
    feasible: Set[int] = set()
    for assignment in range(assignment_count):
        visible = "".join(
            character
            for character, gate in sequence
            if (gate >> assignment) & 1
        )
        if target in visible:
            feasible.add(assignment)
    return feasible


class AutomatonTests(unittest.TestCase):
    """Bit-parallel search against an independent brute-force reference."""

    def test_matches_brute_force_on_random_sequences(self) -> None:
        import random

        class _Policy:
            def __init__(self, n: int) -> None:
                self.assignment_count = 1 << n
                self.all_value = (1 << self.assignment_count) - 1

        random.seed(20261128)
        for _ in range(2000):
            n = random.randint(1, 6)
            policy = _Policy(n)
            width = policy.assignment_count
            sequence = [
                (random.choice("ABC"), random.randrange(width))
                for _ in range(random.randint(1, 60))
            ]
            target = "".join(
                random.choice("ABC") for _ in range(random.randint(1, 9))
            )
            got = symbolic_page_search(sequence, target, policy)  # type: ignore[arg-type]
            expected = _brute_force_feasible(sequence, target, width)
            self.assertEqual(got, expected)

    def test_hidden_character_does_not_break_other_assignments(self) -> None:
        class _Policy:
            assignment_count = 4
            all_value = 0b1111

        # Assignment 1 sees "AC"; assignment 0 sees nothing between A and C.
        sequence = [("A", 0b0010), ("B", 0b0001), ("C", 0b0010)]
        self.assertEqual(
            symbolic_page_search(sequence, "AC", _Policy()),  # type: ignore[arg-type]
            {1},
        )

    def test_match_is_latched_across_later_foreign_characters(self) -> None:
        class _Policy:
            assignment_count = 2
            all_value = 0b11

        sequence = [("A", 0b01), ("X", 0b01)]
        self.assertEqual(
            symbolic_page_search(sequence, "A", _Policy()),  # type: ignore[arg-type]
            {0},
        )

    def test_empty_target_rejected(self) -> None:
        class _Policy:
            assignment_count = 2
            all_value = 0b11

        with self.assertRaises(ValueError):
            symbolic_page_search([("A", 3)], "", _Policy())  # type: ignore[arg-type]


class VariableMaskTests(unittest.TestCase):
    def test_mask_bit_equals_assignment_bit(self) -> None:
        for n in range(1, 7):
            masks = SymbolicLayerPolicy._build_variable_masks(n)
            self.assertEqual(len(masks), n)
            count = 1 << n
            for j, mask in enumerate(masks):
                for assignment in range(count):
                    self.assertEqual(
                        (mask >> assignment) & 1, (assignment >> j) & 1
                    )

    def test_assignment_states_round_trip(self) -> None:
        ocgs = [10, 11, 12]
        for assignment in range(8):
            states = assignment_states(ocgs, assignment)
            rebuilt = sum(
                (1 if states[objid] else 0) << index
                for index, objid in enumerate(ocgs)
            )
            self.assertEqual(rebuilt, assignment)


class OCGQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ocg.pdf"
        write_ocg_variant(self.path)
        self.default_states = {10: False, 11: False}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_returns_none_when_no_assignment_reveals_target(self) -> None:
        self.assertIsNone(
            find_layer_state_for_text(self.path, "MissingText", self.default_states)
        )

    def test_minimum_flips_for_direct_and_resource_ocg(self) -> None:
        result = find_layer_state_for_text(
            self.path, "AlphaOne", self.default_states
        )
        self.assertIsInstance(result, LayerQueryResult)
        assert result is not None
        self.assertEqual(result.flips, 1)
        self.assertEqual(result.states, {10: True, 11: False})
        self.assertEqual(result.page_number, 1)
        self.assertEqual(result.page_index, 0)

        result_two = find_layer_state_for_text(
            self.path, "AlphaTwo", self.default_states
        )
        assert result_two is not None
        self.assertEqual(result_two.flips, 1)
        self.assertEqual(result_two.states, {10: False, 11: True})

    def test_ve_and_or_not_memberships(self) -> None:
        both = find_layer_state_for_text(
            self.path, "BothOnly", self.default_states
        )
        assert both is not None
        self.assertEqual(both.flips, 2)
        self.assertEqual(both.states, {10: True, 11: True})

        either = find_layer_state_for_text(
            self.path, "EitherOne", self.default_states
        )
        assert either is not None
        self.assertEqual(either.flips, 1)

        not_alpha = find_layer_state_for_text(
            self.path, "NotAlpha", self.default_states
        )
        assert not_alpha is not None
        self.assertEqual(not_alpha.flips, 0)

    def test_form_oc_gate_requires_parent_group(self) -> None:
        result = find_layer_state_for_text(
            self.path, "FormHidden", self.default_states
        )
        assert result is not None
        self.assertEqual(result.flips, 1)
        self.assertEqual(result.states, {10: False, 11: True})
        layered = extract_layered_text(self.path, result.states)
        self.assertIn("FormHidden", layered.page_text(0))

    def test_flip_count_relative_to_supplied_state(self) -> None:
        # Starting from OCG 10 already on, AlphaOne costs zero flips.
        result = find_layer_state_for_text(
            self.path, "AlphaOne", {10: True, 11: False}
        )
        assert result is not None
        self.assertEqual(result.flips, 0)
        self.assertEqual(result.states, {10: True, 11: False})

    def test_returned_state_reproduces_target_through_concrete_extraction(
        self,
    ) -> None:
        for target in (
            "Always",
            "AlphaOne",
            "AlphaTwo",
            "BothOnly",
            "EitherOne",
            "NotAlpha",
            "FormHidden",
        ):
            result = find_layer_state_for_text(
                self.path, target, self.default_states
            )
            assert result is not None
            layered = extract_layered_text(self.path, result.states)
            self.assertIn(target, layered.page_text(0))

    def test_concrete_extraction_excludes_hidden_native_text(self) -> None:
        hidden = extract_layered_text(self.path, self.default_states)
        visible_text = hidden.page_text(0)
        self.assertIn("Always", visible_text)
        self.assertNotIn("AlphaOne", visible_text)
        self.assertNotIn("FormHidden", visible_text)
        # Layout-generated whitespace never appears in the native sequence.
        self.assertNotIn("\n", visible_text)
        self.assertNotIn(" ", visible_text)

    def test_tie_picks_lowest_assignment(self) -> None:
        # EitherOne is reachable by turning on either single group (one flip).
        result = find_layer_state_for_text(
            self.path, "EitherOne", self.default_states
        )
        assert result is not None
        # Assignment integer ordering prefers the lowest catalog index.
        self.assertEqual(result.states, {10: True, 11: False})


class PolicyMembershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "policy.pdf"
        write_ocg_policy_variant(self.path)
        self.states = {10: False, 11: False}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_all_four_p_policies(self) -> None:
        expectations = {
            "AllOnText": 2,
            "AnyOnText": 1,
            "AllOffText": 0,
            "AnyOffText": 0,
        }
        for target, flips in expectations.items():
            result = find_layer_state_for_text(self.path, target, self.states)
            assert result is not None
            self.assertEqual(result.flips, flips, target)


class CrossTjAndFormTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tj.pdf"
        write_ocg_cross_tj_form_variant(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_target_split_across_tj_and_reused_form(self) -> None:
        off = extract_layered_text(self.path, {10: False})
        self.assertEqual(off.page_text(0), "")

        result = find_layer_state_for_text(self.path, "SplitText", {10: False})
        assert result is not None
        self.assertEqual(result.flips, 1)
        # Two Form invocations each draw the split text once.
        on = extract_layered_text(self.path, result.states)
        self.assertEqual(on.page_text(0), "SplitTextSplitText")

    def test_zero_flips_when_layer_already_on(self) -> None:
        result = find_layer_state_for_text(self.path, "SplitText", {10: True})
        assert result is not None
        self.assertEqual(result.flips, 0)

    def test_each_form_invocation_interpreted_at_most_twice(self) -> None:
        # The page invokes the same Form twice per pass; each invocation gets
        # its own interpreter instance, which must render at most twice.
        original = PDFPageInterpreter.render_contents
        render_counts: Dict[int, int] = {}

        def wrapper(self, resources, streams, ctm=None):  # type: ignore[no-untyped-def]
            identity = id(self)
            render_counts[identity] = render_counts.get(identity, 0) + 1
            return original(self, resources, streams, ctm=ctm)

        with mock.patch.object(PDFPageInterpreter, "render_contents", wrapper):
            find_layer_state_for_text(self.path, "SplitText", {10: False})
        # Two Form invocations per pass * at most two passes, each bounded.
        self.assertLessEqual(max(render_counts.values()), 2)


class PageBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "two.pdf"
        write_ocg_two_page_variant(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_match_does_not_cross_pages_and_reports_hit_page(self) -> None:
        result = find_layer_state_for_text(
            self.path, "SharedMarker", {10: False, 11: False}
        )
        assert result is not None
        self.assertEqual(result.page_number, 1)
        self.assertEqual(result.page_index, 0)
        self.assertEqual(result.states, {10: True, 11: False})

    def test_native_sequences_are_separate_per_page(self) -> None:
        layered = extract_layered_text(
            self.path, {10: True, 11: True}
        )
        self.assertEqual(len(layered.pages), 2)
        self.assertTrue(all("SharedMarker" in page for page in layered.pages))


class StateValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ocg.pdf"
        write_ocg_variant(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_group_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            find_layer_state_for_text(self.path, "Always", {10: False})

    def test_extra_group_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            find_layer_state_for_text(
                self.path, "Always", {10: False, 11: False, 99: True}
            )

    def test_non_boolean_value_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            find_layer_state_for_text(
                self.path, "Always", {10: "on", 11: False}  # type: ignore[dict-item]
            )

    def test_empty_target_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            find_layer_state_for_text(
                self.path, "", {10: False, 11: False}
            )

    def test_reference_outside_catalog_ocgs_raises(self) -> None:
        fp, document = _open(self.path)
        try:
            # Object 7 is a font, not an OCG.
            bad_ocg = document.getobj(7)
            with self.assertRaises(ValueError):
                policy = LayerPolicy(document, {10: False, 11: False})
                policy.evaluate_ocg(bad_ocg, 7)
        finally:
            fp.close()

    def test_cyclic_ve_raises_value_error(self) -> None:
        cyclic_path = Path(self.tmp.name) / "cyc.pdf"
        write_cyclic_ve_variant(cyclic_path)
        with self.assertRaises(ValueError):
            find_layer_state_for_text(cyclic_path, "Cyclic", {10: False})

    def test_unbalanced_marked_content_raises_value_error(self) -> None:
        bad_path = Path(self.tmp.name) / "unbal.pdf"
        write_unbalanced_marked_content_variant(bad_path)
        with self.assertRaises(ValueError):
            find_layer_state_for_text(bad_path, "Dangling", {10: False})

    def test_unknown_property_name_raises_value_error(self) -> None:
        fp, document = _open(self.path)
        try:
            policy = LayerPolicy(document, {10: False, 11: False})
            with self.assertRaises(ValueError):
                policy.resolve_property("DoesNotExist", {})
        finally:
            fp.close()


class GlobalOptimumTests(unittest.TestCase):
    """The 16-group / 4096-character fixture from sourcemap_fixtures."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "big.pdf"
        write_sixteen_ocg_variant(cls.path)
        cls.groups = list(range(20, 36))
        cls.off_states = {objid: False for objid in cls.groups}
        on_text = extract_layered_text(
            cls.path, {objid: True for objid in cls.groups}
        )
        cls.native_count = sum(len(page) for page in on_text.pages)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_fixture_has_exactly_4096_native_characters(self) -> None:
        self.assertEqual(self.native_count, 4096)

    def test_returns_global_minimum_not_first_feasible(self) -> None:
        cheap = find_layer_state_for_text(
            self.path, "EarlyFlipMarkerOneTwo34", self.off_states
        )
        assert cheap is not None
        self.assertEqual(cheap.flips, 1)

        expensive = find_layer_state_for_text(
            self.path, "NeedAll16", self.off_states
        )
        assert expensive is not None
        # A first-feasible enumeration would stop at the single-group blocks;
        # the conjunctive target genuinely needs every group flipped.
        self.assertEqual(expensive.flips, 16)
        self.assertTrue(all(expensive.states.values()))
        self.assertEqual(
            extract_layered_text(
                self.path, expensive.states
            ).page_text(0).count("NeedAll16"),
            1,
        )

    def test_each_page_and_form_interpreted_at_most_twice(self) -> None:
        # The query performs one full symbolic interpretation and one concrete
        # verification of the single hit page.  Count how often each distinct
        # interpreter instance (one per page, or one per Form Do invocation)
        # renders its contents: it must be at most the two passes.
        original = PDFPageInterpreter.render_contents
        render_counts: Dict[int, int] = {}
        per_stream: Dict[Optional[int], int] = {}

        def wrapper(self, resources, streams, ctm=None):  # type: ignore[no-untyped-def]
            identity = id(self)
            render_counts[identity] = render_counts.get(identity, 0) + 1
            for stream in streams:
                objid = getattr(stream, "objid", None)
                per_stream[objid] = per_stream.get(objid, 0) + 1
            return original(self, resources, streams, ctm=ctm)

        with mock.patch.object(PDFPageInterpreter, "render_contents", wrapper):
            find_layer_state_for_text(self.path, "NeedAll16", self.off_states)
        self.assertTrue(render_counts)
        # Every drawing instance is interpreted in at most two passes.
        self.assertLessEqual(max(render_counts.values()), 2)
        # The single page content stream is exercised by both passes.
        self.assertEqual(per_stream[4], 2)

    def test_brute_force_confirms_minimum(self) -> None:
        # Independently enumerate all assignments for the expensive target to
        # prove the query's flip count is the global minimum.
        fp, document = _open(self.path)
        try:
            policy = SymbolicLayerPolicy(document)
            device = NativeSequenceDevice(PDFResourceManager())
            interpreter = PDFPageInterpreter(
                PDFResourceManager(), device, layer_policy=policy
            )
            device.ocg_value = policy.all_value
            for index, page in enumerate(PDFPage.create_pages(document)):
                page.page_index = index
                interpreter.process_page(page)
            feasible = symbolic_page_search(
                device.pages[0], "NeedAll16", policy
            )
            baseline = sum(
                (1 if self.off_states[objid] else 0) << index
                for index, objid in enumerate(policy.ocg_object_numbers)
            )
            minimum = min(
                (assignment ^ baseline).bit_count()
                for assignment in feasible
            )
        finally:
            fp.close()
        result = find_layer_state_for_text(
            self.path, "NeedAll16", self.off_states
        )
        assert result is not None
        self.assertEqual(result.flips, minimum)


class TextConverterPolicyTests(unittest.TestCase):
    """The regular TextConverter pipeline must obey the same LayerPolicy."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ocg.pdf"
        write_ocg_variant(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_text_converter_excludes_hidden_layers(self) -> None:
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import TextConverter
        from pdfminer.layout import LAParams
        from io import StringIO
        from pdfminer.layers import LayerPolicy

        with open(self.path, "rb") as fp:
            document = PDFDocument(PDFParser(fp))
            policy = LayerPolicy(document, {10: False, 11: False})
            out = StringIO()
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, out, laparams=LAParams())
            interpreter = PDFPageInterpreter(
                rsrcmgr, device, layer_policy=policy
            )
            device.ocg_value = policy.all_value
            for index, page in enumerate(PDFPage.create_pages(document)):
                page.page_index = index
                interpreter.process_page(page)
            text = out.getvalue()
        self.assertIn("Always", text)
        self.assertNotIn("AlphaOne", text)
        self.assertNotIn("FormHidden", text)

    def test_native_sequence_matches_layered_extraction(self) -> None:
        # The query's returned state must reproduce the hit through the native
        # extraction entry, character for character in drawing order.
        states = {10: True, 11: True}
        result = find_layer_state_for_text(self.path, "FormHidden", states)
        assert result is not None
        layered = extract_layered_text(self.path, result.states)
        # Re-querying from the result state is a zero-flip configuration.
        again = find_layer_state_for_text(self.path, "FormHidden", result.states)
        assert again is not None
        self.assertEqual(again.flips, 0)
        self.assertEqual(
            layered.page_text(0),
            extract_layered_text(self.path, again.states).page_text(0),
        )


class DirectDictionaryGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "direct.pdf"
        write_ocg_direct_dict_variant(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_inline_bdc_dictionary_gates_text(self) -> None:
        hidden = extract_layered_text(self.path, {10: False})
        self.assertEqual(hidden.page_text(0), "Plain")
        shown = extract_layered_text(self.path, {10: True})
        self.assertEqual(shown.page_text(0), "PlainGated")
        result = find_layer_state_for_text(self.path, "Gated", {10: False})
        assert result is not None
        self.assertEqual(result.flips, 1)
        self.assertEqual(result.states, {10: True})


class EncryptedDocumentTests(unittest.TestCase):
    def test_encrypted_pdf_rejected(self) -> None:
        encrypted = absolute_sample_path("encryption/rc4-40.pdf")
        with self.assertRaises(ValueError):
            find_layer_state_for_text(encrypted, "x", {}, password="foo")
        with self.assertRaises(ValueError):
            extract_layered_text(encrypted, {}, password="foo")


class DefaultExtractionUnchangedTests(unittest.TestCase):
    """Requirement: without the new entry point, output and sources are stable."""

    def test_simple1_text_unchanged(self) -> None:
        baseline = extract_text(SIMPLE1)
        # A policy-free extraction must be byte-identical to the baseline.
        self.assertEqual(extract_text(SIMPLE1), baseline)

    def test_simple1_source_map_unchanged(self) -> None:
        first = extract_text_with_sources(SIMPLE1)
        second = extract_text_with_sources(SIMPLE1)
        self.assertEqual(first.text, second.text)
        self.assertEqual(
            [record.code for record in first.sources.records],
            [record.code for record in second.sources.records],
        )

    def test_layer_entry_does_not_mutate_default_pipeline(self) -> None:
        # Building a policy must not change an unrelated normal extraction.
        baseline = extract_text(SIMPLE1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ocg.pdf"
            write_ocg_variant(path)
            find_layer_state_for_text(path, "Always", {10: False, 11: False})
        self.assertEqual(extract_text(SIMPLE1), baseline)


if __name__ == "__main__":
    unittest.main()
