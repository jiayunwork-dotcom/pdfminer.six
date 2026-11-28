import gc
import shutil
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import List, Optional

from pdfminer.cmapdb import CMap
from pdfminer.converter import PDFPageAggregator, TextConverter
from pdfminer.high_level import (
    extract_pages,
    extract_text,
    extract_text_to_fp,
    extract_text_with_sources,
)
from pdfminer.layout import LAParams, LTChar, LTContainer
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.sourcemap import NativeSource, SourceMap

from tests.helpers import absolute_sample_path
from tests.sourcemap_fixtures import (
    stream_object,
    write_form_reuse_variant,
    write_multi_unicode_variant,
    write_shared_content_variant,
    write_pdf,
)


SIMPLE1 = absolute_sample_path("simple1.pdf")
SIMPLE4 = absolute_sample_path("simple4.pdf")

SIMPLE1_EXPECTED = (
    "Hello \n\nWorld\n\nHello \n\nWorld\n\n"
    "H e l l o  \n\nW o r l d\n\n"
    "H e l l o  \n\nW o r l d\n\n\f"
)


def locator_signature(source: Optional[NativeSource]):
    if source is None:
        return None
    return (
        source.page_index,
        tuple(
            (
                call.name,
                call.object_number,
                call.call_index,
                call.call_order,
            )
            for call in source.path
        ),
        source.stream.content_index,
        source.stream_object_number,
        source.operator_index,
        source.operator,
        source.operand_index,
        source.string_index,
        source.byte_start,
        source.byte_end,
        source.code,
    )


def text_signatures(source_map: SourceMap):
    return [
        (char, locator_signature(source))
        for _, char, source in source_map.characters()
    ]


class DisabledExtractionTests(unittest.TestCase):
    def test_simple1_without_source_map_is_unchanged(self):
        self.assertEqual(extract_text(SIMPLE1), SIMPLE1_EXPECTED)

        source_map = SourceMap()
        enabled = extract_text(SIMPLE1, source_map=source_map)
        self.assertEqual(enabled, SIMPLE1_EXPECTED)
        self.assertEqual(source_map.text, SIMPLE1_EXPECTED)
        self.assertEqual(len(source_map.char_sources), len(enabled))

    def test_extract_text_to_fp_binds_stringio(self):
        out = StringIO()
        source_map = SourceMap()
        with open(SIMPLE1, "rb") as fp:
            extract_text_to_fp(
                fp, out, laparams=LAParams(), source_map=source_map
            )
        self.assertEqual(source_map.text, out.getvalue())
        self.assertEqual(source_map.text, SIMPLE1_EXPECTED)


class Simple1SourceLocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = extract_text_with_sources(SIMPLE1)
        cls.sources = cls.result.sources

    def test_page_and_stream_object(self):
        native = [source for source in self.sources.records]
        self.assertTrue(native)
        self.assertTrue(all(s.page_index == 0 for s in native))
        self.assertTrue(all(s.page_number == 1 for s in native))
        self.assertTrue(all(s.page_object_number == 4 for s in native))
        self.assertTrue(all(s.stream_object_number == 5 for s in native))
        self.assertTrue(all(s.operator_ref.path == () for s in native))

    def test_text_show_operation_numbers_and_operators(self):
        by_operator = {}
        for source in self.sources.records:
            by_operator.setdefault(source.operator_index, []).append(source)

        self.assertEqual(sorted(by_operator), [0, 1, 2, 3])
        self.assertEqual(
            [by_operator[i][0].operator for i in range(4)],
            ["TJ", "Tj", "Tj", "TJ"],
        )

        # First TJ: [(Hello ) -4166 (World)].  Numeric positioning is not an
        # operand with a string index.
        op0 = by_operator[0]
        self.assertEqual(b"".join(s.code for s in op0[:6]), b"Hello ")
        self.assertEqual(
            [(s.operand_index, s.string_index) for s in op0[:6]],
            [(0, 0)] * 6,
        )
        self.assertEqual(
            [(s.byte_start, s.byte_end) for s in op0[:6]],
            [(i, i + 1) for i in range(6)],
        )
        self.assertEqual(b"".join(s.code for s in op0[6:]), b"World")
        self.assertEqual(
            [(s.operand_index, s.string_index) for s in op0[6:]],
            [(2, 1)] * 5,
        )
        self.assertEqual(
            [(s.byte_start, s.byte_end) for s in op0[6:]],
            [(i, i + 1) for i in range(5)],
        )

        # Both Tj operators contain one string operand with contiguous bytes.
        for op_index in (1, 2):
            records = by_operator[op_index]
            self.assertEqual(len(records), len(b"Hello World"))
            self.assertEqual(b"".join(s.code for s in records), b"Hello World")
            self.assertTrue(all(s.operand_index == 0 for s in records))
            self.assertTrue(all(s.string_index == 0 for s in records))
            self.assertEqual(
                [(s.byte_start, s.byte_end) for s in records],
                [(i, i + 1) for i in range(11)],
            )

    def test_adjacent_tj_strings_are_not_confused(self):
        op3 = [s for s in self.sources.records if s.operator_index == 3]
        h_source = next(s for s in op3 if s.code == b"H")
        e_source = next(s for s in op3 if s.code == b"e")

        # Both are one-byte operands starting at offset zero, but they are
        # different operands in the TJ array and must remain distinct.
        self.assertEqual((h_source.byte_start, h_source.byte_end), (0, 1))
        self.assertEqual((e_source.byte_start, e_source.byte_end), (0, 1))
        self.assertNotEqual(h_source.operand_index, e_source.operand_index)
        self.assertNotEqual(h_source.string_index, e_source.string_index)
        self.assertEqual((h_source.operand_index, h_source.string_index), (0, 0))
        self.assertEqual((e_source.operand_index, e_source.string_index), (2, 1))

    def test_output_interval_finds_exact_drawing(self):
        text = self.result.text
        start = text.index("World")
        intervals = self.result.lookup(start, start + len("World"))
        self.assertEqual("".join(i.text for i in intervals), "World")
        self.assertEqual(len(intervals), len("World"))
        for offset, interval in enumerate(intervals):
            self.assertIsNotNone(interval.source)
            source = interval.source
            assert source is not None
            self.assertEqual(source.operator_index, 0)
            self.assertEqual(source.operand_index, 2)
            self.assertEqual(source.string_index, 1)
            self.assertEqual((source.byte_start, source.byte_end), (offset, offset + 1))

    def test_layout_newlines_and_inserted_spaces_have_no_source(self):
        text = self.result.text

        # The double newline after the first line is generated by the line
        # annotation plus the text-box suffix.
        newline_start = text.index("Hello ") + len("Hello ")
        self.assertEqual(text[newline_start : newline_start + 2], "\n\n")
        self.assertIsNone(self.result.source_at(newline_start))
        self.assertIsNone(self.result.source_at(newline_start + 1))

        spaced = text.index("H e l l o")
        self.assertIsNone(self.result.source_at(spaced + 1))
        self.assertEqual(text[spaced + 1], " ")

        self.assertEqual(text[-1], "\f")
        self.assertIsNone(self.result.source_at(len(text) - 1))


class DecodedStringByteOffsetTests(unittest.TestCase):
    def test_offsets_use_unescaped_operand_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            content = stream_object(
                b"BT /F1 24 Tf 100 700 Td (A\\101B) Tj ET"
            )
            font = (
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
            )
            page = (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>"
            )
            path = write_pdf(
                Path(tmp) / "escaped.pdf",
                {
                    1: b"<< /Type /Catalog /Pages 2 0 R >>",
                    2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                    3: page,
                    4: content,
                    7: font,
                },
            )
            result = extract_text_with_sources(path)

        records = result.sources.records[:3]
        self.assertEqual([s.code for s in records], [b"A", b"A", b"B"])
        self.assertEqual(
            [(s.byte_start, s.byte_end) for s in records],
            [(0, 1), (1, 2), (2, 3)],
        )


class MultiUnicodeAndCMapSpanTests(unittest.TestCase):
    def test_cmap_reports_complete_multibyte_code_span(self):
        cmap = CMap()
        cmap.code2cid = {0: {1: 100}, 2: 200}
        spans = list(cmap.decode_with_spans(b"\x00\x01\x02"))
        self.assertEqual(spans, [(100, 0, 2), (200, 2, 3)])

    def test_all_unicode_chars_share_full_input_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_multi_unicode_variant(Path(tmp) / "multi.pdf")
            result = extract_text_with_sources(path)

        self.assertTrue(result.text.startswith("AB\n"))
        chars = result.characters(0, 2)
        self.assertEqual([item.char for item in chars], ["A", "B"])
        first = chars[0].source
        second = chars[1].source
        self.assertIsNotNone(first)
        self.assertIs(first, second)
        assert first is not None
        self.assertEqual(first.record_id, second.record_id)
        self.assertEqual((first.byte_start, first.byte_end), (0, 1))
        self.assertEqual(first.code, b"A")
        self.assertEqual(result.source_at(2), None)  # layout newline


class ReuseVariantTests(unittest.TestCase):
    def test_repeated_and_nested_form_calls_have_distinct_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_form_reuse_variant(Path(tmp) / "form.pdf")
            result = extract_text_with_sources(path)

        form_sources = [
            s
            for s in result.sources.records
            if s.stream_object_number == 8 and s.code == b"F"
        ]
        self.assertEqual(len(form_sources), 3)

        signatures = {
            tuple(
                (call.object_number, call.call_index, call.call_order)
                for call in source.path
            )
            for source in form_sources
        }
        self.assertEqual(
            signatures,
            {
                ((8, 0, 0),),
                ((8, 1, 1),),
                ((9, 0, 2), (8, 0, 0)),
            },
        )

        # The same stream operator/byte range is present in every call, but
        # records and invocation paths are not merged.
        self.assertTrue(all(s.operator_index == 0 for s in form_sources))
        self.assertTrue(all(s.byte_start == 0 for s in form_sources))
        self.assertEqual(len({s.record_id for s in form_sources}), 3)

        outer_sources = [
            s
            for s in result.sources.records
            if s.stream_object_number == 9 and s.code == b"O"
        ]
        self.assertEqual(len(outer_sources), 1)
        self.assertEqual(
            [
                (c.object_number, c.call_index, c.call_order)
                for c in outer_sources[0].path
            ],
            [(9, 0, 2)],
        )

    def test_shared_content_stream_on_two_pages_is_not_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_shared_content_variant(Path(tmp) / "shared.pdf")
            result = extract_text_with_sources(path)

        pages = {}
        for source in result.sources.records:
            pages.setdefault(source.page_index, []).append(source)

        self.assertEqual(sorted(pages), [0, 1])
        for page_index, records in pages.items():
            self.assertTrue(all(s.page_index == page_index for s in records))
            self.assertTrue(all(s.stream_object_number == 4 for s in records))
            self.assertEqual(
                sorted({s.operator_index for s in records}), [0, 1, 2, 3]
            )

        first_h = next(
            s for s in pages[0] if s.operator_index == 1 and s.code == b"H"
        )
        second_h = next(
            s for s in pages[1] if s.operator_index == 1 and s.code == b"H"
        )
        self.assertEqual(first_h.operator_index, second_h.operator_index)
        self.assertEqual(first_h.byte_range, second_h.byte_range)
        self.assertNotEqual(first_h.record_id, second_h.record_id)
        self.assertNotEqual(first_h.page_index, second_h.page_index)


class LayoutAndConverterConsistencyTests(unittest.TestCase):
    def test_simple4_regrouping_keeps_interval_lookup_valid(self):
        signatures_by_layout = {}
        for line_margin in (0.19, 0.21):
            laparams = LAParams(line_margin=line_margin)
            text_result = extract_text_with_sources(SIMPLE4, laparams=laparams)

            layout_map = SourceMap()
            pages = list(
                extract_pages(
                    SIMPLE4, laparams=laparams, source_map=layout_map
                )
            )
            self.assertEqual(len(pages), 1)
            layout_text = layout_map.index_layout(pages[0])
            self.assertEqual(layout_text, text_result.text)
            self.assertEqual(
                text_signatures(layout_map),
                text_signatures(text_result.sources),
            )
            signatures_by_layout[line_margin] = text_result.text

            text = text_result.text
            needle_start = text.index("Text2")
            interval = text_result.lookup(
                needle_start, needle_start + len("Text2")
            )
            self.assertEqual("".join(item.text for item in interval), "Text2")
            self.assertEqual(len(interval), 5)
            for item in interval:
                assert item.source is not None
                self.assertEqual(item.source.operator_index, 1)
                # simple4 uses a TJ array with several string operands, so
                # output offset is not byte offset inside every operand.
                self.assertEqual(
                    item.text, item.source.code.decode("latin1")
                )

        self.assertIn("Text1\n\nText2", signatures_by_layout[0.19])
        self.assertIn("Text1\nText2\nText3", signatures_by_layout[0.21])

    def test_same_ltpage_sources_reach_converter_output(self):
        source_map = SourceMap()
        resource_manager = PDFResourceManager()
        device = PDFPageAggregator(
            resource_manager, laparams=LAParams(line_margin=0.19)
        )
        interpreter = PDFPageInterpreter(
            resource_manager, device, source_map=source_map
        )

        with open(SIMPLE4, "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
            ltpage = device.get_result()

        out = StringIO()
        converter = TextConverter(
            resource_manager,
            out,
            laparams=LAParams(line_margin=0.19),
            source_map=source_map,
        )
        converter.receive_layout(ltpage)
        text = out.getvalue()
        source_map.bind_text(text)

        lt_chars: List[LTChar] = []

        def collect_chars(item) -> None:
            if isinstance(item, LTChar):
                lt_chars.append(item)
            elif isinstance(item, LTContainer):
                for child in item:
                    collect_chars(child)

        collect_chars(ltpage)
        self.assertEqual(len(lt_chars), len(source_map.records))
        for char in lt_chars:
            self.assertIsInstance(char.source, NativeSource)
            assert isinstance(char.source, NativeSource)
            self.assertIs(source_map.records[char.source.record_id], char.source)

        # Every native output position resolves to the exact same record object.
        for character in source_map.characters():
            if character.source is not None:
                self.assertIs(
                    source_map.records[character.source.record_id],
                    character.source,
                )


class DocumentBindingAndLifecycleTests(unittest.TestCase):
    def test_query_after_close_and_two_identical_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "simple1-copy-one.pdf"
            second_path = Path(tmp) / "simple1-copy-two.pdf"
            shutil.copyfile(SIMPLE1, first_path)
            shutil.copyfile(SIMPLE1, second_path)

            with first_path.open("rb") as fp:
                first = extract_text_with_sources(fp, document_name="one")
            second = extract_text_with_sources(second_path, document_name="two")

            # Path-based extraction closes its own file before returning.
            first_path.unlink()
            second_path.unlink()

            first_source = first.source_at(0)
            second_source = second.source_at(0)
            assert first_source is not None
            assert second_source is not None

            self.assertNotEqual(first.sources.document_id, second.sources.document_id)
            self.assertEqual(first_source.document_id, first.sources.document_id)
            self.assertEqual(second_source.document_id, second.sources.document_id)
            self.assertNotEqual(first_source.document_id, second_source.document_id)

            # Locally identical PDF structure, but records are not shared.
            self.assertEqual(
                locator_signature(first_source)[:-1],
                locator_signature(second_source)[:-1],
            )
            self.assertIsNot(
                first.sources.records[0], second.sources.records[0]
            )

            # Processing the second document has not displaced the first map.
            self.assertEqual(first.text, SIMPLE1_EXPECTED)
            self.assertEqual(second.text, SIMPLE1_EXPECTED)
            self.assertEqual(first_source.code, b"H")
            first_interval = first.lookup(0, 1)[0]
            assert first_interval.source is not None
            self.assertEqual(
                first_interval.source.document_id,
                first.sources.document_id,
            )

    def test_query_does_not_call_page_interpreter(self):
        result = extract_text_with_sources(SIMPLE1)

        def fail(*args, **kwargs):
            raise AssertionError("query must not reinterpret a page")

        original = PDFPageInterpreter.process_page
        PDFPageInterpreter.process_page = fail
        try:
            intervals = result.lookup(0, len(result.text))
            chars = result.characters(0, 10)
            sliced = result.sources[8:13]
        finally:
            PDFPageInterpreter.process_page = original

        self.assertTrue(intervals)
        self.assertEqual(len(chars), 10)
        self.assertEqual("".join(i.text for i in sliced), "World")


class MemoryOrganizationTests(unittest.TestCase):
    def test_compact_character_index_and_shared_operator_descriptors(self):
        result = extract_text_with_sources(SIMPLE1)
        source_map = result.sources

        # Four bytes per output code point rather than a copied location object.
        self.assertEqual(source_map.char_sources.itemsize, 4)
        self.assertEqual(len(source_map.char_sources), len(result.text))

        op0 = [s for s in source_map.records if s.operator_index == 0]
        self.assertTrue(op0)
        self.assertTrue(all(s.operator_ref is op0[0].operator_ref for s in op0))
        self.assertTrue(all(s.path is op0[0].path for s in op0))

        # The map stores no decoded content-stream container.  Stream uids
        # are scoped to the per-extraction session and disappear on commit.
        self.assertFalse(hasattr(source_map, "content_streams"))
        self.assertIsNone(source_map._session)
        gc.collect()


if __name__ == "__main__":
    unittest.main()
