"""Regression tests for SourceMap extraction transaction boundaries.

The scenarios here pin down the contract that one SourceMap belongs to one
extraction call at a time:

* reusing a map across calls gives every successful call an independent
  document binding and never mixes the two extractions together;
* source intervals are half-open ranges over the Unicode the converter
  actually wrote, including binary output and codec-filtered characters;
* text already sitting in the output stream is outside every interval;
* a short write or any mid-extraction exception fails the attempt while the
  previously committed result stays fully queryable;
* two extractions sharing one map while the first is still running fail
  before the second one writes anything, while separate maps stay
  independent.
"""

import threading
import unittest
from io import BytesIO, StringIO
from typing import List, Optional

from pdfminer.converter import TextConverter
from pdfminer.high_level import extract_text, extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.pdfexceptions import SourceMapInUseError
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.sourcemap import SourceMap

from tests.helpers import absolute_sample_path

SIMPLE1 = absolute_sample_path("simple1.pdf")
SIMPLE4 = absolute_sample_path("simple4.pdf")


class PrefixedWriteOnlyTextIO:
    """Write-only stream that already contains text and cannot be read/seeked.

    Mimics a sink (socket/pipe wrapper) with data buffered before the
    extraction call.  The prefix must survive and stay outside the source
    index even though the stream never supports ``getvalue``/``seek``.
    """

    def __init__(self, prefix: str) -> None:
        self._parts: List[str] = [prefix]
        self.mode = "w"

    def write(self, text: str) -> int:
        self._parts.append(text)
        return len(text)

    def getvalue(self) -> str:
        return "".join(self._parts)


class ShortWriteTextIO(StringIO):
    """Text stream that reports a short write once, after the first bytes."""

    def __init__(self, fail_after: int = 3) -> None:
        super().__init__()
        self.fail_after = fail_after
        self._written = 0
        self.write_calls = 0

    def write(self, text: str) -> int:
        self.write_calls += 1
        room = max(0, self.fail_after - self._written)
        accepted = text[: min(len(text), room)]
        super().write(accepted)
        self._written += len(accepted)
        return len(accepted)


class ShortWriteBinaryIO(BytesIO):
    """Binary stream that accepts fewer bytes than it was given."""

    def __init__(self, fail_after: int = 4) -> None:
        super().__init__()
        self.fail_after = fail_after
        self._written = 0

    def write(self, data: bytes) -> int:  # type: ignore[override]
        room = max(0, self.fail_after - self._written)
        accepted = data[: min(len(data), room)]
        n = super().write(accepted)
        self._written += n
        return n


class FailingTextIO(StringIO):
    """Text stream that raises from write() once a threshold is crossed."""

    def __init__(self, fail_after_chars: int = 5) -> None:
        super().__init__()
        self.fail_after_chars = fail_after_chars
        self._written = 0

    def write(self, text: str) -> int:
        self._written += len(text)
        if self._written >= self.fail_after_chars:
            raise RuntimeError("simulated output stream failure")
        return super().write(text)


def _interpreter_error_source_map(source_map: SourceMap, pdf_path: str) -> None:
    """Drive a real interpreter but make layout analysis blow up mid-page."""
    rsrcmgr = PDFResourceManager()
    out = StringIO()
    device = TextConverter(rsrcmgr, out, laparams=LAParams(), source_map=source_map)
    interpreter = PDFPageInterpreter(rsrcmgr, device, source_map=source_map)

    from pdfminer import layout as layout_module

    def boom(self, laparams):  # noqa: ANN001
        raise RuntimeError("simulated layout failure")

    source_map.begin_extraction()
    original_analyze = layout_module.LTLayoutContainer.analyze
    layout_module.LTLayoutContainer.analyze = boom  # type: ignore[assignment]
    try:
        with open(pdf_path, "rb") as fp:
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
    finally:
        layout_module.LTLayoutContainer.analyze = original_analyze  # type: ignore
        if source_map._session is not None:
            source_map.rollback_extraction()


class ReuseAcrossExtractionTests(unittest.TestCase):
    def test_reused_map_rebinds_per_successful_call(self):
        source_map = SourceMap(document_name="first")
        first = extract_text(SIMPLE1, source_map=source_map)
        first_id = source_map.document_id
        self.assertEqual(source_map.text, first)

        # Same file, same pages: the second call must still get its own
        # document identity and replace, not extend, the first result.
        second = extract_text(SIMPLE1, source_map=source_map)
        self.assertEqual(second, first)
        self.assertEqual(source_map.text, second)
        self.assertNotEqual(source_map.document_id, first_id)
        self.assertEqual(source_map.document_name, "first")

        # The committed index matches exactly the new text length; nothing
        # from the first extraction leaked in.
        self.assertEqual(len(source_map.char_sources), len(second))
        native = source_map.lookup(native_only=True)
        self.assertTrue(native)
        self.assertTrue(
            all(
                interval.source.document_id == source_map.document_id
                for interval in native
            )
        )

    def test_reuse_between_different_documents(self):
        source_map = SourceMap()
        text_one = extract_text(SIMPLE1, source_map=source_map)
        id_one = source_map.document_id
        text_four = extract_text(SIMPLE4, source_map=source_map)

        self.assertNotEqual(text_one, text_four)
        self.assertEqual(source_map.text, text_four)
        self.assertNotEqual(source_map.document_id, id_one)
        # Half-open intervals stay inside the new, shorter text.
        intervals = source_map.lookup()
        self.assertEqual((intervals[0].start, intervals[-1].end), (0, len(text_four)))
        self.assertEqual("".join(i.text for i in intervals), text_four)

    def test_queries_see_last_commitment_while_new_extraction_runs(self):
        source_map = SourceMap()
        committed = extract_text(SIMPLE1, source_map=source_map)
        committed_id = source_map.document_id
        committed_first = source_map.source_at(0)

        source_map.begin_extraction()
        try:
            # Mid-extraction queries must keep serving the committed result.
            self.assertEqual(source_map.text, committed)
            self.assertIs(source_map.source_at(0), committed_first)
            self.assertEqual(
                source_map.lookup(0, 5)[0].source.document_id, committed_id
            )
        finally:
            source_map.rollback_extraction()

        # After rollback the committed result is untouched.
        self.assertEqual(source_map.text, committed)
        self.assertIs(source_map.source_at(0), committed_first)

    def test_unbound_map_rejects_queries_after_first_failure(self):
        source_map = SourceMap()
        source_map.begin_extraction()
        source_map.rollback_extraction()
        with self.assertRaises(Exception):
            source_map.source_at(0)
        self.assertIsNone(source_map.text)

    def test_length_mismatch_rolls_back_and_releases_the_map(self):
        source_map = SourceMap()
        committed = extract_text(SIMPLE1, source_map=source_map)
        committed_id = source_map.document_id

        source_map.begin_extraction()
        with self.assertRaises(Exception):
            # Deliberately bind text whose length cannot match the
            # (empty) per-character index of this fresh attempt.
            source_map.bind_text("definitely-the-wrong-length")
        self.assertIsNone(source_map._session)
        self.assertEqual(source_map.text, committed)
        self.assertEqual(source_map.document_id, committed_id)

        # The lock was released: a healthy reuse works immediately.
        retry = extract_text(SIMPLE4, source_map=source_map)
        self.assertEqual(source_map.text, retry)

    def test_calls_without_source_map_are_unchanged(self):
        first = extract_text(SIMPLE1)
        second = extract_text(SIMPLE4)
        self.assertTrue(first.startswith("Hello"))
        self.assertTrue(second.startswith("Text1"))


class PrefixedStreamAndUnicodeIntervalTests(unittest.TestCase):
    def test_prefixed_stream_text_is_not_indexed(self):
        prefix = "PREFIX-ALREADY-HERE\n"
        out = PrefixedWriteOnlyTextIO(prefix)
        source_map = SourceMap()
        with open(SIMPLE4, "rb") as fp:
            extract_text_to_fp(fp, out, laparams=LAParams(), source_map=source_map)

        extracted = "Text1\nText2\nText3\n\n\x0c"
        self.assertEqual(out.getvalue(), prefix + extracted)
        # The map covers only this call's converter output, never the prefix.
        self.assertEqual(source_map.text, extracted)
        self.assertEqual(len(source_map.char_sources), len(extracted))

        # The very first indexed character is the converter's first glyph.
        first_interval = source_map.lookup(0, 1)[0]
        self.assertEqual(first_interval.text, "T")
        self.assertIsNotNone(first_interval.source)

    def test_intervals_are_half_open_over_written_unicode(self):
        source_map = SourceMap()
        extract_text(SIMPLE4, source_map=source_map)
        text = source_map.text

        start = text.index("Text2")
        intervals = source_map.lookup(start, start + len("Text2"))
        self.assertEqual([i.start for i in intervals][0], start)
        self.assertEqual(intervals[-1].end, start + len("Text2"))
        self.assertEqual("".join(i.text for i in intervals), "Text2")
        # End is exclusive: the following generated newline is not included.
        self.assertIsNone(source_map.source_at(start + len("Text2")))

    def test_layout_generated_characters_have_no_native_source(self):
        source_map = SourceMap()
        extract_text(SIMPLE4, source_map=source_map)
        text = source_map.text
        newline = text.index("\n")
        self.assertIsNone(source_map.source_at(newline))
        # Page break emitted by the converter is also unsourced.
        self.assertIsNone(source_map.source_at(len(text) - 1))
        self.assertEqual(text[-1], "\x0c")

    def test_binary_output_indexes_unicode_sent_to_encoder(self):
        out = BytesIO()
        source_map = SourceMap()
        with open(SIMPLE4, "rb") as fp:
            extract_text_to_fp(
                fp,
                out,
                laparams=LAParams(),
                codec="utf-8",
                source_map=source_map,
            )

        decoded = out.getvalue().decode("utf-8")
        self.assertEqual(source_map.text, decoded)
        self.assertEqual(len(source_map.char_sources), len(decoded))
        native = source_map.lookup(native_only=True)
        self.assertTrue(native)
        self.assertEqual("".join(i.text for i in source_map.lookup()), decoded)

    def test_codec_filtered_output_still_matches_index(self):
        # When the codec cannot encode a character and errors are ignored,
        # the indexed text follows exactly what was handed to the encoder.
        out = StringIO()
        source_map = SourceMap()
        with open(SIMPLE4, "rb") as fp:
            extract_text_to_fp(
                fp,
                out,
                laparams=LAParams(),
                codec="ascii",
                source_map=source_map,
            )
        # simple4 is pure ASCII so every character is written; the index
        # matches the stream exactly and every interval stays in bounds.
        self.assertEqual(out.getvalue(), source_map.text)
        self.assertEqual(len(source_map.char_sources), len(out.getvalue()))
        self.assertEqual(source_map.lookup()[-1].end, len(out.getvalue()))

    def test_direct_converter_dropped_character_is_indexed_as_written(self):
        # Drive TextConverter directly with text the codec cannot represent:
        # the source index must describe the post-filter Unicode only.
        source_map = SourceMap()
        out = BytesIO()
        rsrcmgr = PDFResourceManager()
        converter = TextConverter(
            rsrcmgr,
            out,
            codec="ascii",
            laparams=LAParams(),
            source_map=source_map,
        )
        source_map.begin_extraction()
        try:
            # emit() is the converter's write+index path; é cannot be
            # ASCII-encoded, so only the six surviving chars get indexed.
            emitted = converter.write_text("abcédef")
            source_map.append_output(emitted, None)
            source_map.commit_extraction()
        finally:
            if source_map._session is not None:
                source_map.rollback_extraction()

        self.assertEqual(emitted, "abcdef")
        self.assertEqual(out.getvalue(), b"abcdef")
        self.assertEqual(source_map.text, "abcdef")
        self.assertEqual(len(source_map.char_sources), 6)


class FailureAndRetryTests(unittest.TestCase):
    def _commit_first(self, source_map: SourceMap) -> str:
        return extract_text(SIMPLE1, source_map=source_map)

    def test_short_write_on_text_stream_preserves_previous_result(self):
        source_map = SourceMap()
        previous = self._commit_first(source_map)
        previous_id = source_map.document_id

        failing = ShortWriteTextIO(fail_after=3)
        with open(SIMPLE4, "rb") as fp:
            with self.assertRaises(Exception):
                extract_text_to_fp(
                    fp,
                    failing,
                    laparams=LAParams(),
                    source_map=source_map,
                )

        # Already-written bytes stay in the stream, but the map rolls back.
        self.assertGreater(failing.tell(), 0)
        self.assertEqual(source_map.text, previous)
        self.assertEqual(source_map.document_id, previous_id)
        self.assertEqual(len(source_map.char_sources), len(previous))
        self.assertEqual(source_map.source_at(0).code, b"H")

        # A retry on a healthy stream commits a fresh result.
        retry = extract_text(SIMPLE4, source_map=source_map)
        self.assertEqual(source_map.text, retry)
        self.assertNotEqual(source_map.document_id, previous_id)

    def test_short_write_on_binary_stream_preserves_previous_result(self):
        source_map = SourceMap()
        previous = self._commit_first(source_map)
        previous_id = source_map.document_id

        failing = ShortWriteBinaryIO(fail_after=4)
        with open(SIMPLE4, "rb") as fp:
            with self.assertRaises(Exception):
                extract_text_to_fp(
                    fp,
                    failing,
                    laparams=LAParams(),
                    source_map=source_map,
                )

        self.assertEqual(source_map.text, previous)
        self.assertEqual(source_map.document_id, previous_id)
        # Nothing about the failed attempt extended the committed char index.
        self.assertEqual(len(source_map.char_sources), len(previous))

        healthy = BytesIO()
        with open(SIMPLE4, "rb") as fp:
            extract_text_to_fp(fp, healthy, laparams=LAParams(), source_map=source_map)
        self.assertEqual(source_map.text, healthy.getvalue().decode("utf-8"))

    def test_midstream_exception_preserves_previous_result(self):
        source_map = SourceMap()
        previous = self._commit_first(source_map)
        previous_id = source_map.document_id

        failing = FailingTextIO(fail_after_chars=5)
        with open(SIMPLE1, "rb") as fp:
            with self.assertRaises(RuntimeError):
                extract_text_to_fp(
                    fp,
                    failing,
                    laparams=LAParams(),
                    source_map=source_map,
                )

        self.assertEqual(source_map.text, previous)
        self.assertEqual(source_map.document_id, previous_id)

        # Written content stays in the failed stream.
        self.assertTrue(failing.getvalue())

        # Retry succeeds and yields a new binding.
        retry = extract_text(SIMPLE1, source_map=source_map)
        self.assertEqual(retry, previous)
        self.assertNotEqual(source_map.document_id, previous_id)

    def test_interpreter_failure_on_first_use_leaves_map_unbound(self):
        source_map = SourceMap()
        with self.assertRaises(RuntimeError):
            _interpreter_error_source_map(source_map, SIMPLE1)
        self.assertIsNone(source_map.text)
        self.assertEqual(source_map.records, [])
        # A later healthy extraction works and binds normally.
        text = extract_text(SIMPLE1, source_map=source_map)
        self.assertEqual(source_map.text, text)
        self.assertTrue(source_map.records)

    def test_interpreter_failure_after_commit_keeps_old_result(self):
        source_map = SourceMap()
        previous = self._commit_first(source_map)
        previous_id = source_map.document_id
        with self.assertRaises(RuntimeError):
            _interpreter_error_source_map(source_map, SIMPLE4)
        self.assertEqual(source_map.text, previous)
        self.assertEqual(source_map.document_id, previous_id)


class ConcurrencyAndIndependenceTests(unittest.TestCase):
    def test_second_extraction_on_busy_map_fails_before_writing(self):
        source_map = SourceMap()
        source_map.begin_extraction()
        try:
            blocking_out = StringIO()
            with open(SIMPLE1, "rb") as fp:
                with self.assertRaises(SourceMapInUseError):
                    extract_text_to_fp(
                        fp,
                        blocking_out,
                        laparams=LAParams(),
                        source_map=source_map,
                    )
            # The rejected call must not have written a single character.
            self.assertEqual(blocking_out.getvalue(), "")
        finally:
            source_map.rollback_extraction()

    def test_busy_map_fails_via_extract_text_too(self):
        source_map = SourceMap()
        source_map.begin_extraction()
        try:
            with self.assertRaises(SourceMapInUseError):
                extract_text(SIMPLE1, source_map=source_map)
        finally:
            source_map.rollback_extraction()

    def test_separate_maps_extract_independently(self):
        results: List[Optional[str]] = [None, None]
        maps = [SourceMap(), SourceMap()]

        def worker(index: int) -> None:
            path = SIMPLE1 if index == 0 else SIMPLE4
            results[index] = extract_text(path, source_map=maps[index])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(maps[0].text, results[0])
        self.assertEqual(maps[1].text, results[1])
        self.assertNotEqual(maps[0].document_id, maps[1].document_id)
        self.assertEqual(maps[0].source_at(0).code, b"H")
        self.assertEqual(maps[1].source_at(0).code, b"T")

    def test_context_manager_commits_and_rolls_back(self):
        source_map = SourceMap()
        with source_map.extraction():
            self.assertIsNotNone(source_map._session)
        self.assertIsNone(source_map._session)

        with self.assertRaises(RuntimeError):
            with source_map.extraction():
                raise RuntimeError("boom")
        self.assertIsNone(source_map._session)


if __name__ == "__main__":
    unittest.main()
