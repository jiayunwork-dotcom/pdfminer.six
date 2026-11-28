.. _source_mapping:

Source mapping / provenance
***************************

Text extraction decodes character codes and then rearranges glyphs into lines
and boxes.  Two equal strings or nearby coordinates therefore do not prove
that they came from the same drawing operation.

Use :func:`pdfminer.high_level.extract_text_with_sources` to obtain the final
Unicode text and a document-bound :class:`~pdfminer.sourcemap.SourceMap`:

.. code-block:: python

    from pdfminer.high_level import extract_text_with_sources

    result = extract_text_with_sources("simple1.pdf")
    print(result.text)

    for interval in result.lookup(0, 5):  # Unicode half-open interval
        print(interval.start, interval.end, interval.text)
        if interval.source is not None:
            s = interval.source
            print(
                s.page_number,
                s.stream_object_number,
                s.operator_index,
                s.operand_index,
                s.byte_start,
                s.byte_end,
                s.code,
            )

``lookup(start, end)`` groups consecutive output code points that share one
native character-code record.  Use
:meth:`~pdfminer.sourcemap.SourceMap.characters` when exactly one result per
Unicode code point is required.  Layout-generated spaces, line breaks, text-box
suffixes and form-feed page separators have ``source is None``.

A native source contains:

* page index/number and page object number;
* content-stream object number/generation and ``/Contents`` array index;
* the zero-based ordinal of the text-showing operator (``Tj``, ``TJ``, ``'`` or
  ``"``) inside that stream;
* the operand and string-operand index inside ``TJ``;
* a half-open byte interval in the decoded/unescaped PDF string operand;
* the CID/character code and a small interned copy of those code bytes;
* the complete Form-XObject invocation path, including repeated and nested
  calls.

If one character code maps to several Unicode characters through ToUnicode,
every resulting code point points to the same record and therefore to the same
complete input byte span.

The same :class:`~pdfminer.sourcemap.SourceMap` can be passed to
:func:`~pdfminer.high_level.extract_text` or
:func:`~pdfminer.high_level.extract_pages`.  For a layout tree, call
:meth:`~pdfminer.sourcemap.SourceMap.index_layout` (or ``index_layouts`` for
several pages) to verify the same sources after :class:`~pdfminer.layout.LAParams`
regrouping.

Reusing a SourceMap and failure semantics
=========================================

A single :class:`~pdfminer.sourcemap.SourceMap` can be handed to several
extraction calls in sequence.  Every successful call is an independent
extraction: it gets a fresh monotonic document id (even for the same file and
page selection) and its result atomically replaces the previous binding.
Intervals are half-open ranges over the Unicode characters the text converter
actually wrote during that call:

* text already present in the output stream before the call is not part of any
  interval, and the stream never needs to be readable or seekable;
* for binary output the index follows the exact Unicode segment handed to the
  encoder, including characters dropped by codec error handling;
* layout-added spaces, newlines and form feeds remain ``source is None``.

While an extraction is running, queries keep returning the last fully
committed result.  If interpretation, layout analysis or writing raises, or
the output stream reports a short write, the attempt is rolled back: the
previously committed index stays complete and queryable, and a first-ever
failure leaves the map unbound.  Content already written into the output
stream is not withdrawn.  Starting a second extraction with the same map
before the first one finishes raises
:class:`~pdfminer.pdfexceptions.SourceMapInUseError` before any text is
written; calls that use separate maps proceed independently.  Calls without a
``source_map`` argument behave exactly as before.

The boundaries are also available directly through
:meth:`~pdfminer.sourcemap.SourceMap.begin_extraction`,
:meth:`~pdfminer.sourcemap.SourceMap.commit_extraction`,
:meth:`~pdfminer.sourcemap.SourceMap.rollback_extraction` and the
:meth:`~pdfminer.sourcemap.SourceMap.extraction` context manager.

Memory trade-off
================

The map stores a four-byte record id for every output Unicode code point in an
``array('i')``.  Each distinct decoded character-code occurrence has one
slotted ``NativeSource`` record.  Page/operator/stream descriptors are shared by
all codes in an operation, and Form invocation paths are shared by all
operations in that invocation.  The map deliberately does **not** retain whole
content streams or the PDF file/parser objects.  Consequently:

* a one-character query is O(1) and an interval query is proportional to the
  queried interval;
* reopening/reinterpreting a page is never required;
* repeated text or repeated XObject references do not duplicate a content
  stream per output character;
* the cost is O(number of output code points + number of native code
  occurrences), with a compact four-byte index in addition to ordinary layout
  objects.

Query results contain only primitive, detached data and small code bytes, so
they remain usable after the input file is closed.  Each extraction gets its
own monotonic document id; processing two byte-identical files produces two
independent maps and never shares records between documents.

From the repository root, ``python3 tools/sourcemap_demo.py`` demonstrates all
of these behaviours on ``samples/simple1.pdf``, ``samples/simple4.pdf`` and
small generated reuse/multi-Unicode variants.
