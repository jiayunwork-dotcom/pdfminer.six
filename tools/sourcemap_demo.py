#!/usr/bin/env python3
"""Terminal demonstration for pdfminer.six source/provenance mapping.

Run from the repository root:

    python3 tools/sourcemap_demo.py

The script uses samples/simple1.pdf and samples/simple4.pdf and creates small
reuse/multi-Unicode variants in a temporary directory.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfminer.high_level import extract_text, extract_text_with_sources  # noqa: E402
from pdfminer.layout import LAParams  # noqa: E402
from pdfminer.sourcemap import NativeSource  # noqa: E402
from tests.sourcemap_fixtures import (  # noqa: E402
    write_form_reuse_variant,
    write_multi_unicode_variant,
    write_shared_content_variant,
)


def heading(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def describe(source: Optional[NativeSource]) -> str:
    if source is None:
        return "<generated layout/converter text: no native source>"
    path = " -> ".join(
        f"{call.name}#call{call.call_index}" for call in source.path
    ) or "<page content>"
    return (
        f"doc={source.document_id} page={source.page_number} "
        f"stream_obj={source.stream_object_number} path={path} "
        f"op#{source.operator_index}:{source.operator} "
        f"operand={source.operand_index} string={source.string_index} "
        f"bytes[{source.byte_start}:{source.byte_end}] code={source.code!r}"
    )


def main() -> int:
    simple1 = ROOT / "samples" / "simple1.pdf"
    simple4 = ROOT / "samples" / "simple4.pdf"

    heading("1. Opt-in mapping; default output remains unchanged")
    plain = extract_text(str(simple1))
    mapped = extract_text_with_sources(str(simple1))
    print(f"extract_text(simple1) == mapped.text: {plain == mapped.text}")
    print(
        f"text length={len(mapped.text)}, "
        f"source records={len(mapped.sources.records)}"
    )
    assert plain == mapped.text

    heading("2. Unicode interval -> page/stream/operator/operand/byte range")
    pos = mapped.text.index("World")
    print(f"querying final text interval [{pos}:{pos + 5}) = 'World'")
    for interval in mapped.lookup(pos, pos + 5):
        print(f"  output[{interval.start}:{interval.end}] {interval.text!r}")
        print(f"    {describe(interval.source)}")

    op3_h = next(
        s
        for s in mapped.sources.records
        if s.operator_index == 3 and s.code == b"H"
    )
    op3_e = next(
        s
        for s in mapped.sources.records
        if s.operator_index == 3 and s.code == b"e"
    )
    print("Adjacent one-byte TJ operands both use byte [0:1], but:")
    print(
        "  H: "
        f"operand_index={op3_h.operand_index}, "
        f"string_index={op3_h.string_index}"
    )
    print(
        "  e: "
        f"operand_index={op3_e.operand_index}, "
        f"string_index={op3_e.string_index}"
    )
    assert (op3_h.operand_index, op3_h.string_index) != (
        op3_e.operand_index,
        op3_e.string_index,
    )

    heading(
        "3. One code -> multiple Unicode chars; "
        "generated spaces/newlines -> None"
    )
    with tempfile.TemporaryDirectory() as tmp:
        multi_path = write_multi_unicode_variant(Path(tmp) / "multi.pdf")
        multi = extract_text_with_sources(str(multi_path))
        a = multi.source_at(0)
        b = multi.source_at(1)
        print(f"mapped text prefix={multi.text[:2]!r}")
        print(f"A: {describe(a)}")
        print(f"B: {describe(b)}")
        assert a is b

        heading("4. Form XObject call paths and shared content on pages")
        form_path = write_form_reuse_variant(Path(tmp) / "form.pdf")
        form = extract_text_with_sources(str(form_path))
        f_sources = [
            s
            for s in form.sources.records
            if s.stream_object_number == 8 and s.code == b"F"
        ]
        print("\nRepeated/nested Form XObject calls to object 8:")
        for source in f_sources:
            print("  " + describe(source))
        assert len(f_sources) == 3

        shared_path = write_shared_content_variant(Path(tmp) / "shared.pdf")
        shared = extract_text_with_sources(str(shared_path))
        print("\nSame content-stream object 4 reused by two pages:")
        for page_index in (0, 1):
            source = next(
                s
                for s in shared.sources.records
                if s.page_index == page_index
                and s.operator_index == 1
                and s.code == b"H"
            )
            print("  " + describe(source))

        heading("5. LAParams regrouping changes offsets, not source identity")
        for line_margin in (0.19, 0.21):
            params = LAParams(line_margin=line_margin)
            result4 = extract_text_with_sources(str(simple4), laparams=params)
            pos = result4.text.index("Text2")
            interval = result4.lookup(pos, pos + 5)
            ops = {
                item.source.operator_index
                for item in interval
                if item.source is not None
            }
            print(
                f"line_margin={line_margin}: Text2 at [{pos}:{pos + 5}), "
                f"text={result4.text!r}, source operators={sorted(ops)}"
            )
            assert ops == {1}

        heading("6. Closed file and two byte-identical documents stay bound separately")
        copy_path = Path(tmp) / "identical-copy.pdf"
        shutil.copyfile(simple1, copy_path)
        with simple1.open("rb") as fp:
            first = extract_text_with_sources(fp, document_name="first-handle")
        second = extract_text_with_sources(str(copy_path), document_name="second")
        # The `with` block has now closed the file object.
        print(f"first.document_id={first.sources.document_id} (file handle closed)")
        print(f"second.document_id={second.sources.document_id}")
        print(f"first source still queryable: {describe(first.source_at(0))}")
        print(f"second source still queryable: {describe(second.source_at(0))}")
        assert first.sources.document_id != second.sources.document_id

    heading("7. Index organization / memory")
    sm = mapped.sources
    print(f"output code points: {len(sm.char_sources)}")
    print(f"bytes per output code point: {sm.char_sources.itemsize} (array('i'))")
    print(f"unique native code records: {len(sm.records)}")
    op0 = [s for s in sm.records if s.operator_index == 0]
    shared = sum(s.operator_ref is op0[0].operator_ref for s in op0)
    print(f"operator #0 records sharing one OperatorRef: {shared}/{len(op0)}")
    print("Query is direct array indexing + interval scan; it never reparses a page.")

    print("\nAll demo assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
