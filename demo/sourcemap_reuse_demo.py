#!/usr/bin/env python3
"""Terminal demo: reuse one SourceMap across extractions safely.

Run from the repository root with an isolated interpreter::

    python3 -I demo/sourcemap_reuse_demo.py

The script shows, on the real samples:

1. two successful extractions sharing one SourceMap get independent document
   identities, and queries always describe the last completed extraction;
2. a stream that reports a short write fails the second extraction while the
   first extraction's sources remain fully intact and queryable;
3. retrying on a healthy stream then commits a fresh result.
"""

import sys
from io import BytesIO

sys.path.insert(0, "/home/node/.local/lib/python3.11/site-packages")
sys.path.insert(0, "/workspace")

from pdfminer.high_level import extract_text, extract_text_to_fp  # noqa: E402
from pdfminer.layout import LAParams  # noqa: E402
from pdfminer.pdfexceptions import PDFIOError  # noqa: E402
from pdfminer.sourcemap import SourceMap  # noqa: E402


SAMPLE1 = "/workspace/samples/simple1.pdf"
SAMPLE4 = "/workspace/samples/simple4.pdf"


class ShortWriteBinaryIO(BytesIO):
    """Binary sink that accepts only the first ``fail_after`` bytes."""

    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self.fail_after = fail_after
        self._written = 0

    def write(self, data):  # type: ignore[override]
        room = max(0, self.fail_after - self._written)
        accepted = data[: min(len(data), room)]
        n = super().write(accepted)
        self._written += n
        return n


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    source_map = SourceMap(document_name="reused-map")

    banner("1. First successful extraction (simple1.pdf)")
    first_text = extract_text(SAMPLE1, source_map=source_map)
    first_id = source_map.document_id
    first_h = source_map.source_at(0)
    print(f"document_id      = {first_id}")
    print(f"text length      = {len(first_text)}")
    print(f"source_at(0)     = {first_h.code!r} on page "
          f"{first_h.page_number}, operator {first_h.operator}")

    banner("2. Reuse the same SourceMap on simple4.pdf (independent identity)")
    second_text = extract_text(SAMPLE4, source_map=source_map)
    second_id = source_map.document_id
    assert second_id != first_id
    print(f"new document_id  = {second_id}  (different from {first_id})")
    print(f"bound text length= {len(second_text)}  (old result replaced, "
          f"not appended)")
    start = second_text.index("Text2")
    intervals = source_map.lookup(start, start + 5)
    print("lookup('Text2')  =",
          [(i.text, i.source.code.decode()) for i in intervals])

    banner("3. Failed extraction: binary stream reports a short write")
    failing = ShortWriteBinaryIO(fail_after=4)
    try:
        with open(SAMPLE1, "rb") as fp:
            extract_text_to_fp(
                fp, failing, laparams=LAParams(), source_map=source_map
            )
    except PDFIOError as exc:
        print(f"extraction failed as expected: {exc}")
    print(f"bytes left in the failed stream: {failing.getvalue()!r} "
          "(not rolled back)")

    # The last *completed* result is still the simple4 extraction.
    assert source_map.document_id == second_id
    assert source_map.text == second_text
    still_here = source_map.lookup(start, start + 5)
    print("previous result retained: document_id =", source_map.document_id)
    print("previous intervals still queryable  =",
          [(i.text, i.source.code.decode()) for i in still_here])

    banner("4. Retry on a healthy stream -> fresh committed result")
    healthy = BytesIO()
    with open(SAMPLE1, "rb") as fp:
        extract_text_to_fp(
            fp, healthy, laparams=LAParams(), source_map=source_map
        )
    retry_id = source_map.document_id
    print(f"retry document_id = {retry_id}")
    print(f"bound text length = {len(source_map.text)}")
    print(f"stream == bound   = "
          f"{healthy.getvalue().decode('utf-8') == source_map.text}")
    print("source_at(0)     =", source_map.source_at(0).code)
    print("\nAll checks passed: the previous completed source result "
          "survived the failure.")


if __name__ == "__main__":
    main()
