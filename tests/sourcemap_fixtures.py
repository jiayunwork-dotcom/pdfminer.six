"""Small PDF builders for source-map and provenance tests.

The files are intentionally hand-authored, uncompressed PDFs so that object
numbers and content-stream byte offsets are stable assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def stream_object(data: bytes, attributes: str = "") -> bytes:
    attrs = f" /Length {len(data)}"
    if attributes:
        attrs += " " + attributes
    return (
        f"<<{attrs} >>\nstream\n".encode("ascii")
        + data
        + b"\nendstream"
    )


def write_pdf(path: Path, objects: Dict[int, bytes]) -> Path:
    """Write a single-part, unencrypted PDF with a classic xref table."""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: Dict[int, int] = {}
    for object_number in sorted(objects):
        offsets[object_number] = len(out)
        out += f"{object_number} 0 obj\n".encode("ascii")
        out += objects[object_number]
        out += b"\nendobj\n"

    xref_offset = len(out)
    max_object = max(objects)
    out += f"xref\n0 {max_object + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for object_number in range(1, max_object + 1):
        if object_number in offsets:
            out += (
                f"{offsets[object_number]:010d} 00000 n \n".encode("ascii")
            )
        else:
            out += b"0000000000 65535 f \n"

    out += (
        f"trailer\n<< /Size {max_object + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(bytes(out))
    return path


def simple1_content_bytes() -> bytes:
    """Return the decoded content stream from samples/simple1.pdf."""
    sample = SAMPLES_DIR / "simple1.pdf"
    with sample.open("rb") as fp:
        document = PDFDocument(PDFParser(fp))
        page = next(PDFPage.create_pages(document))
        stream = page.contents[0]
        return stream.get_data()


def write_shared_content_variant(path: Path) -> Path:
    """Two page objects referencing the exact same content-stream object."""
    content = stream_object(simple1_content_bytes())
    font = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /MacRomanEncoding >>"
    )
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    return write_pdf(
        path,
        {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            3: page,
            4: content,
            5: font,
            6: page,
        },
    )


def write_form_reuse_variant(path: Path) -> Path:
    """A page calls one Form twice and another nested Form once."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    form = stream_object(
        b"BT /F1 18 Tf 0 0 Td (FormText) Tj ET",
        "/Type /XObject /Subtype /Form /BBox [0 0 200 40] "
        "/Resources << /Font << /F1 7 0 R >> >>",
    )
    outer = stream_object(
        b"BT /F1 18 Tf 0 0 Td (Outer) Tj ET q 1 0 0 1 30 0 cm /Fm1 Do Q",
        "/Type /XObject /Subtype /Form /BBox [0 0 300 80] "
        "/Resources << /Font << /F1 7 0 R >> "
        "/XObject << /Fm1 8 0 R >> >>",
    )
    page_content = stream_object(
        b"BT /F1 24 Tf 100 720 Td (PageText) Tj ET "
        b"q 1 0 0 1 50 680 cm /Fm1 Do Q "
        b"q 1 0 0 1 50 630 cm /Fm1 Do Q "
        b"q 1 0 0 1 50 580 cm /Outer Do Q"
    )
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 7 0 R >> "
        b"/XObject << /Fm1 8 0 R /Outer 9 0 R >> >> >>"
    )
    return write_pdf(
        path,
        {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: page_content,
            7: font,
            8: form,
            9: outer,
        },
    )


def write_multi_unicode_variant(path: Path) -> Path:
    """One one-byte character code maps through ToUnicode to ``AB``."""
    tounicode = stream_object(
        b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
1 beginbfchar
<41> <00410042>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""
    )
    font = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding /ToUnicode 8 0 R >>"
    )
    content = stream_object(b"BT /F1 24 Tf 100 700 Td (A) Tj ET")
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>"
    )
    return write_pdf(
        path,
        {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: content,
            7: font,
            8: tounicode,
        },
    )
