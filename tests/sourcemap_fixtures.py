"""Small PDF builders for source-map and provenance tests.

The files are intentionally hand-authored, uncompressed PDFs so that object
numbers and content-stream byte offsets are stable assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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


def _ocg_dict(name: str) -> bytes:
    return (
        b"<< /Type /OCG /Name (" + name.encode("ascii") + b") >>"
    )


def write_ocg_variant(path: Path) -> Path:
    """One page covering direct/resource OCG gates, OCMD VE and Form /OC.

    Layout (all text Helvetica, native drawing order):

    * ``Always`` is never gated.
    * ``AlphaOne`` sits behind a direct BDC/OC dictionary pointing at OCG 10.
    * ``AlphaTwo`` sits behind a resource property name (``/L2``) pointing at
      OCG 11.
    * ``BothOnly`` is gated by an OCMD (15) whose ``/VE`` is
      ``[/And 10 0 R 11 0 R]``.
    * ``EitherOne`` is gated by an OCMD (16) with ``/VE [/Or 10 0 R 11 0 R]``.
    * ``NotAlpha`` is gated by an OCMD (17) with ``/VE [/Not 10 0 R]``.
    * The nested Form (18) has its own ``/OC 11 0 R`` entry and prints
      ``FormHidden`` wrapped in a BDC gate on OCG 10, exercising conjunctive
      nested visibility.
    """
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    form = stream_object(
        b"BT /F1 18 Tf 0 0 Td (FormHidden) Tj ET",
        "/Type /XObject /Subtype /Form /BBox [0 0 200 40] "
        "/OC 11 0 R "
        "/Resources << /Font << /F1 7 0 R >> >>",
    )
    page_content = stream_object(
        b"BT /F1 24 Tf 50 740 Td (Always) Tj ET\n"
        b"/L1 BDC\n"
        b"BT /F1 24 Tf 50 700 Td (AlphaOne) Tj ET\n"
        b"EMC\n"
        b"/L2 BDC\n"
        b"BT /F1 24 Tf 50 660 Td (AlphaTwo) Tj ET\n"
        b"EMC\n"
        b"/LBoth BDC\n"
        b"BT /F1 24 Tf 50 620 Td (BothOnly) Tj ET\n"
        b"EMC\n"
        b"/LEither BDC\n"
        b"BT /F1 24 Tf 50 580 Td (EitherOne) Tj ET\n"
        b"EMC\n"
        b"/LNot BDC\n"
        b"BT /F1 24 Tf 50 540 Td (NotAlpha) Tj ET\n"
        b"EMC\n"
        b"q 1 0 0 1 50 500 cm /Fm1 Do Q"
    )
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 7 0 R >> "
        b"/XObject << /Fm1 18 0 R >> "
        b"/Properties << "
        b"/L1 << /OC 10 0 R >> "
        b"/L2 << /OC 11 0 R >> "
        b"/LBoth << /OC 15 0 R >> "
        b"/LEither << /OC 16 0 R >> "
        b"/LNot << /OC 17 0 R >> "
        b">> >> >>"
    )
    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [10 0 R 11 0 R] >> >>"
    )
    return write_pdf(
        path,
        {
            1: catalog,
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: page_content,
            7: font,
            10: _ocg_dict("Alpha"),
            11: _ocg_dict("Beta"),
            15: b"<< /Type /OCMD /VE [/And 10 0 R 11 0 R] >>",
            16: b"<< /Type /OCMD /VE [/Or 10 0 R 11 0 R] >>",
            17: b"<< /Type /OCMD /VE [/Not 10 0 R] >>",
            18: form,
        },
    )


def write_ocg_policy_variant(path: Path) -> Path:
    """OCMD without VE exercising the four /P policies and bare OCG refs."""
    page_content = stream_object(
        b"/AllOn BDC\n"
        b"BT /F1 24 Tf 50 720 Td (AllOnText) Tj ET\n"
        b"EMC\n"
        b"/AnyOn BDC\n"
        b"BT /F1 24 Tf 50 680 Td (AnyOnText) Tj ET\n"
        b"EMC\n"
        b"/AllOff BDC\n"
        b"BT /F1 24 Tf 50 640 Td (AllOffText) Tj ET\n"
        b"EMC\n"
        b"/AnyOff BDC\n"
        b"BT /F1 24 Tf 50 600 Td (AnyOffText) Tj ET\n"
        b"EMC"
    )
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 7 0 R >> "
        b"/Properties << "
        b"/AllOn << /OC 20 0 R >> "
        b"/AnyOn << /OC 21 0 R >> "
        b"/AllOff << /OC 22 0 R >> "
        b"/AnyOff << /OC 23 0 R >> "
        b">> >> >>"
    )
    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [10 0 R 11 0 R] >> >>"
    )
    return write_pdf(
        path,
        {
            1: catalog,
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: page_content,
            7: font,
            10: _ocg_dict("Gamma"),
            11: _ocg_dict("Delta"),
            20: b"<< /Type /OCMD /OCGs [10 0 R 11 0 R] /P /AllOn >>",
            21: b"<< /Type /OCMD /OCGs [10 0 R 11 0 R] /P /AnyOn >>",
            22: b"<< /Type /OCMD /OCGs [10 0 R 11 0 R] /P /AllOff >>",
            23: b"<< /Type /OCMD /OCGs [10 0 R 11 0 R] /P /AnyOff >>",
        },
    )


def write_ocg_cross_tj_form_variant(path: Path) -> Path:
    """OCG-gated text split across a TJ array and reused via two Form calls."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    form = stream_object(
        b"BT /F1 18 Tf 0 0 Td [(Sp)(lit)250(Text)] TJ ET",
        "/Type /XObject /Subtype /Form /BBox [0 0 300 40] "
        "/Resources << /Font << /F1 7 0 R >> >>",
    )
    inner = (
        b"q 1 0 0 1 50 700 cm /Fm1 Do Q "
        b"q 1 0 0 1 50 650 cm /Fm1 Do Q"
    )
    page_content = stream_object(b"/L1 BDC " + inner + b" EMC")
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 7 0 R >> "
        b"/XObject << /Fm1 8 0 R >> "
        b"/Properties << /L1 << /OC 10 0 R >> >> >> >>"
    )
    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [10 0 R] >> >>"
    )
    return write_pdf(
        path,
        {
            1: catalog,
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: page_content,
            7: font,
            8: form,
            10: _ocg_dict("Split"),
        },
    )


def write_sixteen_ocg_variant(path: Path) -> Path:
    """Deterministic 16-OCG fixture with exactly 4096 native characters.

    Every OCG gates an equal block of 254 native characters on one page
    (16 * 254 = 4064); the 23-character early marker and the 9-character
    ``NeedAll16`` marker add thirty-two more, for exactly 4096 native
    characters in total and every OCG off in the default state table.
    ``NeedAll16`` sits behind a conjunctive OCMD over all sixteen groups (a
    16-flip target) and the early marker behind a single-group gate (a
    1-flip decoy), giving the search a non-trivial global optimum that no
    first-feasible strategy finds.
    """
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    groups = [20 + index for index in range(16)]
    ocg_refs = b" ".join(f"{objid} 0 R".encode("ascii") for objid in groups)

    content = bytearray()
    property_entries = []
    for index, objid in enumerate(groups):
        name = f"G{index:02d}"
        property_entries.append(f"/{name} << /OC {objid} 0 R >>".encode("ascii"))
        # 254 native characters per group: a repeated marker letter keeps the
        # stream ASCII and avoids any layout-generated characters.
        chunk = (chr(ord("A") + (index % 26)) * 254).encode("ascii")
        content += b"/" + name.encode("ascii") + b" BDC\n"
        content += b"BT /F1 10 Tf 50 "
        content += str(760 - index * 34).encode("ascii")
        content += b" Td (" + chunk + b") Tj ET\n"
        content += b"EMC\n"

    # Low-cost decoy: one flip reveals this 23-character marker.
    content += b"/Early BDC\n"
    content += (
        b"BT /F1 18 Tf 50 200 Td "
        b"(EarlyFlipMarkerOneTwo34) Tj ET\n"
    )
    content += b"EMC\n"
    # Expensive target: a conjunctive OCMD over all sixteen groups.
    content += b"/All BDC\n"
    content += b"BT /F1 18 Tf 50 170 Td (NeedAll16) Tj ET\n"
    content += b"EMC\n"
    all_ve = b"[/And " + ocg_refs + b"]"
    property_entries.append(b"/All << /OC 40 0 R >>")
    property_entries.append(b"/Early << /OC 20 0 R >>")
    properties = b"/Properties << " + b" ".join(property_entries) + b" >>"

    page_content = stream_object(bytes(content))
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 7 0 R >> " + properties + b" >> >>"
    )
    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [" + ocg_refs + b"] >> >>"
    )
    objects = {
        1: catalog,
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: page,
        4: page_content,
        7: font,
        40: b"<< /Type /OCMD /VE " + all_ve + b" >>",
    }
    for index, objid in enumerate(groups):
        objects[objid] = _ocg_dict(f"Group{index:02d}")
    return write_pdf(path, objects)


def write_cyclic_ve_variant(path: Path) -> Path:
    """Two OCMDs referencing each other through /VE (a cycle)."""
    page_content = stream_object(
        b"/Loop BDC\n"
        b"BT /F1 24 Tf 50 720 Td (Cyclic) Tj ET\n"
        b"EMC"
    )
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 7 0 R >> "
        b"/Properties << /Loop << /OC 30 0 R >> >> >> >>"
    )
    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [10 0 R] >> >>"
    )
    return write_pdf(
        path,
        {
            1: catalog,
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: page_content,
            7: font,
            10: _ocg_dict("Lonely"),
            30: b"<< /Type /OCMD /VE [/Or 10 0 R 31 0 R] >>",
            31: b"<< /Type /OCMD /VE [/Not 30 0 R] >>",
        },
    )


def write_unbalanced_marked_content_variant(path: Path) -> Path:
    """A BDC without its EMC inside the page content stream."""
    page_content = stream_object(
        b"/L1 BDC\n"
        b"BT /F1 24 Tf 50 720 Td (Dangling) Tj ET"
    )
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 7 0 R >> "
        b"/Properties << /L1 << /OC 10 0 R >> >> >> >>"
    )
    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [10 0 R] >> >>"
    )
    return write_pdf(
        path,
        {
            1: catalog,
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: page_content,
            7: font,
            10: _ocg_dict("Dangling"),
        },
    )


def write_ocg_direct_dict_variant(path: Path) -> Path:
    """A BDC whose property list is an inline dictionary (not a resource name)."""
    page_content = stream_object(
        b"BT /F1 24 Tf 50 720 Td (Plain) Tj ET\n"
        b"/Tag << /OC 10 0 R  >> BDC\n"
        b"BT /F1 24 Tf 50 680 Td (Gated) Tj ET\n"
        b"EMC"
    )
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>"
    )
    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [10 0 R] >> >>"
    )
    return write_pdf(
        path,
        {
            1: catalog,
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: page,
            4: page_content,
            7: font,
            10: _ocg_dict("Direct"),
        },
    )


def write_ocg_two_page_variant(path: Path) -> Path:
    """The same marker on two pages behind different single-group gates."""
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    def page_content(objid: int, label: str) -> bytes:
        return stream_object(
            b"/L1 BDC\n"
            b"BT /F1 24 Tf 50 720 Td (" + label.encode("ascii") + b") Tj ET\n"
            b"EMC"
        )
    def page(contents_obj: int, gate_obj: int) -> bytes:
        return (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            + f"/Contents {contents_obj} 0 R ".encode("ascii")
            + b"/Resources << /Font << /F1 7 0 R >> "
            + f"/Properties << /L1 << /OC {gate_obj} 0 R >> >> >> >>".encode(
                "ascii"
            )
        )

    catalog = (
        b"<< /Type /Catalog /Pages 2 0 R "
        b"/OCProperties << /OCGs [10 0 R 11 0 R] >> >>"
    )
    return write_pdf(
        path,
        {
            1: catalog,
            2: b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
            3: page(4, 10),
            4: page_content(10, "SharedMarker"),
            6: page(5, 11),
            5: page_content(11, "SharedMarker"),
            7: font,
            10: _ocg_dict("PageOne"),
            11: _ocg_dict("PageTwo"),
        },
    )
