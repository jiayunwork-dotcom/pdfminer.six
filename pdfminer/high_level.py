"""Functions that can be used for the most common use-cases for pdfminer.six"""

import logging
import sys
from io import StringIO
from typing import Any, BinaryIO, Container, Dict, Iterator, Optional, Tuple, cast

from .pdfexceptions import PDFValueError
from .converter import (
    XMLConverter,
    HTMLConverter,
    TextConverter,
    PDFPageAggregator,
    HOCRConverter,
)
from .image import ImageWriter
from .layers import (
    LayeredText,
    LayerPolicy,
    LayerQueryResult,
    NativeSequenceDevice,
    SymbolicLayerPolicy,
    assignment_states,
    symbolic_page_search,
)
from .layout import LAParams, LTPage
from .pdfdevice import PDFDevice, TagExtractor
from .pdfdocument import PDFDocument
from .pdfinterp import PDFResourceManager, PDFPageInterpreter
from .pdfpage import PDFPage
from .pdfparser import PDFParser
from .sourcemap import SourceMap, SourceMappedText
from .utils import open_filename, FileOrName, AnyIO


def extract_text_to_fp(
    inf: BinaryIO,
    outfp: AnyIO,
    output_type: str = "text",
    codec: str = "utf-8",
    laparams: Optional[LAParams] = None,
    maxpages: int = 0,
    page_numbers: Optional[Container[int]] = None,
    password: str = "",
    scale: float = 1.0,
    rotation: int = 0,
    layoutmode: str = "normal",
    output_dir: Optional[str] = None,
    strip_control: bool = False,
    debug: bool = False,
    disable_caching: bool = False,
    source_map: Optional[SourceMap] = None,
    layer_policy: Optional[LayerPolicy] = None,
    **kwargs: Any,
) -> None:
    """Parses text from inf-file and writes to outfp file-like object.

    Takes loads of optional arguments but the defaults are somewhat sane.
    Beware laparams: Including an empty LAParams is not the same as passing
    None!

    :param inf: a file-like object to read PDF structure from, such as a
        file handler (using the builtin `open()` function) or a `BytesIO`.
    :param outfp: a file-like object to write the text to.
    :param output_type: May be 'text', 'xml', 'html', 'hocr', 'tag'.
        Only 'text' works properly.
    :param codec: Text decoding codec
    :param laparams: An LAParams object from pdfminer.layout. Default is None
        but may not layout correctly.
    :param maxpages: How many pages to stop parsing after
    :param page_numbers: zero-indexed page numbers to operate on.
    :param password: For encrypted PDFs, the password to decrypt.
    :param scale: Scale factor
    :param rotation: Rotation factor
    :param layoutmode: Default is 'normal', see
        pdfminer.converter.HTMLConverter
    :param output_dir: If given, creates an ImageWriter for extracted images.
    :param strip_control: Does what it says on the tin
    :param debug: Output more logging data
    :param disable_caching: Does what it says on the tin
    :param source_map: Optional SourceMap to populate for text output.  When
        ``outfp`` is a StringIO, it is bound automatically before return.
    :param other:
    :return: nothing, acting as it does on two streams. Use StringIO to get
        strings.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    imagewriter = None
    if output_dir:
        imagewriter = ImageWriter(output_dir)

    rsrcmgr = PDFResourceManager(caching=not disable_caching)
    device: Optional[PDFDevice] = None

    if output_type != "text" and outfp == sys.stdout:
        outfp = sys.stdout.buffer

    if output_type == "text":
        device = TextConverter(
            rsrcmgr,
            outfp,
            codec=codec,
            laparams=laparams,
            imagewriter=imagewriter,
            source_map=source_map,
        )
    elif output_type == "xml":
        device = XMLConverter(
            rsrcmgr,
            outfp,
            codec=codec,
            laparams=laparams,
            imagewriter=imagewriter,
            stripcontrol=strip_control,
        )

    elif output_type == "html":
        device = HTMLConverter(
            rsrcmgr,
            outfp,
            codec=codec,
            scale=scale,
            layoutmode=layoutmode,
            laparams=laparams,
            imagewriter=imagewriter,
        )

    elif output_type == "hocr":
        device = HOCRConverter(
            rsrcmgr, outfp, codec=codec, laparams=laparams, stripcontrol=strip_control
        )

    elif output_type == "tag":
        # Binary I/O is required, but we have no good way to test it here.
        device = TagExtractor(rsrcmgr, cast(BinaryIO, outfp), codec=codec)

    else:
        msg = f"Output type can be text, html, xml or tag but is " f"{output_type}"
        raise PDFValueError(msg)

    assert device is not None
    if source_map is not None and output_type != "text":
        raise PDFValueError(
            "source_map is only supported when output_type is 'text'"
        )
    if layer_policy is not None and output_type != "text":
        raise PDFValueError(
            "layer_policy is only supported when output_type is 'text'"
        )
    interpreter = PDFPageInterpreter(
        rsrcmgr, device, source_map=source_map, layer_policy=layer_policy
    )
    if layer_policy is not None and hasattr(device, "ocg_value"):
        device.ocg_value = layer_policy.all_value
    for page in PDFPage.get_pages(
        inf,
        page_numbers,
        maxpages=maxpages,
        password=password,
        caching=not disable_caching,
    ):
        page.rotate = (page.rotate + rotation) % 360
        interpreter.process_page(page)

    device.close()
    if source_map is not None and isinstance(outfp, StringIO):
        source_map.bind_text(outfp.getvalue())


def extract_text(
    pdf_file: FileOrName,
    password: str = "",
    page_numbers: Optional[Container[int]] = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: Optional[LAParams] = None,
    source_map: Optional[SourceMap] = None,
    layer_policy: Optional[LayerPolicy] = None,
) -> str:
    """Parse and return the text contained in a PDF file.

    :param pdf_file: Either a file path or a file-like object for the PDF file
        to be worked on.
    :param password: For encrypted PDFs, the password to decrypt.
    :param page_numbers: List of zero-indexed page numbers to extract.
    :param maxpages: The maximum number of pages to parse
    :param caching: If resources should be cached
    :param codec: Text decoding codec
    :param laparams: An LAParams object from pdfminer.layout. If None, uses
        some default settings that often work well.
    :param source_map: Optional SourceMap to populate with provenance for the
        returned text.  The return value remains the extracted string.
    :return: a string containing all of the text extracted.
    """
    if laparams is None:
        laparams = LAParams()

    with open_filename(pdf_file, "rb") as fp, StringIO() as output_string:
        fp = cast(BinaryIO, fp)  # we opened in binary mode
        rsrcmgr = PDFResourceManager(caching=caching)
        device = TextConverter(
            rsrcmgr,
            output_string,
            codec=codec,
            laparams=laparams,
            source_map=source_map,
        )
        interpreter = PDFPageInterpreter(
            rsrcmgr,
            device,
            source_map=source_map,
            layer_policy=layer_policy,
        )
        if layer_policy is not None and hasattr(device, "ocg_value"):
            device.ocg_value = layer_policy.all_value

        for page in PDFPage.get_pages(
            fp,
            page_numbers,
            maxpages=maxpages,
            password=password,
            caching=caching,
        ):
            interpreter.process_page(page)

        text = output_string.getvalue()
        if source_map is not None:
            source_map.bind_text(text)
        return text


def extract_text_with_sources(
    pdf_file: FileOrName,
    password: str = "",
    page_numbers: Optional[Container[int]] = None,
    maxpages: int = 0,
    caching: bool = True,
    codec: str = "utf-8",
    laparams: Optional[LAParams] = None,
    document_name: Optional[str] = None,
) -> SourceMappedText:
    """Extract text and a document-bound Unicode interval source index.

    See :class:`pdfminer.sourcemap.SourceMap` for the query API.
    """
    source_map = SourceMap(document_name=document_name)
    text = extract_text(
        pdf_file,
        password=password,
        page_numbers=page_numbers,
        maxpages=maxpages,
        caching=caching,
        codec=codec,
        laparams=laparams,
        source_map=source_map,
    )
    return SourceMappedText(text, source_map)


# Descriptive aliases used by some callers.
extract_text_sources = extract_text_with_sources
extract_text_with_source_map = extract_text_with_sources
extract_source_map = extract_text_with_sources


def _interpret_native(
    document: PDFDocument,
    policy: Any,
    device: NativeSequenceDevice,
) -> NativeSequenceDevice:
    rsrcmgr = PDFResourceManager()
    interpreter = PDFPageInterpreter(rsrcmgr, device, layer_policy=policy)
    device.ocg_value = policy.all_value
    for index, page in enumerate(PDFPage.create_pages(document)):
        page.page_index = index
        interpreter.process_page(page)
    return device


def extract_layered_text(
    pdf_file: FileOrName,
    states: Dict[int, bool],
    password: str = "",
) -> LayeredText:
    """Extract native visible Unicode per page for one concrete OCG state.

    Only native characters drawn under the given state table are returned, in
    per-page drawing order; layout-generated whitespace and page breaks do not
    participate.  The same :class:`~pdfminer.layers.LayerPolicy` drives the
    interpreter and this extractor, so hidden BDC/Form content never reaches
    the output.
    """
    with open_filename(pdf_file, "rb") as fp:
        binary = cast(BinaryIO, fp)
        document = PDFDocument(PDFParser(binary), password=password)
        policy = LayerPolicy(document, states)
        device = NativeSequenceDevice(PDFResourceManager())
        _interpret_native(document, policy, device)
    pages = tuple(
        "".join(character for character, _gate in entries)
        for entries in device.pages
    )
    return LayeredText(pages=pages, states=dict(policy.states))


def find_layer_state_for_text(
    pdf_file: FileOrName,
    target: str,
    states: Dict[int, bool],
    password: str = "",
) -> Optional[LayerQueryResult]:
    """Find the minimum-flip OCG state revealing ``target`` on one page.

    :param pdf_file: path or file-like object for an unencrypted PDF.
    :param target: non-empty native Unicode string; matching never crosses a
        page boundary.
    :param states: complete boolean state table keyed by catalog OCG indirect
        object numbers.  Missing/extra groups or invalid references raise
        :class:`ValueError`.
    :returns: a :class:`LayerQueryResult` with a complete flipped state table,
        the hit page and the flip count, or ``None`` when no assignment can
        reveal the target.  Ties are broken by the lowest assignment integer.
    """
    if not isinstance(target, str) or not target:
        raise PDFValueError("Target text must be a non-empty string")
    with open_filename(pdf_file, "rb") as fp:
        binary = cast(BinaryIO, fp)
        document = PDFDocument(PDFParser(binary), password=password)
        # Constructing the concrete policy validates the complete state table
        # up front; the symbolic policy reuses the same catalog checks below.
        LayerPolicy(document, states)
        policy = SymbolicLayerPolicy(document)
        device = NativeSequenceDevice(PDFResourceManager())
        _interpret_native(document, policy, device)
        ocg_object_numbers = policy.ocg_object_numbers

        best: Optional[Tuple[int, int, int]] = None
        for page_index, entries in enumerate(device.pages):
            feasible = symbolic_page_search(entries, target, policy)
            if not feasible:
                continue
            for assignment in feasible:
                flips = (
                    assignment
                    ^ sum(
                        (1 if states[objid] else 0) << index
                        for index, objid in enumerate(ocg_object_numbers)
                    )
                ).bit_count()
                candidate = (flips, assignment, page_index)
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            return None
        flips, assignment, page_index = best
        result_states = assignment_states(ocg_object_numbers, assignment)

        # Second, concrete interpretation of the hit page only: proves that the
        # regular extraction pipeline reproduces the target under the returned
        # state (every page/Form instance is still interpreted at most twice).
        verify_device = NativeSequenceDevice(PDFResourceManager())
        verify_policy = LayerPolicy(document, result_states)
        rsrcmgr = PDFResourceManager()
        interpreter = PDFPageInterpreter(
            rsrcmgr, verify_device, layer_policy=verify_policy
        )
        verify_device.ocg_value = verify_policy.all_value
        verify_pages = [
            page
            for index, page in enumerate(PDFPage.create_pages(document))
            if index == page_index
        ]
        assert len(verify_pages) == 1
        verify_pages[0].page_index = page_index
        interpreter.process_page(verify_pages[0])
        visible = "".join(
            character for character, _gate in verify_device.pages[0]
        )
        if target not in visible:
            # Defensive: symbolic and concrete backends share one parser.
            raise PDFValueError(
                "Layer query internal error: the selected state does not "
                "reproduce the target during concrete extraction"
            )

    return LayerQueryResult(
        states=result_states,
        page_index=page_index,
        page_number=page_index + 1,
        flips=flips,
    )


def extract_pages(
    pdf_file: FileOrName,
    password: str = "",
    page_numbers: Optional[Container[int]] = None,
    maxpages: int = 0,
    caching: bool = True,
    laparams: Optional[LAParams] = None,
    source_map: Optional[SourceMap] = None,
) -> Iterator[LTPage]:
    """Extract and yield LTPage objects

    :param pdf_file: Either a file path or a file-like object for the PDF file
        to be worked on.
    :param password: For encrypted PDFs, the password to decrypt.
    :param page_numbers: List of zero-indexed page numbers to extract.
    :param maxpages: The maximum number of pages to parse
    :param caching: If resources should be cached
    :param laparams: An LAParams object from pdfminer.layout. If None, uses
        some default settings that often work well.
    :return: LTPage objects
    """
    if laparams is None:
        laparams = LAParams()

    with open_filename(pdf_file, "rb") as fp:
        fp = cast(BinaryIO, fp)  # we opened in binary mode
        resource_manager = PDFResourceManager(caching=caching)
        device = PDFPageAggregator(resource_manager, laparams=laparams)
        interpreter = PDFPageInterpreter(
            resource_manager, device, source_map=source_map
        )
        for page in PDFPage.get_pages(
            fp, page_numbers, maxpages=maxpages, password=password, caching=caching
        ):
            interpreter.process_page(page)
            layout = device.get_result()
            yield layout
