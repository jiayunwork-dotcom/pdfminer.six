"""Queryable provenance for text extracted from PDFs.

The source map is intentionally organised as a compact character index plus a
list of unique native source records.  Every output Unicode code point stores a
four-byte record id (``-1`` means layout/converter generated text).  A source
record is shared by all Unicode code points produced from one PDF character
code.  Operator, stream and page descriptors are themselves shared by all
codes in one text-showing operation.

The structures contain only plain data.  In particular they do not retain the
PDF file object, parser, document or decoded content-stream bytes, so query
results remain available after the input file is closed.
"""

from __future__ import annotations

import weakref
from array import array
from dataclasses import dataclass
from itertools import count
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from .pdfexceptions import PDFValueError

if TYPE_CHECKING:
    from .layout import LTItem
    from .pdfpage import PDFPage
    from .pdftypes import PDFStream


@dataclass(frozen=True)
class PageRef:
    """The page on which a drawing operation was executed."""

    __slots__ = ("document_id", "page_index", "page_number", "page_object_number")

    document_id: int
    page_index: int
    page_number: int
    page_object_number: Optional[int]


@dataclass(frozen=True)
class SourceCall:
    """One Form XObject invocation in a content-stream call path.

    ``call_index`` counts invocations of the same XObject from the immediate
    parent stream.  ``call_order`` counts all ``Do`` invocations in that
    immediate parent stream.  The complete path distinguishes the same Form
    reached through different nested or repeated calls.
    """

    __slots__ = (
        "name",
        "object_number",
        "generation",
        "call_index",
        "call_order",
    )

    name: Optional[str]
    object_number: Optional[int]
    generation: Optional[int]
    call_index: int
    call_order: int


@dataclass(frozen=True)
class StreamRef:
    """A document-local reference to one content-stream object."""

    __slots__ = (
        "uid",
        "object_number",
        "generation",
        "content_index",
        "kind",
    )

    uid: int
    object_number: Optional[int]
    generation: Optional[int]
    content_index: int
    kind: str


@dataclass(frozen=True)
class OperatorRef:
    """Shared location of one text-showing PDF operator."""

    __slots__ = ("page", "path", "stream", "operator", "operator_index")

    page: PageRef
    path: Tuple[SourceCall, ...]
    stream: StreamRef
    operator: str
    operator_index: int


@dataclass(frozen=True)
class NativeSource:
    """Exact native origin of one decoded PDF character code.

    ``byte_start`` and ``byte_end`` are offsets in the decoded/unescaped bytes
    of the PDF string operand.  The interval is half-open.  When one code maps
    to several Unicode characters, all of those output characters refer to the
    same ``NativeSource`` and therefore the same complete code interval.
    """

    __slots__ = (
        "record_id",
        "document_id",
        "operator_ref",
        "operand_index",
        "string_index",
        "byte_start",
        "byte_end",
        "cid",
        "code",
    )

    record_id: int
    document_id: int
    operator_ref: OperatorRef
    operand_index: int
    string_index: int
    byte_start: int
    byte_end: int
    cid: Optional[int]
    code: bytes

    @property
    def page(self) -> PageRef:
        return self.operator_ref.page

    @property
    def page_index(self) -> int:
        return self.operator_ref.page.page_index

    @property
    def page_number(self) -> int:
        return self.operator_ref.page.page_number

    @property
    def page_object_number(self) -> Optional[int]:
        return self.operator_ref.page.page_object_number

    @property
    def path(self) -> Tuple[SourceCall, ...]:
        return self.operator_ref.path

    @property
    def stream(self) -> StreamRef:
        return self.operator_ref.stream

    @property
    def stream_object_number(self) -> Optional[int]:
        return self.operator_ref.stream.object_number

    @property
    def content_stream_object_number(self) -> Optional[int]:
        return self.operator_ref.stream.object_number

    @property
    def op_index(self) -> int:
        return self.operator_ref.operator_index

    @property
    def operator_number(self) -> int:
        """One-based form of :attr:`operator_index`."""
        return self.operator_ref.operator_index + 1

    @property
    def start(self) -> int:
        return self.byte_start

    @property
    def end(self) -> int:
        return self.byte_end

    @property
    def byte_range(self) -> Tuple[int, int]:
        return (self.byte_start, self.byte_end)

    @property
    def operator(self) -> str:
        return self.operator_ref.operator

    @property
    def operator_index(self) -> int:
        return self.operator_ref.operator_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "document_id": self.document_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "page_object_number": self.page_object_number,
            "path": [
                {
                    "name": call.name,
                    "object_number": call.object_number,
                    "generation": call.generation,
                    "call_index": call.call_index,
                    "call_order": call.call_order,
                }
                for call in self.path
            ],
            "stream": {
                "uid": self.stream.uid,
                "object_number": self.stream.object_number,
                "generation": self.stream.generation,
                "content_index": self.stream.content_index,
                "kind": self.stream.kind,
            },
            "operator": self.operator,
            "operator_index": self.operator_index,
            "operand_index": self.operand_index,
            "string_index": self.string_index,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "cid": self.cid,
            "code_hex": self.code.hex(),
            "code_latin1": self.code.decode("latin1", errors="replace"),
        }


class SourceInterval(NamedTuple):
    """A run of output code points sharing one source record."""

    start: int
    end: int
    text: str
    source: Optional[NativeSource]

    @property
    def native(self) -> Optional[NativeSource]:
        return self.source

    @property
    def is_native(self) -> bool:
        return self.source is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "source": None if self.source is None else self.source.to_dict(),
        }


class CharacterSource(NamedTuple):
    """Provenance for exactly one output Unicode code point."""

    index: int
    char: str
    source: Optional[NativeSource]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "char": self.char,
            "source": None if self.source is None else self.source.to_dict(),
        }


class SourceMap:
    """Document-bound index from final extracted text to native PDF sources."""

    # Monotonic, process-local document binding.  It is intentionally not an
    # object id or PDF object number, both of which can be reused.
    _document_ids = count(1)

    def __init__(self, document_name: Optional[str] = None) -> None:
        self.document_id = next(SourceMap._document_ids)
        self.document_name = document_name

        self.records: List[NativeSource] = []
        # One signed 32-bit record id per output Unicode code point.
        self.char_sources: "array[int]" = array("i")
        self.text: Optional[str] = None

        self._page_count = 0
        self.current_page: Optional[PageRef] = None
        self._stream_uids: "weakref.WeakKeyDictionary[PDFStream, int]" = (
            weakref.WeakKeyDictionary()
        )
        self._next_stream_uid = 1
        self._call_counts: Dict[Tuple[Any, ...], int] = {}
        self._code_pool: Dict[bytes, bytes] = {}

    # -- extraction/recording API -------------------------------------

    def begin_page(self, page: "PDFPage") -> PageRef:
        page_index = getattr(page, "page_index", None)
        if page_index is None:
            page_index = self._page_count
        self._page_count += 1

        pageid = getattr(page, "pageid", None)
        page_object_number = pageid if isinstance(pageid, int) else None
        self.current_page = PageRef(
            document_id=self.document_id,
            page_index=int(page_index),
            page_number=int(page_index) + 1,
            page_object_number=page_object_number,
        )
        return self.current_page

    def stream_ref(
        self,
        stream: "PDFStream",
        content_index: int = 0,
        kind: Optional[str] = None,
    ) -> StreamRef:
        # Resolve an indirect reference if a caller passed one.  Imported here
        # to avoid a circular import at module import time.
        from .pdftypes import PDFObjRef, PDFStream, resolve1

        if isinstance(stream, PDFObjRef):
            stream = resolve1(stream)
        if not isinstance(stream, PDFStream):
            # Content should always be a stream, but keep source tracking
            # non-fatal for unusual fixtures.
            return StreamRef(
                uid=-1,
                object_number=None,
                generation=None,
                content_index=content_index,
                kind=kind or "unknown",
            )

        uid = self._stream_uids.get(stream)
        if uid is None:
            uid = self._next_stream_uid
            self._next_stream_uid += 1
            self._stream_uids[stream] = uid

        if kind is None:
            kind = "form-xobject" if self.current_page is not None else "content"
        return StreamRef(
            uid=uid,
            object_number=stream.objid,
            generation=stream.genno,
            content_index=content_index,
            kind=kind,
        )

    def make_call(
        self,
        *,
        parent_path: Sequence[SourceCall],
        name: str,
        xobject: "PDFStream",
        call_order: int,
    ) -> SourceCall:
        from .pdftypes import PDFObjRef, resolve1

        obj = resolve1(xobject) if isinstance(xobject, PDFObjRef) else xobject
        object_number = getattr(obj, "objid", None)
        generation = getattr(obj, "genno", None)

        assert self.current_page is not None
        identity: Any
        if object_number is not None:
            identity = ("obj", object_number, generation)
        else:
            identity = ("uid", self.stream_ref(obj).uid)

        key = (
            self.current_page.page_index,
            tuple(parent_path),
            identity,
        )
        call_index = self._call_counts.get(key, 0)
        self._call_counts[key] = call_index + 1

        return SourceCall(
            name=name,
            object_number=object_number,
            generation=generation,
            call_index=call_index,
            call_order=call_order,
        )

    def begin_operator(
        self,
        *,
        path: Sequence[SourceCall],
        stream: "PDFStream",
        content_index: int,
        operator: str,
        operator_index: int,
    ) -> OperatorRef:
        assert self.current_page is not None
        stream_ref = self.stream_ref(
            stream,
            content_index=content_index,
            kind="page-contents" if not path else "form-xobject",
        )
        return OperatorRef(
            page=self.current_page,
            path=tuple(path),
            stream=stream_ref,
            operator=operator,
            operator_index=operator_index,
        )

    def add_native(
        self,
        operator_ref: OperatorRef,
        *,
        operand_index: int,
        string_index: int,
        byte_start: int,
        byte_end: int,
        cid: Optional[int],
        code: bytes,
    ) -> NativeSource:
        # Intern repeated code byte strings: this stores one small bytes object
        # per distinct code rather than allocating a copy for every glyph.
        pooled_code = self._code_pool.setdefault(code, code)
        record = NativeSource(
            record_id=len(self.records),
            document_id=self.document_id,
            operator_ref=operator_ref,
            operand_index=operand_index,
            string_index=string_index,
            byte_start=byte_start,
            byte_end=byte_end,
            cid=cid,
            code=pooled_code,
        )
        self.records.append(record)
        return record

    def append_output(self, text: str, source: Any = None) -> None:
        """Append the source ids for one exact converter text write."""
        if not text:
            return
        record_id = -1
        if isinstance(source, NativeSource):
            record_id = source.record_id
        elif isinstance(source, int):
            record_id = source
        if record_id < -1 or record_id >= len(self.records):
            raise PDFValueError(f"Source record {record_id} is not in this SourceMap")
        self.char_sources.extend(array("i", [record_id]) * len(text))

    def bind_text(self, text: str) -> None:
        """Bind the final Unicode text produced by the text converter."""
        if len(text) != len(self.char_sources):
            raise PDFValueError(
                "Source map length "
                f"{len(self.char_sources)} does not match extracted text length "
                f"{len(text)}"
            )
        self.text = text

    # -- layout-tree indexing -----------------------------------------

    def append_layout(
        self,
        root: "LTItem",
        *,
        page_break: bool = True,
        page_number_text: Optional[str] = None,
    ) -> str:
        """Append one layout tree using TextConverter's traversal order."""
        from io import StringIO

        from .layout import LTChar, LTContainer, LTText, LTTextBox

        out = StringIO()

        def emit(text: str, source: Any = None) -> None:
            out.write(text)
            self.append_output(text, source)

        if page_number_text is not None:
            emit(page_number_text, None)

        def render(item: "LTItem") -> None:
            if isinstance(item, LTContainer):
                for child in item:
                    render(child)
            elif isinstance(item, LTText):
                text = item.get_text()
                source = item.source if isinstance(item, LTChar) else None
                emit(text, source)
            if isinstance(item, LTTextBox):
                emit("\n", None)

        render(root)
        if page_break:
            emit("\f", None)
        return out.getvalue()

    def index_layout(
        self,
        root: "LTItem",
        *,
        page_break: bool = True,
        page_number_text: Optional[str] = None,
    ) -> str:
        """Reset output and index one layout tree in converter traversal order."""
        self.char_sources = array("i")
        text = self.append_layout(
            root,
            page_break=page_break,
            page_number_text=page_number_text,
        )
        self.bind_text(text)
        return text

    def index_layouts(
        self,
        roots: Sequence["LTItem"],
        *,
        page_break: bool = True,
    ) -> str:
        """Index several pages and bind their concatenated converter text."""
        self.char_sources = array("i")
        parts = [
            self.append_layout(root, page_break=page_break) for root in roots
        ]
        text = "".join(parts)
        self.bind_text(text)
        return text

    # -- query API -----------------------------------------------------

    def _require_text(self) -> str:
        if self.text is None:
            raise PDFValueError(
                "SourceMap is not bound to extracted text; call bind_text() first"
            )
        return self.text

    def source_at(self, index: int) -> Optional[NativeSource]:
        text = self._require_text()
        if index < 0:
            index += len(text)
        if index < 0 or index >= len(text):
            raise IndexError("Source index is outside the extracted text")
        record_id = self.char_sources[index]
        if record_id < 0:
            return None
        return self.records[record_id]

    def lookup(
        self,
        start: int = 0,
        end: Optional[int] = None,
        *,
        native_only: bool = False,
    ) -> List[SourceInterval]:
        """Return source runs for the Unicode half-open interval ``[start, end)``.

        ``end=None`` means the end of the extracted text.  Runs of consecutive
        output code points with the same source record are grouped.  Generated
        spaces/newlines have ``source is None`` and are retained unless
        ``native_only=True``.
        """
        text = self._require_text()
        n = len(text)
        if end is None:
            end = n
        if not (0 <= start <= end <= n):
            raise IndexError(f"Source interval [{start}, {end}) is outside [0, {n})")

        result: List[SourceInterval] = []
        i = start
        while i < end:
            record_id = self.char_sources[i]
            j = i + 1
            while j < end and self.char_sources[j] == record_id:
                j += 1
            source = None if record_id < 0 else self.records[record_id]
            if source is not None or not native_only:
                result.append(
                    SourceInterval(
                        start=i,
                        end=j,
                        text=text[i:j],
                        source=source,
                    )
                )
            i = j
        return result

    def characters(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ) -> List[CharacterSource]:
        """Return one exact mapping per Unicode code point in ``[start, end)``."""
        text = self._require_text()
        n = len(text)
        if end is None:
            end = n
        if not (0 <= start <= end <= n):
            raise IndexError(f"Source interval [{start}, {end}) is outside [0, {n})")
        return [
            CharacterSource(index=i, char=text[i], source=self.source_at(i))
            for i in range(start, end)
        ]

    def query(
        self,
        start: int = 0,
        end: Optional[int] = None,
        *,
        native_only: bool = False,
    ) -> List[SourceInterval]:
        """Alias for :meth:`lookup`."""
        return self.lookup(start, end, native_only=native_only)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            text = self._require_text()
            start, stop, step = key.indices(len(text))
            if step != 1:
                raise IndexError("SourceMap slices do not support a step")
            return self.lookup(start, stop)
        if isinstance(key, int):
            return self.source_at(key)
        raise TypeError("SourceMap index must be an int or slice")

    def iter_native(self) -> Iterator[SourceInterval]:
        yield from self.lookup(native_only=True)

    def to_list(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.lookup()]


class SourceMappedText:
    """Convenience result returned by :func:`extract_text_with_sources`."""

    __slots__ = ("text", "sources")

    def __init__(self, text: str, sources: SourceMap) -> None:
        self.text = text
        self.sources = sources

    @property
    def source_map(self) -> SourceMap:
        """Alias for :attr:`sources`."""
        return self.sources

    def lookup(
        self,
        start: int = 0,
        end: Optional[int] = None,
        *,
        native_only: bool = False,
    ) -> List[SourceInterval]:
        return self.sources.lookup(start, end, native_only=native_only)

    def source_at(self, index: int) -> Optional[NativeSource]:
        return self.sources.source_at(index)

    def characters(
        self, start: int = 0, end: Optional[int] = None
    ) -> List[CharacterSource]:
        return self.sources.characters(start, end)

    def query(
        self,
        start: int = 0,
        end: Optional[int] = None,
        *,
        native_only: bool = False,
    ) -> List[SourceInterval]:
        return self.sources.lookup(start, end, native_only=native_only)

    def __getitem__(self, key: Any) -> Any:
        return self.sources[key]

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return (
            f"<SourceMappedText {len(self.text)} chars, "
            f"{len(self.sources.records)} sources>"
        )
