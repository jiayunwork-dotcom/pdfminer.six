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

import threading
import weakref
from array import array
from contextlib import contextmanager
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

from .pdfexceptions import PDFValueError, SourceMapInUseError

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


class _ExtractionSession:
    """Scratch state for one extraction attempt.

    All recording during an extraction happens here.  The scratch buffers are
    only published onto the owning :class:`SourceMap` by a successful
    ``commit``; any failure rolls them back and leaves the previously
    committed index intact.
    """

    __slots__ = (
        "document_id",
        "document_name",
        "records",
        "char_sources",
        "text_parts",
        "current_page",
        "_page_count",
        "_stream_uids",
        "_next_stream_uid",
        "_call_counts",
        "_code_pool",
    )

    def __init__(self, document_id: int, document_name: Optional[str]) -> None:
        self.document_id = document_id
        self.document_name = document_name

        self.records: List[NativeSource] = []
        # One signed 32-bit record id per output Unicode code point.
        self.char_sources: "array[int]" = array("i")
        # Exact Unicode text the text converter actually wrote during this
        # extraction.  Prefix text already present in the output stream is
        # never part of it.
        self.text_parts: List[str] = []

        self._page_count = 0
        self.current_page: Optional[PageRef] = None
        self._stream_uids: "weakref.WeakKeyDictionary[PDFStream, int]" = (
            weakref.WeakKeyDictionary()
        )
        self._next_stream_uid = 1
        self._call_counts: Dict[Tuple[Any, ...], int] = {}
        self._code_pool: Dict[bytes, bytes] = {}

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


class SourceMap:
    """Document-bound index from final extracted text to native PDF sources.

    A SourceMap is reusable: each successful extraction replaces the previous
    committed result as one atomic transaction.  While an extraction is in
    progress every query keeps seeing the last committed result, and a failed
    extraction (interpreter/layout/write error or a stream short write) leaves
    that previous result untouched.
    """

    # Monotonic, process-local document binding.  It is intentionally not an
    # object id or PDF object number, both of which can be reused.
    _document_ids = count(1)

    def __init__(self, document_name: Optional[str] = None) -> None:
        # document_id/document_name describe the last committed extraction.
        # They are replaced (never reused) whenever an extraction commits.
        self.document_id = next(SourceMap._document_ids)
        self.document_name = document_name

        self.records: List[NativeSource] = []
        # One signed 32-bit record id per output Unicode code point.
        self.char_sources: "array[int]" = array("i")
        self.text: Optional[str] = None

        # Scratch state for the extraction currently in progress, if any.
        self._session: Optional[_ExtractionSession] = None
        # Held for the whole duration of an extraction so that a second
        # extraction using the same map fails before anything is written.
        self._extraction_lock = threading.Lock()

    # -- extraction transaction boundaries -----------------------------

    def begin_extraction(
        self, document_name: Optional[str] = None
    ) -> "_ExtractionSession":
        """Start a new extraction attempt on this map.

        Raises :class:`SourceMapInUseError` if an earlier extraction using the
        same map has not finished yet.  The previously committed result stays
        queryable until the new extraction commits.
        """
        if not self._extraction_lock.acquire(blocking=False):
            raise SourceMapInUseError(
                "This SourceMap is already serving an in-progress extraction"
            )
        try:
            if document_name is None:
                document_name = self.document_name
            self._session = _ExtractionSession(
                next(SourceMap._document_ids), document_name
            )
        except BaseException:
            self._extraction_lock.release()
            raise
        return self._session

    def commit_extraction(self) -> None:
        """Publish the current extraction, making its result the query view."""
        session = self._require_session()
        text = session.text
        if len(text) != len(session.char_sources):
            # A failed commit must release the map just like a rollback.
            self.rollback_extraction()
            raise PDFValueError(
                "Source map length "
                f"{len(session.char_sources)} does not match extracted text length "
                f"{len(text)}"
            )
        # Publish the scratch buffers as one step.  Queries made before this
        # point saw the previous committed result.
        self.records = session.records
        self.char_sources = session.char_sources
        self.text = text
        self.document_id = session.document_id
        self.document_name = session.document_name
        self._session = None
        self._extraction_lock.release()

    def rollback_extraction(self) -> None:
        """Discard the current extraction and keep the committed result."""
        if self._session is None:
            return
        self._session = None
        self._extraction_lock.release()

    @contextmanager
    def extraction(self, document_name: Optional[str] = None) -> Iterator[None]:
        """Context manager wrapping :meth:`begin_extraction`/commit/rollback."""
        self.begin_extraction(document_name=document_name)
        try:
            yield
        except BaseException:
            self.rollback_extraction()
            raise
        else:
            self.commit_extraction()

    def _require_session(self) -> _ExtractionSession:
        if self._session is None:
            raise PDFValueError(
                "No extraction is active on this SourceMap; call "
                "begin_extraction() first"
            )
        return self._session

    def _session_or_begin(self) -> _ExtractionSession:
        """Return the active session, lazily starting one for direct use.

        Code that drives the interpreter/converter directly (rather than
        through the high-level extraction functions) never calls
        :meth:`begin_extraction`.  Start an implicit session on the first
        recording call so that sequence still works.

        A lazily started session continues from the committed state: it
        adopts the committed records (and their document id) so that
        :class:`NativeSource` objects already attached to layout items keep
        valid record ids when a later phase indexes that layout tree without
        re-parsing the PDF.
        """
        if self._session is None:
            self.begin_extraction()
            assert self._session is not None
            session = self._session
            session.records = list(self.records)
            if self.text is not None:
                session.document_id = self.document_id
        return self._session

    # -- extraction/recording API -------------------------------------

    def begin_page(self, page: "PDFPage") -> PageRef:
        session = self._session_or_begin()
        page_index = getattr(page, "page_index", None)
        if page_index is None:
            page_index = session._page_count
        session._page_count += 1

        pageid = getattr(page, "pageid", None)
        page_object_number = pageid if isinstance(pageid, int) else None
        session.current_page = PageRef(
            document_id=session.document_id,
            page_index=int(page_index),
            page_number=int(page_index) + 1,
            page_object_number=page_object_number,
        )
        return session.current_page

    def stream_ref(
        self,
        stream: "PDFStream",
        content_index: int = 0,
        kind: Optional[str] = None,
    ) -> StreamRef:
        session = self._session_or_begin()
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

        uid = session._stream_uids.get(stream)
        if uid is None:
            uid = session._next_stream_uid
            session._next_stream_uid += 1
            session._stream_uids[stream] = uid

        if kind is None:
            kind = "form-xobject" if session.current_page is not None else "content"
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
        session = self._session_or_begin()
        from .pdftypes import PDFObjRef, resolve1

        obj = resolve1(xobject) if isinstance(xobject, PDFObjRef) else xobject
        object_number = getattr(obj, "objid", None)
        generation = getattr(obj, "genno", None)

        assert session.current_page is not None
        identity: Any
        if object_number is not None:
            identity = ("obj", object_number, generation)
        else:
            identity = ("uid", self.stream_ref(obj).uid)

        key = (
            session.current_page.page_index,
            tuple(parent_path),
            identity,
        )
        call_index = session._call_counts.get(key, 0)
        session._call_counts[key] = call_index + 1

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
        session = self._session_or_begin()
        assert session.current_page is not None
        stream_ref = self.stream_ref(
            stream,
            content_index=content_index,
            kind="page-contents" if not path else "form-xobject",
        )
        return OperatorRef(
            page=session.current_page,
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
        session = self._session_or_begin()
        # Intern repeated code byte strings: this stores one small bytes object
        # per distinct code rather than allocating a copy for every glyph.
        pooled_code = session._code_pool.setdefault(code, code)
        record = NativeSource(
            record_id=len(session.records),
            document_id=session.document_id,
            operator_ref=operator_ref,
            operand_index=operand_index,
            string_index=string_index,
            byte_start=byte_start,
            byte_end=byte_end,
            cid=cid,
            code=pooled_code,
        )
        session.records.append(record)
        return record

    def append_output(self, text: str, source: Any = None) -> None:
        """Record one exact converter text write.

        ``text`` must be the Unicode string actually handed to the output
        stream (after codec filtering); layout-generated characters pass
        ``source=None`` and are marked as having no native origin.
        """
        if not text:
            return
        session = self._session_or_begin()
        record_id = -1
        if isinstance(source, NativeSource):
            record_id = source.record_id
        elif isinstance(source, int):
            record_id = source
        if record_id < -1 or record_id >= len(session.records):
            raise PDFValueError(f"Source record {record_id} is not in this SourceMap")
        session.char_sources.extend(array("i", [record_id]) * len(text))
        session.text_parts.append(text)

    def bind_text(self, text: str) -> None:
        """Bind text for a direct, non-high-level extraction and commit it.

        High-level extraction accumulates the converter's writes itself; this
        entry point supports callers that drive the interpreter and
        :class:`~pdfminer.converter.TextConverter` directly.  A length
        mismatch rolls the attempt back instead of leaving the map busy.
        """
        session = self._session_or_begin()
        if len(text) != len(session.char_sources):
            self.rollback_extraction()
            raise PDFValueError(
                "Source map length "
                f"{len(session.char_sources)} does not match extracted text length "
                f"{len(text)}"
            )
        session.text_parts = [text]
        self.commit_extraction()

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
        session = self._session_or_begin()
        session.char_sources = array("i")
        session.text_parts = []
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
        session = self._session_or_begin()
        session.char_sources = array("i")
        session.text_parts = []
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
