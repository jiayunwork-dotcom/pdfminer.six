"""Optional content group (OCG) visibility, layered extraction and queries.

PDF optional content is described in section 8.11 of the PDF 1.7 reference.
A piece of page or Form XObject content belongs to one or more optional
content groups through marked-content operators (``BDC`` with a direct
``/OC`` dictionary or a resource property name) or through a Form's ``/OC``
entry.  An optional content membership dictionary (OCMD) can combine groups
with the boolean ``/VE`` expression (``/And``, ``/Or``, ``/Not``) or, when
no expression is present, with one of the ``/P`` policies ``/AllOn``,
``/AnyOn``, ``/AllOff`` and ``/AnyOff``.

Two evaluation backends share one set of parsing rules:

* :class:`LayerPolicy` evaluates the visibility structure against a concrete
  boolean state table and drives ordinary text extraction.
* :class:`SymbolicLayerPolicy` evaluates the same structure symbolically.
  Every OCG becomes a variable of a Boolean function over at most 16 groups
  and the function is represented as a truth table stored in one Python big
  integer (bit ``a`` is the function value for the assignment whose binary
  encoding is ``a``).  One symbolic interpretation of every page and Form
  instance therefore records, for each native character in drawing order, the
  exact set of assignments that would draw it.

Search cost trade-off (see :func:`find_layer_state_for_text`): with ``n``
groups, a page/Form instance is interpreted symbolically once and a second
time only for the single hit page during concrete verification, so no stream
is ever interpreted more than twice.  Exact optimality is guaranteed because
the symbolic pass records the full Boolean function (every one of the
``2**n`` assignments) rather than probing assignments one at a time.  The
bit-parallel Shift-And automaton holds ``m * 2**n`` state bits for a target
of length ``m``; each native character costs a constant number of big-int
operations, so runtime is linear in page length, doubles per additional
group, and memory grows with the product ``m * 2**n``.  Inputs that drive the
cost up are therefore: more groups (the exponential ``2**n`` factor), long
pages combined with long targets, deeply nested ``/VE`` expressions, and
OCMDs attached to nearly every marked sequence, since each membership is
evaluated.  The cap of 16 groups keeps ``2**n`` at 65536 rows, which is why
the fixed 16-group / 4096-character fixture stays comfortably within a few
big-int words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

from .pdfdevice import PDFTextDevice
from .pdffont import PDFUnicodeNotDefined
from .pdftypes import PDFObjRef, dict_value, list_value, resolve1
from .pdfexceptions import PDFValueError
from .psparser import LIT, PSLiteral, literal_name
from .utils import Matrix

if TYPE_CHECKING:
    from .pdfdocument import PDFDocument
    from .pdfinterp import PDFGraphicState, PDFResourceManager

LITERAL_OCG = LIT("OCG")
LITERAL_OCMD = LIT("OCMD")
LITERAL_AND = LIT("And")
LITERAL_OR = LIT("Or")
LITERAL_NOT = LIT("Not")
LITERAL_ALL_ON = LIT("AllOn")
LITERAL_ANY_ON = LIT("AnyOn")
LITERAL_ALL_OFF = LIT("AllOff")
LITERAL_ANY_OFF = LIT("AnyOff")

# The truth-table representation stays manageable up to this size (2**16
# assignments per Boolean function).
MAX_OCG_GROUPS = 16


def _ocg_object_numbers(document: "PDFDocument") -> List[int]:
    """Return catalog OCG object numbers in catalog order."""
    try:
        properties = dict_value(document.catalog["OCProperties"])
    except KeyError:
        return []
    ocgs = list_value(properties.get("OCGs", []))
    result: List[int] = []
    for ref in ocgs:
        obj = resolve1(ref)
        objid = ref.objid if isinstance(ref, PDFObjRef) else getattr(obj, "objid", None)
        if objid is None:
            raise PDFValueError(
                "Optional content groups must be indirect objects in the "
                "catalog /OCProperties/OCGs array"
            )
        if not isinstance(obj, dict) or obj.get("Type") is not LITERAL_OCG:
            raise PDFValueError(
                f"Catalog OCG object {objid} is not an /OCG dictionary"
            )
        result.append(objid)
    if len(result) != len(set(result)):
        raise PDFValueError("Catalog /OCGs array contains duplicate groups")
    return result


def _reference_id(raw: Any, obj: Any) -> Optional[int]:
    if isinstance(raw, PDFObjRef):
        return raw.objid
    return getattr(obj, "objid", None)


class LayerPolicy:
    """Concrete boolean visibility policy for one document state table.

    ``states`` maps every catalog OCG indirect object number to its on/off
    value.  The table must be complete: a missing or an extra group, a group
    reference outside the catalog, a cyclic ``/VE`` expression or an
    unresolvable property name all raise :class:`ValueError` instead of
    silently treating content as visible or hidden.
    """

    #: Leaf value drawn when no optional-content gate is active.
    all_value: Any = True

    def __init__(
        self,
        document: "PDFDocument",
        states: Mapping[int, bool],
    ) -> None:
        if document.encryption is not None:
            raise PDFValueError(
                "Optional content queries are only supported for unencrypted "
                "PDFs"
            )
        self._document = document
        self.ocg_object_numbers = _ocg_object_numbers(document)
        self.ocg_index: Dict[int, int] = {
            objid: i for i, objid in enumerate(self.ocg_object_numbers)
        }
        if len(self.ocg_object_numbers) > MAX_OCG_GROUPS:
            raise PDFValueError(
                f"At most {MAX_OCG_GROUPS} optional content groups are "
                f"supported, found {len(self.ocg_object_numbers)}"
            )
        if not isinstance(states, Mapping):
            raise PDFValueError("Layer states must be a mapping of OCG id to bool")
        state_keys = set(states)
        ocg_keys: Set[int] = set(self.ocg_object_numbers)
        missing = sorted(ocg_keys - state_keys)
        extra = sorted(state_keys - ocg_keys)
        if missing:
            raise PDFValueError(
                f"Layer state table is missing OCG object number(s) {missing}"
            )
        if extra:
            raise PDFValueError(
                "Layer state table references unknown OCG object number(s) "
                f"{extra}"
            )
        for objid, value in states.items():
            if not isinstance(value, bool):
                raise PDFValueError(
                    f"Layer state for OCG {objid} must be a bool, got {value!r}"
                )
        self.states: Dict[int, bool] = dict(states)

    # -- shared structure resolution ----------------------------------

    def resolve_property(
        self, props: Any, resources: Mapping[object, object]
    ) -> Any:
        """Resolve a BDC/DP/OC property operand to a dictionary."""
        if isinstance(props, PSLiteral) or isinstance(props, str):
            name = literal_name(props) if isinstance(props, PSLiteral) else props
            properties = resources.get("Properties")
            if properties is None:
                raise PDFValueError(
                    f"Marked content property /{name} is not defined: the "
                    "resource dictionary has no /Properties entry"
                )
            resolved_props = dict_value(properties)
            props = resolved_props.get(name)
            if props is None:
                # Parsers may hand back either interned literals or plain str.
                props = resolved_props.get(LIT(name))
            if props is None:
                raise PDFValueError(
                    f"Marked content property /{name} is not listed in "
                    "/Resources/Properties"
                )
        obj = props
        if isinstance(obj, PDFObjRef):
            obj = obj.resolve()
        if isinstance(obj, dict):
            obj = self._normalise_inline_refs(obj)
        if not isinstance(obj, dict):
            raise PDFValueError(
                f"Optional content property must be a dictionary, got {obj!r}"
            )
        return obj

    def _normalise_inline_refs(self, obj: Dict[Any, Any]) -> Dict[Any, Any]:
        """Repair indirect references inside a content-stream inline dict.

        The content-stream tokenizer does not resolve ``N G R`` sequences, so
        an inline ``<< /OC 10 0 R >>`` arrives with the object number under
        ``/OC`` and the literal ``R`` keyword parked under integer key ``0``.
        Rebuild a proper :class:`PDFObjRef` when a document is available.
        """
        oc = obj.get("OC")
        if isinstance(oc, PDFObjRef) or oc is None:
            return obj
        marker = obj.get(0)
        marker_name = getattr(marker, "name", marker)
        if isinstance(oc, int) and marker_name in ("R", b"R"):
            return {**obj, "OC": PDFObjRef(self._document, oc, None)}
        return obj

    def evaluate(
        self, props: Any, resources: Mapping[object, object]
    ) -> Any:
        """Evaluate an ``/OC`` property; concrete policies return ``bool``."""
        # A direct OCG/OCMD object reference (e.g. a Form /OC entry).
        if isinstance(props, PDFObjRef):
            referenced = resolve1(props)
            if isinstance(referenced, dict):
                ref_type = referenced.get("Type")
                if ref_type is LITERAL_OCMD:
                    return self.evaluate_ocmd(referenced)
                if ref_type is LITERAL_OCG or "Type" not in referenced:
                    return self.evaluate_ocg(referenced, props)
        obj = self.resolve_property(props, resources)
        if "OC" in obj:
            # A marked-content property wrapper: << /OC <ocg-or-ocmd> >>.
            return self.evaluate_oc_entry(obj["OC"])
        type_literal = obj.get("Type")
        if type_literal is LITERAL_OCMD:
            return self.evaluate_ocmd(obj)
        if type_literal is LITERAL_OCG or "Type" not in obj:
            return self.evaluate_ocg(obj, props)
        raise PDFValueError(
            f"Unsupported optional content dictionary type: {type_literal!r}"
        )

    def evaluate_oc_entry(self, entry: Any) -> Any:
        """Evaluate the resolved value of an ``/OC`` entry."""
        if isinstance(entry, PDFObjRef):
            referenced = entry.resolve()
            if isinstance(referenced, dict):
                ref_type = referenced.get("Type")
                if ref_type is LITERAL_OCMD:
                    return self.evaluate_ocmd(referenced)
                if ref_type is LITERAL_OCG or "Type" not in referenced:
                    return self.evaluate_ocg(referenced, entry)
        if isinstance(entry, dict):
            ref_type = entry.get("Type")
            if ref_type is LITERAL_OCMD:
                return self.evaluate_ocmd(entry)
            if ref_type is LITERAL_OCG or "Type" not in entry:
                return self.evaluate_ocg(entry, entry)
        raise PDFValueError(f"Unsupported /OC entry: {entry!r}")

    def evaluate_ocg(self, obj: Mapping[str, Any], raw: Any) -> Any:
        # ``raw`` is normally the PDFObjRef carried by /OC, a /VE leaf or an
        # OCMD /OCGs array; a resolved direct dictionary falls back to its
        # own object id.
        objid = _reference_id(raw, obj)
        if objid is None and isinstance(raw, PDFObjRef):
            objid = raw.objid
        if objid not in self.ocg_index:
            raise PDFValueError(
                f"Optional content group object {objid!r} is not listed in "
                "the catalog /OCProperties/OCGs array"
            )
        return self.ocg_leaf(cast(int, objid))

    def evaluate_ocmd(self, ocmd: Mapping[str, Any]) -> Any:
        if "VE" in ocmd:
            return self.evaluate_ve(list_value(ocmd["VE"]), set())
        policy = ocmd.get("P", LITERAL_ALL_ON)
        values = []
        for ref in list_value(ocmd.get("OCGs", [])):
            resolved = resolve1(ref)
            values.append(self.evaluate_ocg(resolved, ref))
        return self.combine_by_policy(policy, cast(Sequence[Any], values))

    def combine_by_policy(
        self, policy: Any, values: Sequence[Any]
    ) -> Any:
        if policy is LITERAL_ALL_ON:
            return self.op_and(values)
        if policy is LITERAL_ANY_ON:
            return self.op_or(values)
        if policy is LITERAL_ALL_OFF:
            return self.op_and([self.op_not(value) for value in values])
        if policy is LITERAL_ANY_OFF:
            return self.op_or([self.op_not(value) for value in values])
        raise PDFValueError(f"Unsupported OCMD /P policy: {policy!r}")

    def evaluate_ve(self, node: Any, active: Set[Tuple[str, Any]]) -> Any:
        if isinstance(node, PDFObjRef):
            referenced = resolve1(node)
            if isinstance(referenced, dict) and referenced.get("Type") is LITERAL_OCMD:
                identity = ("id", id(referenced))
                if identity in active:
                    raise PDFValueError("Cyclic /VE reference through an OCMD")
                active = set(active)
                active.add(identity)
                if "VE" in referenced:
                    return self.evaluate_ve(list_value(referenced["VE"]), active)
                return self.evaluate_ocmd(referenced)
            if isinstance(referenced, dict) and referenced.get("Type") is LITERAL_OCG:
                return self.evaluate_ocg(referenced, node)
        node = resolve1(node)
        if isinstance(node, dict) and node.get("Type") is LITERAL_OCMD:
            identity = ("id", id(node))
            if identity in active:
                raise PDFValueError("Cyclic /VE reference through an OCMD")
            active = set(active)
            active.add(identity)
            if "VE" in node:
                return self.evaluate_ve(list_value(node["VE"]), active)
            return self.evaluate_ocmd(node)
        if not isinstance(node, (list, tuple)) or not node:
            raise PDFValueError(f"Malformed /VE expression: {node!r}")
        operator = node[0]
        operands = node[1:]
        if operator is LITERAL_NOT:
            if len(operands) != 1:
                raise PDFValueError("/VE /Not takes exactly one operand")
            return self.op_not(self.evaluate_ve(operands[0], active))
        if operator in (LITERAL_AND, LITERAL_OR):
            values = [self.evaluate_ve(operand, active) for operand in operands]
            if operator is LITERAL_AND:
                return self.op_and(cast(Sequence[Any], values))
            return self.op_or(cast(Sequence[Any], values))
        # A bare OCG reference leaf.
        if isinstance(node, dict) and node.get("Type") is LITERAL_OCG:
            objid = getattr(node, "objid", None)
            if objid is None:
                raise PDFValueError(
                    "OCG used inside a /VE expression must be an indirect object"
                )
            return self.evaluate_ocg(node, node)
        raise PDFValueError(f"Unsupported /VE operator or leaf: {operator!r}")

    # -- concrete boolean algebra -------------------------------------

    def ocg_leaf(self, objid: int) -> bool:
        return self.states[objid]

    def op_and(self, values: Sequence[Any]) -> bool:
        return all(cast(Sequence[bool], values))

    def op_or(self, values: Sequence[Any]) -> bool:
        return any(cast(Sequence[bool], values))

    def op_not(self, value: Any) -> bool:
        return not value


class SymbolicLayerPolicy(LayerPolicy):
    """Boolean-function backend evaluating all assignments at once.

    A value is a ``2**n``-bit truth table (Python big integer); bit ``a`` is
    the value of the function for the assignment whose little-endian binary
    encoding is ``a`` (bit ``j`` of ``a`` is the state of the OCG at catalog
    index ``j``).  No concrete state table is needed.
    """

    def __init__(self, document: "PDFDocument") -> None:
        # Bypass the concrete constructor's state-table validation while
        # reusing catalog/OCG validation.
        if document.encryption is not None:
            raise PDFValueError(
                "Optional content queries are only supported for unencrypted "
                "PDFs"
            )
        self._document = document
        self.ocg_object_numbers = _ocg_object_numbers(document)
        self.ocg_index = {
            objid: i for i, objid in enumerate(self.ocg_object_numbers)
        }
        if len(self.ocg_object_numbers) > MAX_OCG_GROUPS:
            raise PDFValueError(
                f"At most {MAX_OCG_GROUPS} optional content groups are "
                f"supported, found {len(self.ocg_object_numbers)}"
            )
        self.states: Dict[int, bool] = {}
        self.assignment_count = 1 << len(self.ocg_object_numbers)
        # Truth tables have one bit per assignment, so the all-visible constant
        # has assignment_count bits set (e.g. 4 groups -> 16 one-bits).
        self.all_value = (1 << self.assignment_count) - 1
        self._variable_masks = self._build_variable_masks(
            len(self.ocg_object_numbers)
        )
        self._ocmd_cache: Dict[int, Any] = {}

    @staticmethod
    def _build_variable_masks(n: int) -> List[int]:
        """Build the truth table for each OCG Boolean variable.

        Bit ``a`` of variable ``j``'s mask is set exactly when bit ``j`` of the
        assignment integer ``a`` is 1.  For ``n`` groups the tables hold
        ``2**n`` bits.
        """
        assignment_count = 1 << n
        masks: List[int] = []
        for j in range(n):
            mask = 0
            for assignment in range(assignment_count):
                if (assignment >> j) & 1:
                    mask |= 1 << assignment
            masks.append(mask)
        return masks

    def ocg_leaf(self, objid: int) -> int:
        return self._variable_masks[self.ocg_index[objid]]

    def op_and(self, values: Sequence[Any]) -> int:
        result = self.all_value
        for value in cast(Sequence[int], values):
            result &= value
        return result

    def op_or(self, values: Sequence[Any]) -> int:
        result = 0
        for value in cast(Sequence[int], values):
            result |= value
        return result

    def op_not(self, value: Any) -> int:
        return self.all_value ^ cast(int, value)

    def evaluate_ocmd(self, ocmd: Mapping[str, Any]) -> Any:
        objid = getattr(ocmd, "objid", None)
        if objid is not None and objid in self._ocmd_cache:
            return self._ocmd_cache[objid]
        result = super().evaluate_ocmd(ocmd)
        if objid is not None:
            self._ocmd_cache[objid] = result
        return result


class NativeSequenceDevice(PDFTextDevice):
    """Collect native Unicode characters per page in drawing order.

    Unlike :class:`~pdfminer.converter.TextConverter`, no layout analysis and
    no generated whitespace are involved: every Unicode code point emitted by
    a font through ``render_char`` is appended while the current optional
    content gate is satisfied.  The interpreter updates :attr:`ocg_value`,
    which is either a concrete ``bool`` or a symbolic truth-table mask.
    """

    def __init__(self, rsrcmgr: "PDFResourceManager") -> None:
        PDFTextDevice.__init__(self, rsrcmgr)
        self.pages: List[List[Tuple[str, Any]]] = []
        self._current: Optional[List[Tuple[str, Any]]] = None
        # Visible by default; the interpreter overwrites this whenever a
        # layer policy is installed.
        self.ocg_value: Any = True

    def begin_page(self, page: object, ctm: Matrix) -> None:
        self._current = []

    def end_page(self, page: object) -> None:
        assert self._current is not None
        self.pages.append(self._current)
        self._current = None

    def render_char(
        self,
        matrix: Matrix,
        font: object,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs: object,
        graphicstate: "PDFGraphicState",
        source: Optional[object] = None,
    ) -> float:
        if not self.ocg_value:
            return 0
        try:
            text = cast(Any, font).to_unichr(cid)
        except PDFUnicodeNotDefined:
            text = "(cid:%d)" % cid
        assert self._current is not None
        self._current.append((text, self.ocg_value))
        return 0


@dataclass(frozen=True)
class LayeredText:
    """Per-page native visible Unicode sequences for one concrete state."""

    pages: Tuple[str, ...]
    states: Dict[int, bool]

    @property
    def page_texts(self) -> Tuple[str, ...]:
        return self.pages

    def page_text(self, page_index: int) -> str:
        return self.pages[page_index]


@dataclass(frozen=True)
class LayerQueryResult:
    """A minimum-flip state table that reveals the requested target text."""

    states: Dict[int, bool]
    page_index: int
    page_number: int
    flips: int


def assignment_states(
    ocg_object_numbers: Sequence[int], assignment: int
) -> Dict[int, bool]:
    """Decode a little-endian assignment integer into a full state table."""
    return {
        objid: bool((assignment >> index) & 1)
        for index, objid in enumerate(ocg_object_numbers)
    }


def _expand_assignment_mask(
    gate: int, target_len: int, assignment_count: int
) -> int:
    """Repeat an assignment truth table across every pattern-position block."""
    mask = (1 << assignment_count) - 1
    expanded = 0
    for position in range(target_len):
        expanded |= (gate & mask) << (position * assignment_count)
    return expanded


def symbolic_page_search(
    sequence: Sequence[Tuple[str, int]],
    target: str,
    policy: SymbolicLayerPolicy,
) -> Set[int]:
    """Bit-parallel Shift-And search over every OCG assignment.

    The automaton state is one big integer partitioned into ``m``
    pattern-position blocks of ``2**n`` bits.  Block ``j`` (global bits
    ``[j*2**n, (j+1)*2**n)``) stores the Shift-And match bit for pattern
    position ``j`` and every OCG assignment simultaneously: bit ``a`` inside
    the block is set exactly when assignment ``a`` has just matched the prefix
    ending at position ``j``.  Advancing one input character therefore shifts
    by a full block (``<< 2**n``), which moves prefix matches from pattern
    position ``j`` to ``j+1`` while keeping the assignment coordinate intact.

    The cost per native character is a constant number of big-integer
    operations on ``m * 2**n`` bits.

    :returns: the set of assignment integers for which ``target`` occurs.
    """
    if not target:
        raise PDFValueError("Target text must be a non-empty string")
    assignment_count = policy.assignment_count
    target_len = len(target)
    state_width = target_len * assignment_count
    full_mask = (1 << state_width) - 1
    # The shift destination never includes pattern position zero: new
    # length-one matches are seeded there directly from the current gate.
    shift_destination_mask = full_mask & ~((1 << assignment_count) - 1)
    accept_block_start = (target_len - 1) * assignment_count
    accept_mask = ((1 << assignment_count) - 1) << accept_block_start

    # Position bits (in block coordinates) where each target character occurs.
    char_positions: Dict[str, List[int]] = {}
    for position, character in enumerate(target):
        char_positions.setdefault(character, []).append(position)

    broadcast_cache: Dict[Tuple[str, int], int] = {}

    def broadcast_for(character: str, gate: int) -> int:
        """Gate truth table repeated in every pattern block this char fills."""
        key = (character, gate)
        cached = broadcast_cache.get(key)
        if cached is not None:
            return cached
        broadcast = 0
        for position in char_positions[character]:
            broadcast |= gate << (position * assignment_count)
        broadcast_cache[key] = broadcast
        return broadcast

    visible_cache: Dict[int, int] = {}

    def visible_mask_for(gate: int) -> int:
        cached = visible_cache.get(gate)
        if cached is not None:
            return cached
        expanded = _expand_assignment_mask(gate, target_len, assignment_count)
        visible_cache[gate] = expanded
        return expanded

    seed_mask = (1 << assignment_count) - 1
    state = 0
    # Latched per-assignment bit: set once an assignment has produced a full
    # match, it stays set through every later drawing position.
    ever_matched = 0
    for character, gate in sequence:
        visible_mask = visible_mask_for(gate)
        hidden_mask = full_mask & ~visible_mask
        # Assignments for which this drawing position is hidden keep any
        # prefix match at its current pattern position.
        retained = state & hidden_mask
        if gate == 0 or character not in char_positions:
            # Hidden positions advance nothing; a visible pattern-foreign
            # character breaks open prefixes for the assignments that see it.
            state = retained
            continue
        broadcast = broadcast_for(character, gate)
        visible_transition = (
            ((state << assignment_count) & shift_destination_mask)
            | (gate & seed_mask)
        ) & broadcast
        state = visible_transition | retained
        ever_matched |= (state & accept_mask) >> accept_block_start

    accepted = ever_matched
    feasible: Set[int] = set()
    while accepted:
        low_bit = accepted & -accepted
        bit_position = low_bit.bit_length() - 1
        feasible.add(bit_position)
        accepted ^= low_bit
    return feasible
