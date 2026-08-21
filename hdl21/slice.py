"""
# Hdl21 Slices

References by numeric index into Signals and other Connectable types.
"""

from typing import Optional, Union, Any, Set
from weakref import WeakSet

# Local imports
from .datatype import datatype, AllowArbConfig
from .connect import connectable
from .sliceable import sliceable, is_sliceable
from .concat import concatable


@sliceable
@concatable
@connectable
@datatype(config=AllowArbConfig)
class Slice:
    """
    # Slice

    Subset of the indices of a parent `Connectable`,
    commonly Signals, Concatenations, and other Slices.
    Typically produced via square-bracket indexing into said `Connectable`s.

    Hdl21 slices are indexed "Python style", in the senses that:
    * Negative indices are supported, and count from the "end" of the Signal.
    * Slice-ranges such as `sig[0:2]` are supported, and *inclusive* of the start, while *exclusive* of the end index.
    * Negative-range slices such as `sig[2:0:-1]`, again *inclusive* of the start, *exclusive* of the end index, and *reversed*.

    Popular HDLs commonly use different signal-indexing conventions.
    Hdl21's own primary exchange format (in ProtoBuf) does as well,
    eschewing adopting inclusive-endpoints and negative-indexing.
    """

    # Parent Connectable.
    # Really of union-type `Sliceable`, which is more painful to type-check statically,
    # although the constructor does it procedurally.
    parent: Any
    # Python index, i.e. that passed to square brackets
    index: Union[int, slice]

    def __post_init__(self):
        if not is_sliceable(self.parent):
            raise TypeError(f"{self.parent} is not Sliceable")
        self._connected_ports: Set["PortRef"] = set()
        self._inner: Optional[SliceInner] = None
        self._slices: WeakSet[Slice] = set()
        self._concats: WeakSet["Concat"] = set()

    @property
    def top(self) -> int:
        return _get_inner(self).top

    @property
    def bot(self) -> int:
        return _get_inner(self).bot

    @property
    def step(self) -> int:
        return _get_inner(self).step

    @property
    def width(self) -> int:
        return _get_inner(self).width

    def __repr__(self):
        return f"Slice(parent={self.parent}, index={self.index})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Slice):
            return False
        return self.parent is other.parent and _index_key(self.index) == _index_key(
            other.index
        )

    def __hash__(self) -> bool:
        return hash((id(self.parent), _index_key(self.index)))


@datatype
class SliceInner:
    """Inner, private, resolved attributes of a `Slice`.
    Designed solely to be created by `_slice_inner` and stored as the `Slice._inner` field.
    """

    top: int  # Top index (exclusive)
    bot: int  # Bottom index (inclusive)
    step: int  # Python-convention step size
    width: int


def _index_key(index: Union[int, slice]):
    """Create an equality/ hash key without requiring the parent width."""

    if isinstance(index, int):
        return index
    return (index.start, index.stop, index.step)


def _slice_inner(slize: Slice) -> SliceInner:
    """Calculate the inner resolved fields for `slize`"""

    parent = slize.parent
    index = slize.index
    from .elab.helpers.width import width

    parent_width = width(parent)

    if isinstance(index, int):
        if index >= parent_width or index < -parent_width:
            raise ValueError(f"Out-of-bounds index {index} into {parent}")
        if index < 0:
            index += parent_width
        return SliceInner(top=index + 1, bot=index, step=1, width=1)

    if isinstance(index, slice):
        start, stop, step = index.indices(parent_width)
        indices = range(start, stop, step)
        if not indices:
            raise ValueError(f"Empty slice {index} into {parent}")
        first = indices[0]
        last = indices[-1]
        return SliceInner(
            top=max(first, last) + 1,
            bot=min(first, last),
            step=step,
            width=len(indices),
        )

    # Shouldn't be reachable, but blow up if we (somehow) get here.
    raise TypeError("Internal Error: Slice index should be an int or (python) slice")


def _get_inner(slice: Slice) -> SliceInner:
    """Get a slice's `SliceInner`, calculating it inline if necessary"""
    if slice._inner is None:
        slice._inner = _slice_inner(slice)
    return slice._inner
