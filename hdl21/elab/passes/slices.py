"""
# Slice and Concatenation Resolution
"""

# Std-Lib Imports
from typing import List, Union

# Local imports
from ...module import Module
from ...signal import Signal
from ...slice import Slice
from ...concat import Concat
from ...portref import PortRef
from ...bundle import BundleRef
from ..helpers.width import width, Sliceable

# Import the base class
from .base import ElabPass


class SliceResolver(ElabPass):
    """Elaboration pass to resolve slices and concatenations to concrete signals.
    Modifies connections to any nested slices, nested concatenations, or combinations thereof.
    "Full-width" `Slice`s e.g. `sig[:]` are replaced with their parent `Signal`s.

    `Slice`s with non-unit `step` are converted to `Concat`s."""

    def elaborate_module(self, module: Module) -> Module:
        # All arrays must be flattened before getting here, or fail
        if module.instarrays:
            msg = f"Error attempting to resolve slices on {module} - "
            msg += f"still has Instance Arrays {module.instarrays}"
            raise RuntimeError(msg)

        # Then do the real work, updating any necessary connections on each instance
        for inst in module.instances.values():
            # Update all `Slice` and `Concat` valued connections to remove nested `Slice`s
            for portname, conn in inst.conns.items():
                if isinstance(conn, (Slice, Concat)):
                    resolved = _resolve_sliceable(conn)
                    inst.connect(portname, resolved)
                # All other connection-types (Signals, Interfaces) are fine

        return module


def _resolve_sliceable(conn: Sliceable) -> Sliceable:
    """Resolve a `Sliceable` to flat-concatenation-amenable elements."""
    if isinstance(conn, Signal):
        return conn  # Nothing to do
    if isinstance(conn, Slice):
        return _resolve_slice(conn)
    if isinstance(conn, Concat):
        return _resolve_concat(conn)
    if isinstance(conn, (PortRef, BundleRef)):
        return _resolve_ref(conn)
    raise TypeError(f"Invalid attempt to resolve slicing on {conn}")


def _list_slice(slize: Slice) -> List[Slice]:
    """Internal recursive helper for `resolve_slice`.
    Returns a list of Slices in which each element has a concrete Signal for its parent.
    """

    parent_bits = _list_bits(slize.parent)
    selected = parent_bits[slize.index]
    return [selected] if isinstance(selected, Slice) else list(selected)


def _list_bits(conn: Sliceable) -> List[Slice]:
    """Resolve a sliceable connection into ordered, concrete one-bit slices."""

    if isinstance(conn, Signal):
        return [conn[index] for index in range(width(conn))]
    if isinstance(conn, Slice):
        return _list_slice(conn)
    if isinstance(conn, Concat):
        return [bit for part in conn.parts for bit in _list_bits(part)]
    if isinstance(conn, (PortRef, BundleRef)):
        return _list_bits(_resolve_ref(conn))
    raise TypeError(f"Invalid attempt to resolve slicing on {conn}")


def _resolve_slice(slize: Slice) -> Sliceable:
    """Resolve a `Slice` to one or more with "concrete" `Signal`s as parents.

    Slices of other Slices and Slices of Concats are both valid design-time constructions.
    For example:
    ```python
    h.Concat(sig1, sig2, sig3)[1] # Slice of a Concat
    sig4[0:2][1] # Slice of a Slice
    ```

    While these may not frequently be created by designers, they are (at least) often created by array broadcasting.
    As some point their parents must be resolved to their original Signals, at minimum before export-level name resolution.

    Resolving Concatenations can generally resolve to more than one Slice, as in:
    ```python
    h.Concat(sig1[0], sig2[0], sig3[0])[0:1] # Requires slices of `sig1` and `sig2`
    ```
    Such cases create and return a Concatenation."""

    if isinstance(slize.parent, Signal) and slize.step == 1:
        if slize.width == slize.parent.width:
            return slize.parent
        return slize

    # Break out the slice elements in a list
    ls = _list_slice(slize)
    # And convert to either a single element or Concat
    if len(ls) == 1:  # Resolved to single Slice
        return ls[0]
    elif len(ls) > 1:  # Multiple parts required - concatenate them
        return Concat(*ls)

    raise RuntimeError(f"Error resolving Slice {slize}")


def _resolve_concat(conc: Concat) -> Concat:
    """Resolve a Concatenation into (a) concrete Signals and (b) Slices of concrete Signals.
    Removes nested concatenations and resolves slices along the way."""

    if not len(conc.parts):
        raise RuntimeError("Concatenation with no parts")

    parts = []
    for part in conc.parts:
        resolved = _resolve_sliceable(part)
        if isinstance(resolved, Concat):
            parts.extend(resolved.parts)
        else:
            parts.append(resolved)
    return Concat(*parts)


def _resolve_ref(ref: Union[PortRef, BundleRef]) -> Sliceable:
    """Resolve a reference to a Port or Bundle"""
    if ref.resolved is None:
        raise RuntimeError(f"Unresolved reference {ref}")
    return _resolve_sliceable(ref.resolved)
