"""
Flatten a module by moving all nested instances, ports and signals to the root level.

See function `flatten()` for details.
"""

import copy
from pydantic.dataclasses import dataclass
from dataclasses import field, replace
from typing import Any, Dict, Generator, Iterator, List, Optional

import hdl21 as h
from .signal import _copy_to_internal
from .datatype import AllowArbConfig


def _walk_conns(conns, parents=tuple()):
    for key, val in conns.items():
        if isinstance(val, dict):
            yield from _walk_conns(val, parents + (key,))
        else:
            yield parents + (key,), val


def _flat_conns(conns):
    return {":".join(reversed(path)): value.name for path, value in _walk_conns(conns)}


@dataclass(config=AllowArbConfig)
class FlattenedInstance:
    """
    # Flattened Instance
    An instance that survives flattening, because its contents are either (a) Primitive or (b) External.
    """

    inst: h.Instance
    path: List[h.Instance] = field(default_factory=list)
    conns: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Assert that this instance's target is either a primitive, or external
        if not isinstance(self.inst.of, (h.PrimitiveCall, h.ExternalModuleCall)):
            raise ValueError(f"Invalid flattened instance {self}")

    def __str__(self):
        name = self.make_name()
        path = [p.name for p in self.path]
        conns = _flat_conns(self.conns)
        return str(dict(name=name, path=path, conns=conns))

    def make_name(self):
        return ":".join([p.name or "_" for p in self.path])


def walk(
    m: h.Module,
    parents: List[h.Instance],
    conns: Optional[Dict[str, Any]] = None,
) -> Generator[FlattenedInstance, None, None]:
    if conns is None:
        conns = {**m.signals, **m.ports}
    for inst in m.instances.values():
        new_conns = {}
        new_parents = parents + [inst]
        for src_port_name, sig in inst.conns.items():
            new_conns[src_port_name] = _map_connection(sig, m, parents, conns)

        if isinstance(inst.of, h.PrimitiveCall):
            yield FlattenedInstance(inst, new_parents, new_conns)
        else:
            yield from walk(inst.of, new_parents, new_conns)


def _map_connection(conn, m, parents, conns):
    """Map one module-local connection into the flattened parent namespace."""

    if isinstance(conn, h.Signal):
        key = conn.name
        if key in conns:
            return conns[key]

        new_name = ":".join([parent.name for parent in parents] + [key])
        if key in m.signals:
            return replace(_copy_to_internal(m.signals[key]), name=new_name)
        if key in m.ports:
            return replace(_copy_to_internal(m.ports[key]), name=new_name)
        raise ValueError(f"signal {key} not found")

    if isinstance(conn, h.Slice):
        from .elab.passes.slices import _resolve_sliceable

        mapped = _map_connection(conn.parent, m, parents, conns)[conn.index]
        return _resolve_sliceable(mapped)

    if isinstance(conn, h.Concat):
        return h.Concat(
            *[_map_connection(part, m, parents, conns) for part in conn.parts]
        )

    if isinstance(conn, (h.PortRef, h.BundleInstance, h.AnonymousBundle)):
        msg = f"Error: {conn} should not have reached this stage in flattening"
        raise RuntimeError(msg)
    raise TypeError(f"Invalid connection {conn}")


def _connection_signals(conn) -> Iterator[h.Signal]:
    """Yield the concrete Signals underlying a connection."""

    if isinstance(conn, h.Signal):
        yield conn
    elif isinstance(conn, h.Slice):
        yield from _connection_signals(conn.parent)
    elif isinstance(conn, h.Concat):
        for part in conn.parts:
            yield from _connection_signals(part)
    else:
        raise TypeError(f"Invalid flattened connection {conn}")


def _rebase_connection(conn, module: h.Module):
    """Rebuild a connection from Signals owned by the flattened module."""

    if isinstance(conn, h.Signal):
        return _find_signal_or_port(module, conn.name)
    if isinstance(conn, h.Slice):
        from .elab.passes.slices import _resolve_sliceable

        rebased = _rebase_connection(conn.parent, module)[conn.index]
        return _resolve_sliceable(rebased)
    if isinstance(conn, h.Concat):
        return h.Concat(*[_rebase_connection(part, module) for part in conn.parts])
    raise TypeError(f"Invalid flattened connection {conn}")


def _find_signal_or_port(m: h.Module, name: str) -> h.Signal:
    """Find a signal or port by name"""

    port = m.ports.get(name, None)
    if port is not None:
        return port

    sig = m.signals.get(name, None)
    if sig is not None:
        return sig

    raise ValueError(f"Signal {name} not found in module {m.name}")


def is_flat(m: h.Instantiable) -> bool:
    """Boolean indication of whether `Instantiable` `m` is already flat."""

    if isinstance(m, (h.PrimitiveCall, h.ExternalModuleCall)):
        return True
    if isinstance(m, h.Module):
        instancelike = (
            list(m.instances.values())
            + list(m.instarrays.values())
            + list(m.instbundles.values())
        )
        return all(
            isinstance(inst.of, (h.PrimitiveCall, h.ExternalModuleCall))
            for inst in instancelike
        )
    raise TypeError(f"Invalid `Instantiable` argument to `is_flat`: {m}")


def flatten(m: h.Instantiable) -> h.Instantiable:
    r"""Flatten a module by moving all nested instances, ports and signals to the root level.

    For example, if we have a buffer module with two inverters, `inv_1` and `inv_2`, each with
    two transistors, `pmos` and `nmos`, the module hierarchy looks like this:
    All signals and ports will be moved to the root level too.

    ```
    buffer
    ├── inv_1
    │   ├── pmos
    │   └── nmos
    └── inv_2
        ├── pmos
        └── nmos
    ```

    This function will flatten the module to the following structure:

    ```
    buffer_flat
    ├── inv_1:pmos
    ├── inv_1:nmos
    ├── inv_2:pmos
    └── inv_2:nmos
    ```

    Nested signals and ports will be renamed and moved to the root level as well. For example, say a
    module with two buffers, `buffer_1` and `buffer_2`, each with two inverters. On the top level,
    the original ports `vdd`, `vss`, `vin` and `vout` are preserved. Nested signals and ports such
    the connection between two buffers, the internal signal between two inverters in buffer 1 will
    be renamed and moved to the root level as `buffer_1_vout`, `buffer_1:inv_1_vout`,
    `buffer_2:inv_1_vout`.

    See tests/test_flatten.py for more examples.
    """

    m = h.elaborate(m)
    if is_flat(m):
        return m

    # recursively walk the module and collect all primitive instances
    nodes: List[FlattenedInstance] = list(walk(m, parents=[]))

    # Create our new, flattened Module
    # Note that by virtue of going through elaboration above, `m.name` should be set.
    # Check for it nonetheless, and raise an error if not.
    if m.name is None:
        raise ValueError(f"Anonymous Module {m} cannot be flattened. (Give it a name.)")
    new_module = h.Module(m.name + "_flat")
    for port in m.ports.values():
        new_module.add(copy.copy(port))

    # add all signals to the root level
    for n in nodes:
        for conn in n.conns.values():
            for sig in _connection_signals(conn):
                if sig.name not in new_module.ports and sig.name not in new_module.signals:
                    new_module.add(copy.copy(sig))

    # add all connections to the root level with names resolved
    for n in nodes:
        new_inst = new_module.add(n.inst.of(), name=n.make_name())

        for src_port_name, conn in n.conns.items():
            new_inst.connect(src_port_name, _rebase_connection(conn, new_module))

    return new_module
