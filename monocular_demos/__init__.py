"""Monocular biomechanics demos.

Importing this package first restores the ``collections`` ABC aliases that were
removed in Python 3.10 (``collections.Mapping`` and friends). Some transitive
dependencies still do ``from collections import Mapping`` at import time -- in
particular ``mujoco.mjx`` imports ``trimesh``, which imports ``networkx``, and
older ``networkx`` releases (e.g. 2.2) use the legacy alias. Restoring the
aliases here, before any submodule pulls that chain in, lets those imports
succeed on modern Python.
"""

import collections
import collections.abc

for _name in ("Mapping", "MutableMapping", "Sequence", "Iterable", "Callable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
