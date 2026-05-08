from importlib.metadata import PackageNotFoundError, version

from pyacvd.clustering import Clustering

try:
    __version__ = version("pyacvd")
except PackageNotFoundError:
    __version__ = "unknown"


# Register the ``acvd`` PyVista accessor on pyvista >= 0.48. The
# classic ``Clustering`` API works on every supported pyvista version
# (>= 0.37); the accessor is a thin convenience layer on top and is
# silently skipped on older releases.
import pyvista as _pv

if hasattr(_pv, "register_dataset_accessor"):
    from pyacvd import _accessor  # noqa: F401  (registers ``acvd``)


__all__ = ["Clustering", "__version__"]
